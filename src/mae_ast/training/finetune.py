"""
Fine-tuning supervisionato di MAE-AST.

Durante il fine-tuning:

- il decoder non viene utilizzato;
- reconstruction head e pretraining classification head
  vengono congelate;
- l'encoder processa tutte le patch;
- viene applicato mean pooling;
- la finetune head produce le classi downstream.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mae_ast.losses import finetune_loss
from mae_ast.models.mae_ast import MAEASTModel
from mae_ast.training.tracking import build_experiment_tracker
from mae_ast.training.utils import (
    AverageMeter,
    RuntimeMetrics,
    amp_enabled,
    append_jsonl,
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    get_peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    save_checkpoint,
    scheduler_step_unit,
    sync_device,
)


def freeze_pretraining_modules(
        model: MAEASTModel,
) -> None:
    """
    Congela i moduli utilizzati esclusivamente
    durante il pretraining.
    """

    modules_to_freeze = [
        model.decoder,
        model.decoder_norm,
        model.reconstruction_head,
        model.pretrain_classification_head,
    ]

    if isinstance(
            model.encoder_to_decoder,
            torch.nn.Module,
    ):
        modules_to_freeze.append(
            model.encoder_to_decoder
        )

    for module in modules_to_freeze:
        for parameter in module.parameters():
            parameter.requires_grad = False

    model.mask_token.requires_grad = False


def trainable_parameters(
        model: torch.nn.Module,
):
    """
    Restituisce solamente i parametri realmente trainabili.
    """

    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def load_pretrained_weights(
        model: MAEASTModel,
        checkpoint_path: str | Path,
        device: torch.device,
) -> None:
    """
    Carica i pesi ottenuti dal pretraining.

    La finetune head viene esclusa perché deve essere
    inizializzata per il task downstream.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(
            "finetune_head."
        )
    }

    missing, unexpected = (
        model.load_state_dict(
            filtered_state_dict,
            strict=False,
        )
    )

    print(
        f"Pesi pretrained caricati da: "
        f"{checkpoint_path}"
    )

    print(
        f"Missing keys: {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
    )


def accuracy(
        logits: torch.Tensor,
        labels: torch.Tensor,
) -> float:
    """
    Calcola la classification accuracy.
    """

    predictions = logits.argmax(
        dim=1
    )

    return float(
        (
                predictions == labels
        )
        .float()
        .mean()
        .item()
    )


def train_finetune_epoch(
        model: MAEASTModel,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        device: torch.device,
        cfg: DictConfig,
        global_step: int,
) -> tuple[
    dict[str, float],
    int,
    bool,
    RuntimeMetrics,
]:
    """
    Esegue una singola epoca di fine-tuning.
    """

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    step_time_meter = AverageMeter()

    num_examples = 0

    max_steps = (
        None
        if cfg.training.max_steps is None
        else int(cfg.training.max_steps)
    )

    grad_clip_norm = (
        None
        if cfg.training.grad_clip_norm is None
        else float(
            cfg.training.grad_clip_norm
        )
    )

    use_amp = amp_enabled(
        cfg,
        device,
    )

    scheduler_per_batch = (
        scheduler_step_unit(cfg) == "batch"
    )

    reset_peak_gpu_memory(
        device
    )

    epoch_start = (
        time.perf_counter()
    )

    stop_training = False

    for batch in dataloader:

        if (
                max_steps is not None
                and global_step >= max_steps
        ):
            stop_training = True
            break

        spectrogram = batch[
            "spectrogram"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        batch_size = (
            spectrogram.shape[0]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        sync_device(device)

        step_start = (
            time.perf_counter()
        )

        with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
        ):
            output = (
                model.forward_finetune(
                    spectrogram
                )
            )

            losses = finetune_loss(
                output=output,
                labels=labels,
                label_smoothing=float(
                    cfg.loss.label_smoothing
                ),
            )

        if scaler is not None:

            scaler.scale(
                losses.total
            ).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(
                        model
                    ),
                    grad_clip_norm,
                )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            losses.total.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(
                        model
                    ),
                    grad_clip_norm,
                )

            optimizer.step()

        if scheduler_per_batch:
            scheduler.step()

        sync_device(device)

        step_time = (
                time.perf_counter()
                - step_start
        )

        global_step += 1
        num_examples += batch_size

        batch_accuracy = accuracy(
            output.logits.detach(),
            labels,
        )

        loss_meter.update(
            float(
                losses.total
                .detach()
                .item()
            ),
            batch_size,
        )

        accuracy_meter.update(
            batch_accuracy,
            batch_size,
        )

        step_time_meter.update(
            step_time
        )

        if (
                global_step
                % int(
            cfg.training.log_interval
        )
                == 0
        ):
            print(
                "[FINETUNE] "
                f"step={global_step} "
                f"loss={loss_meter.average:.5f} "
                f"acc={accuracy_meter.average:.4f}"
            )

    elapsed = (
            time.perf_counter()
            - epoch_start
    )

    runtime = RuntimeMetrics(
        elapsed_sec=elapsed,

        step_time_sec_avg=(
            step_time_meter.average
        ),

        examples_per_sec=(
                num_examples
                / max(elapsed, 1e-12)
        ),

        peak_gpu_mem_mb=(
            get_peak_gpu_memory_mb(
                device
            )
        ),
    )

    return (
        {
            "loss":
                loss_meter.average,

            "accuracy":
                accuracy_meter.average,
        },
        global_step,
        stop_training,
        runtime,
    )


@torch.no_grad()
def validate_finetune(
        model: MAEASTModel,
        dataloader: DataLoader,
        device: torch.device,
        cfg: DictConfig,
) -> dict[str, float]:
    """
    Valutazione supervisionata.
    """

    model.eval()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    use_amp = amp_enabled(
        cfg,
        device,
    )

    for batch in dataloader:

        spectrogram = batch[
            "spectrogram"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
        ):
            output = (
                model.forward_finetune(
                    spectrogram
                )
            )

            losses = finetune_loss(
                output=output,
                labels=labels,
                label_smoothing=float(
                    cfg.loss.label_smoothing
                ),
            )

        batch_size = (
            spectrogram.shape[0]
        )

        loss_meter.update(
            float(
                losses.total.item()
            ),
            batch_size,
        )

        accuracy_meter.update(
            accuracy(
                output.logits,
                labels,
            ),
            batch_size,
        )

    return {
        "loss":
            loss_meter.average,

        "accuracy":
            accuracy_meter.average,
    }


def run_finetuning(
        model: MAEASTModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
        output_dir: str | Path,
        pretrained_checkpoint: str | Path | None = None,
) -> dict:
    """
    Gestisce l'intero fine-tuning.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model.to(device)

    if pretrained_checkpoint is not None:
        load_pretrained_weights(
            model=model,
            checkpoint_path=pretrained_checkpoint,
            device=device,
        )

    freeze_pretraining_modules(
        model
    )

    parameters = trainable_parameters(
        model
    )

    print(
        "Parametri trainabili: "
        f"{sum(p.numel() for p in parameters):,}"
    )

    optimizer = build_optimizer(
        parameters,
        cfg,
    )

    print(
        "Optimizer: "
        f"{optimizer.__class__.__name__} "
        f"lr={optimizer.param_groups[0]['lr']:.6g}"
    )

    if cfg.training.max_steps is not None:
        total_steps = int(
            cfg.training.max_steps
        )
    else:
        total_steps = (
                int(cfg.training.epochs)
                * len(train_loader)
        )

    scheduler = build_scheduler(
        optimizer,
        cfg,
        total_steps,
    )

    print(
        "Scheduler: "
        f"{scheduler.__class__.__name__} "
        f"step={scheduler_step_unit(cfg)}"
    )

    scaler = build_grad_scaler(
        amp_enabled(
            cfg,
            device,
        )
    )

    global_step = 0
    best_val_accuracy = 0.0

    log_path = (
            output_dir
            / "finetune_log.jsonl"
    )

    best_path = (
            output_dir
            / "best.pt"
    )

    last_path = (
            output_dir
            / "last.pt"
    )

    tracker = build_experiment_tracker(
        cfg=cfg,
        output_dir=output_dir,
        job_type="finetune",
    )

    for epoch in range(
            1,
            int(cfg.training.epochs) + 1,
    ):

        print(
            f"\nFine-tuning epoch "
            f"{epoch}/{cfg.training.epochs}"
        )

        train_metrics, global_step, stop_training, runtime = (
            train_finetune_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                cfg=cfg,
                global_step=global_step,
            )
        )

        val_metrics = validate_finetune(
            model=model,
            dataloader=val_loader,
            device=device,
            cfg=cfg,
        )

        print(
            "[VAL] "
            f"loss={val_metrics['loss']:.5f} "
            f"acc={val_metrics['accuracy']:.4f}"
        )

        append_jsonl(
            log_path,
            {
                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "train":
                    train_metrics,

                "validation":
                    val_metrics,

                "runtime": {
                    "epoch_time_sec":
                        runtime.elapsed_sec,

                    "step_time_sec_avg":
                        runtime.step_time_sec_avg,

                    "examples_per_sec":
                        runtime.examples_per_sec,

                    "peak_gpu_mem_mb":
                        runtime.peak_gpu_mem_mb,
                },
            },
        )

        tracker_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "optimization/learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "runtime/epoch_time_sec": runtime.elapsed_sec,
            "runtime/step_time_sec_avg": runtime.step_time_sec_avg,
            "runtime/examples_per_sec": runtime.examples_per_sec,
        }

        if runtime.peak_gpu_mem_mb is not None:
            tracker_metrics[
                "runtime/peak_gpu_mem_mb"
            ] = runtime.peak_gpu_mem_mb

        tracker.log(
            tracker_metrics,
            step=global_step,
        )

        # La recipe ESC-50 usa uno scheduler a epoche: avanza
        # solamente dopo avere concluso training + validation dell'epoca.
        if scheduler_step_unit(cfg) == "epoch":
            scheduler.step()

        if (
                val_metrics["accuracy"]
                > best_val_accuracy
        ):
            best_val_accuracy = (
                val_metrics[
                    "accuracy"
                ]
            )

            if cfg.checkpoint.save_best:
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    cfg=cfg,
                    best_metric=best_val_accuracy,
                    best_metric_name="val_accuracy",
                )

        if cfg.checkpoint.save_last:
            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                best_metric=best_val_accuracy,
                best_metric_name="val_accuracy",
            )

        if stop_training:
            break

    tracker.log(
        {
            "best/val_accuracy": best_val_accuracy,
            "global_step": global_step,
        },
        step=global_step,
    )

    tracker.finish()

    return {
        "best_val_accuracy":
            best_val_accuracy,

        "global_step":
            global_step,

        "output_dir":
            str(output_dir),
    }