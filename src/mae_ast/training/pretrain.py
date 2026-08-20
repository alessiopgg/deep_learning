"""
Loop di pretraining self-supervised per MAE-AST.

Questo modulo non costruisce dataset o modelli.
Riceve componenti già creati e si occupa solamente di:

- forward;
- loss;
- backward;
- optimizer;
- scheduler;
- validation;
- logging;
- checkpoint.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mae_ast.losses import pretrain_loss
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
    load_training_checkpoint,
    reset_peak_gpu_memory,
    save_checkpoint,
    sync_device,
)


def resolve_pretraining_total_epochs(
        cfg: DictConfig,
        steps_per_epoch: int,
) -> int:
    """
    Determina quante epoche deve poter attraversare il loop di pretraining.

    Se ``training.max_steps`` è valorizzato, il numero di update diventa
    l'orizzonte primario della run e ``training.epochs`` non limita più il
    training. Il numero di epoche viene quindi derivato automaticamente.

    Se ``max_steps`` è ``null``, resta valido il comportamento storico basato
    su ``training.epochs``.
    """

    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch deve essere maggiore di zero."
        )

    configured_max_steps = cfg.training.get(
        "max_steps",
        None,
    )

    if configured_max_steps is None:
        total_epochs = int(
            cfg.training.epochs
        )

        if total_epochs <= 0:
            raise ValueError(
                "training.epochs deve essere maggiore di zero."
            )

        return total_epochs

    max_steps = int(
        configured_max_steps
    )

    if max_steps <= 0:
        raise ValueError(
            "training.max_steps deve essere maggiore di zero quando impostato."
        )

    return max(
        1,
        math.ceil(
            max_steps / steps_per_epoch
        ),
    )


def train_pretrain_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        device: torch.device,
        cfg: DictConfig,
        global_step: int,
        start_batch_in_epoch: int = 0,
        epoch_seed: int | None = None,
        periodic_checkpoint_callback: Callable[[int, int], None] | None = None,
) -> tuple[
    dict[str, float],
    int,
    bool,
    RuntimeMetrics,
    int,
    bool,
]:
    """
    Esegue una singola epoca di pretraining.

    ``start_batch_in_epoch`` permette di riprendere una epoca interrotta.
    Se il DataLoader usa un sampler resumable, la stessa permutazione viene
    ricostruita e il sampler parte direttamente dal primo campione non ancora
    processato, senza leggere e scartare i batch precedenti.
    """

    model.train()

    total_meter = AverageMeter()
    reconstruction_meter = AverageMeter()
    classification_meter = AverageMeter()
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
        else float(cfg.training.grad_clip_norm)
    )

    checkpoint_every_steps = cfg.checkpoint.get(
        "save_every_steps",
        None,
    )

    if checkpoint_every_steps is not None:
        checkpoint_every_steps = int(
            checkpoint_every_steps
        )
        if checkpoint_every_steps <= 0:
            checkpoint_every_steps = None

    use_amp = amp_enabled(
        cfg,
        device,
    )

    reset_peak_gpu_memory(device)

    if (
            epoch_seed is not None
            and getattr(dataloader, "generator", None) is not None
    ):
        dataloader.generator.manual_seed(
            int(epoch_seed)
        )

    sampler = getattr(dataloader, "sampler", None)
    sampler_is_resumable = (
        hasattr(sampler, "set_epoch")
        and hasattr(sampler, "start_index")
    )

    if sampler_is_resumable:
        if dataloader.batch_size is None:
            raise ValueError(
                "Il resume efficiente richiede un batch_size esplicito."
            )

        sample_offset = (
            int(start_batch_in_epoch)
            * int(dataloader.batch_size)
        )

        sampler.set_epoch(
            seed=(
                int(epoch_seed)
                if epoch_seed is not None
                else 0
            ),
            start_index=sample_offset,
        )

    epoch_start = time.perf_counter()

    stop_training = False
    epoch_completed = True
    last_completed_batch = int(
        start_batch_in_epoch
    )

    if sampler_is_resumable:
        full_batches_in_epoch = (
            int(start_batch_in_epoch)
            + len(dataloader)
        )
        batch_iterator = enumerate(
            dataloader,
            start=int(start_batch_in_epoch) + 1,
        )
    else:
        full_batches_in_epoch = len(dataloader)
        batch_iterator = enumerate(
            dataloader,
            start=1,
        )

    for batch_idx, batch in batch_iterator:

        if (
                not sampler_is_resumable
                and batch_idx <= start_batch_in_epoch
        ):
            continue

        if (
                max_steps is not None
                and global_step >= max_steps
        ):
            stop_training = True
            epoch_completed = False
            break

        spectrogram = batch[
            "spectrogram"
        ].to(
            device,
            non_blocking=True,
        )

        batch_size = spectrogram.shape[0]

        optimizer.zero_grad(
            set_to_none=True
        )

        sync_device(device)
        step_start = time.perf_counter()

        with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
        ):
            output = model.forward_pretrain(
                spectrogram
            )

            losses = pretrain_loss(
                output,
                loss_cfg=cfg.loss,
            )

        if scaler is not None:
            scaler.scale(
                losses.total
            ).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            losses.total.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            optimizer.step()

        scheduler.step()

        sync_device(device)

        step_time = (
            time.perf_counter()
            - step_start
        )

        global_step += 1
        last_completed_batch = batch_idx
        num_examples += batch_size

        total_meter.update(
            float(losses.total.detach().item()),
            batch_size,
        )
        reconstruction_meter.update(
            float(losses.reconstruction.detach().item()),
            batch_size,
        )
        classification_meter.update(
            float(losses.classification.detach().item()),
            batch_size,
        )
        step_time_meter.update(step_time)

        if (
                global_step
                % int(cfg.training.log_interval)
                == 0
        ):
            lr = float(
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"[PRETRAIN] "
                f"step={global_step} "
                f"loss={total_meter.average:.5f} "
                f"rec={reconstruction_meter.average:.5f} "
                f"cls={classification_meter.average:.5f} "
                f"lr={lr:.3e}"
            )

        if (
                periodic_checkpoint_callback is not None
                and checkpoint_every_steps is not None
                and global_step % checkpoint_every_steps == 0
        ):
            periodic_checkpoint_callback(
                last_completed_batch,
                global_step,
            )

        if (
                max_steps is not None
                and global_step >= max_steps
        ):
            stop_training = True
            epoch_completed = (
                batch_idx >= full_batches_in_epoch
            )
            break

    elapsed = (
        time.perf_counter()
        - epoch_start
    )

    metrics = RuntimeMetrics(
        elapsed_sec=elapsed,
        step_time_sec_avg=(
            step_time_meter.average
        ),
        examples_per_sec=(
            num_examples
            / max(elapsed, 1e-12)
        ),
        peak_gpu_mem_mb=(
            get_peak_gpu_memory_mb(device)
        ),
    )

    losses_dict = {
        "loss_total": total_meter.average,
        "loss_reconstruction": reconstruction_meter.average,
        "loss_classification": classification_meter.average,
    }

    return (
        losses_dict,
        global_step,
        stop_training,
        metrics,
        last_completed_batch,
        epoch_completed,
    )


@torch.no_grad()
def validate_pretrain(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        cfg: DictConfig,
) -> dict[str, float]:
    """
    Validation del pretraining.

    La mask viene resa deterministica tramite un generatore
    con seed fisso.
    """

    model.eval()

    total_meter = AverageMeter()
    reconstruction_meter = AverageMeter()
    classification_meter = AverageMeter()

    use_amp = amp_enabled(
        cfg,
        device,
    )

    if device.type == "cuda":
        generator = torch.Generator(
            device=device
        )
    else:
        generator = torch.Generator()

    generator.manual_seed(
        int(
            cfg.training.val_mask_seed
        )
    )

    for batch in dataloader:

        spectrogram = batch[
            "spectrogram"
        ].to(
            device,
            non_blocking=True,
        )

        batch_size = (
            spectrogram.shape[0]
        )

        with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
        ):
            output = (
                model.forward_pretrain(
                    spectrogram,
                    generator=generator,
                )
            )

            losses = pretrain_loss(
                output,
                loss_cfg=cfg.loss,
            )

        total_meter.update(
            float(
                losses.total.item()
            ),
            batch_size,
        )

        reconstruction_meter.update(
            float(
                losses.reconstruction.item()
            ),
            batch_size,
        )

        classification_meter.update(
            float(
                losses.classification.item()
            ),
            batch_size,
        )

    return {
        "loss_total":
            total_meter.average,

        "loss_reconstruction":
            reconstruction_meter.average,

        "loss_classification":
            classification_meter.average,
    }


def run_pretraining(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
        output_dir: str | Path,
        resume_checkpoint: str | Path | None = None,
) -> dict:
    """
    Gestisce l'intero pretraining, incluso il resume completo della run.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model.to(device)

    optimizer = build_optimizer(
        model.parameters(),
        cfg,
    )

    configured_max_steps = (
        None
        if cfg.training.max_steps is None
        else int(cfg.training.max_steps)
    )

    if configured_max_steps is not None:
        total_steps = configured_max_steps
    else:
        total_steps = (
            int(cfg.training.epochs)
            * len(train_loader)
        )

    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        total_steps=total_steps,
    )

    scaler = build_grad_scaler(
        amp_enabled(cfg, device)
    )

    best_val_loss = math.inf
    global_step = 0
    start_epoch = 1
    start_batch_in_epoch = 0

    train_log_path = (
        output_dir / "train_log.jsonl"
    )
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    if resume_checkpoint is not None:
        resume_state = load_training_checkpoint(
            path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_best_metric_name="val_loss",
            current_cfg=cfg,
        )

        global_step = resume_state.global_step
        best_val_loss = resume_state.best_metric

        if resume_state.epoch_completed:
            start_epoch = resume_state.epoch + 1
            start_batch_in_epoch = 0
        else:
            start_epoch = max(
                1,
                resume_state.epoch,
            )
            start_batch_in_epoch = (
                resume_state.batch_in_epoch
            )

        if start_batch_in_epoch > len(train_loader):
            raise ValueError(
                "Checkpoint incompatibile con il DataLoader corrente: "
                f"batch_in_epoch={start_batch_in_epoch}, "
                f"batch_per_epoch={len(train_loader)}."
            )

        print(
            "\nResume checkpoint caricato: "
            f"{resume_checkpoint}"
        )
        print(
            "Stato ripristinato: "
            f"epoch={resume_state.epoch}, "
            f"batch_in_epoch={resume_state.batch_in_epoch}, "
            f"epoch_completed={resume_state.epoch_completed}, "
            f"global_step={global_step}, "
            f"best_val_loss={best_val_loss:.6f}"
        )

    tracker = build_experiment_tracker(
        cfg=cfg,
        output_dir=output_dir,
        job_type="pretrain",
    )

    print(f"\nAvvio pretraining su {device}")
    print(f"Esperimento: {cfg.experiment.name}")
    print(
        f"Masking: {cfg.masking.strategy} "
        f"{float(cfg.masking.ratio):.0%}"
    )

    if resume_checkpoint is not None:
        tracker.log(
            {
                "resume/enabled": 1,
                "resume/start_epoch": start_epoch,
                "resume/start_batch_in_epoch": start_batch_in_epoch,
                "global_step": global_step,
            },
            step=global_step,
        )

    total_epochs = resolve_pretraining_total_epochs(
        cfg=cfg,
        steps_per_epoch=len(train_loader),
    )

    if configured_max_steps is not None:
        print(
            "Orizzonte training: "
            f"max_steps={configured_max_steps} "
            f"(fino a {total_epochs} epoche necessarie)."
        )

    if start_epoch > total_epochs:
        print(
            "Il checkpoint ha già completato tutte le epoche configurate."
        )
        tracker.finish()
        return {
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "output_dir": str(output_dir),
        }

    base_seed = int(
        cfg.experiment.seed
    )

    for epoch in range(
            start_epoch,
            total_epochs + 1,
    ):

        resume_batch = (
            start_batch_in_epoch
            if epoch == start_epoch
            else 0
        )

        if configured_max_steps is None:
            epoch_label = (
                f"{epoch}/{total_epochs}"
            )
        else:
            epoch_label = (
                f"{epoch} "
                f"(target step {configured_max_steps})"
            )

        print(
            f"\nEpoch {epoch_label}"
        )

        if resume_batch > 0:
            print(
                "Ripresa intra-epoca: "
                f"riparto direttamente dopo {resume_batch} batch già completati."
            )

        def _save_periodic_checkpoint(
                batch_in_epoch: int,
                step: int,
        ) -> None:
            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=step,
                cfg=cfg,
                best_metric=best_val_loss,
                best_metric_name="val_loss",
                batch_in_epoch=batch_in_epoch,
                epoch_completed=False,
            )

            print(
                "[CHECKPOINT] "
                f"last.pt salvato a step={step} "
                f"(epoch={epoch}, batch={batch_in_epoch})"
            )

        (
            train_metrics,
            global_step,
            stop_training,
            runtime,
            batch_in_epoch,
            epoch_completed,
        ) = train_pretrain_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            cfg=cfg,
            global_step=global_step,
            start_batch_in_epoch=resume_batch,
            epoch_seed=base_seed + epoch,
            periodic_checkpoint_callback=(
                _save_periodic_checkpoint
                if bool(cfg.checkpoint.save_last)
                else None
            ),
        )

        val_metrics = validate_pretrain(
            model=model,
            dataloader=val_loader,
            device=device,
            cfg=cfg,
        )

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "resumed_from_batch": resume_batch,
            "epoch_completed": epoch_completed,
            "train": train_metrics,
            "validation": val_metrics,
            "runtime": {
                "epoch_time_sec": runtime.elapsed_sec,
                "step_time_sec_avg": runtime.step_time_sec_avg,
                "examples_per_sec": runtime.examples_per_sec,
                "peak_gpu_mem_mb": runtime.peak_gpu_mem_mb,
            },
        }

        append_jsonl(
            train_log_path,
            record,
        )

        tracker_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train/loss_total": train_metrics["loss_total"],
            "train/loss_reconstruction": train_metrics["loss_reconstruction"],
            "train/loss_classification": train_metrics["loss_classification"],
            "val/loss_total": val_metrics["loss_total"],
            "val/loss_reconstruction": val_metrics["loss_reconstruction"],
            "val/loss_classification": val_metrics["loss_classification"],
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

        print(
            "[VAL] "
            f"loss={val_metrics['loss_total']:.5f} "
            f"rec={val_metrics['loss_reconstruction']:.5f} "
            f"cls={val_metrics['loss_classification']:.5f}"
        )

        if (
                val_metrics["loss_total"]
                < best_val_loss
        ):
            best_val_loss = val_metrics[
                "loss_total"
            ]

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
                    best_metric=best_val_loss,
                    best_metric_name="val_loss",
                    batch_in_epoch=(
                        0
                        if epoch_completed
                        else batch_in_epoch
                    ),
                    epoch_completed=epoch_completed,
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
                best_metric=best_val_loss,
                best_metric_name="val_loss",
                batch_in_epoch=(
                    0
                    if epoch_completed
                    else batch_in_epoch
                ),
                epoch_completed=epoch_completed,
            )

        start_batch_in_epoch = 0

        if stop_training:
            break

    tracker.log(
        {
            "best/val_loss": best_val_loss,
            "global_step": global_step,
        },
        step=global_step,
    )

    tracker.finish()

    return {
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "output_dir": str(output_dir),
    }

