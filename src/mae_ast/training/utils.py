"""
Utility condivise tra pretraining e fine-tuning.

Questo modulo raccoglie solamente funzionalità comuni:
- seed e riproducibilità;
- scelta del device;
- metriche medie;
- AMP / GradScaler;
- optimizer;
- scheduler;
- checkpoint;
- logging locale JSONL;
- misurazione tempo e memoria GPU.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR


# -------------------------------------------------------------------------
# Riproducibilità
# -------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Imposta il seed principale del progetto.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Seleziona automaticamente il device migliore disponibile.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# -------------------------------------------------------------------------
# Metriche
# -------------------------------------------------------------------------

class AverageMeter:
    """
    Calcola una media pesata incrementale.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(
            self,
            value: float,
            n: int = 1,
    ) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0

        return self.total / self.count


@dataclass
class RuntimeMetrics:
    """
    Metriche computazionali di una fase di training.
    """

    elapsed_sec: float
    step_time_sec_avg: float
    examples_per_sec: float
    peak_gpu_mem_mb: float | None


# -------------------------------------------------------------------------
# GPU / timing
# -------------------------------------------------------------------------

def sync_device(
        device: torch.device,
) -> None:
    """
    Sincronizza CUDA prima di una misura temporale.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_gpu_memory(
        device: torch.device,
) -> None:
    """
    Azzera il contatore del picco di memoria GPU.
    """

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_gpu_memory_mb(
        device: torch.device,
) -> float | None:
    """
    Restituisce il picco di memoria GPU allocata in MB.
    """

    if device.type != "cuda":
        return None

    return (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
    )


# -------------------------------------------------------------------------
# AMP
# -------------------------------------------------------------------------

def amp_enabled(
        cfg: DictConfig,
        device: torch.device,
) -> bool:
    """
    AMP viene attivato solamente su CUDA.
    """

    return bool(
        cfg.training.amp
        and device.type == "cuda"
    )


def build_grad_scaler(
        use_amp: bool,
):
    """
    Costruisce il GradScaler utilizzato con mixed precision.
    """

    if not torch.cuda.is_available():
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    except TypeError:
        # Compatibilità con versioni meno recenti di PyTorch.
        return torch.cuda.amp.GradScaler(
            enabled=use_amp
        )


# -------------------------------------------------------------------------
# Optimizer
# -------------------------------------------------------------------------

def build_optimizer(
        parameters,
        cfg: DictConfig,
) -> torch.optim.Optimizer:
    """
    Costruisce l'optimizer richiesto dalla configurazione.

    - ``adamw``: default del progetto/pretraining;
    - ``adam``: usato dalla recipe downstream SSAST/ESC-50.
    """

    optimizer_name = str(
        cfg.training.get(
            "optimizer",
            "adamw",
        )
    ).lower()

    kwargs = {
        "lr": float(
            cfg.training.learning_rate
        ),
        "betas": (
            float(cfg.training.betas[0]),
            float(cfg.training.betas[1]),
        ),
        "eps": float(
            cfg.training.get(
                "eps",
                1.0e-8,
            )
        ),
        "weight_decay": float(
            cfg.training.weight_decay
        ),
    }

    if optimizer_name == "adamw":
        return AdamW(
            parameters,
            **kwargs,
        )

    if optimizer_name == "adam":
        return Adam(
            parameters,
            **kwargs,
        )

    raise ValueError(
        f"Optimizer non supportato: {optimizer_name}"
    )


# -------------------------------------------------------------------------
# Scheduler
# -------------------------------------------------------------------------

def build_polynomial_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
) -> LambdaLR:
    """
    Scheduler con warmup lineare seguito da polynomial decay.
    """

    if total_steps <= 0:
        raise ValueError(
            "total_steps deve essere maggiore di zero."
        )

    warmup_steps = max(
        0,
        min(
            warmup_steps,
            total_steps,
        ),
    )

    def lr_lambda(
            current_step: int,
    ) -> float:

        # Warmup lineare.
        if (
                warmup_steps > 0
                and current_step < warmup_steps
        ):
            return (
                    float(current_step + 1)
                    / float(warmup_steps)
            )

        if total_steps == warmup_steps:
            return 1.0

        progress = (
                           current_step - warmup_steps
                   ) / float(
            total_steps - warmup_steps
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return (
                1.0 - progress
        ) ** power

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def build_multistep_epoch_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: DictConfig,
) -> MultiStepLR:
    """
    Scheduler a epoche usato nel fine-tuning ESC-50.

    Replica il comportamento della pipeline SSAST: il learning rate
    viene moltiplicato per ``lr_decay_gamma`` a partire da
    ``lr_decay_start_epoch`` e poi ogni ``lr_decay_step_epochs``.
    """

    start_epoch = int(
        cfg.training.lr_decay_start_epoch
    )

    step_epochs = int(
        cfg.training.lr_decay_step_epochs
    )

    gamma = float(
        cfg.training.lr_decay_gamma
    )

    total_epochs = int(
        cfg.training.epochs
    )

    if start_epoch < 1:
        raise ValueError(
            "lr_decay_start_epoch deve essere >= 1."
        )

    if step_epochs < 1:
        raise ValueError(
            "lr_decay_step_epochs deve essere >= 1."
        )

    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            "lr_decay_gamma deve essere in (0, 1]."
        )

    milestones = list(
        range(
            start_epoch,
            total_epochs + 1,
            step_epochs,
        )
    )

    return MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma,
    )


def scheduler_step_unit(
        cfg: DictConfig,
) -> str:
    """
    Restituisce quando deve avanzare lo scheduler: batch oppure epoca.
    """

    scheduler_name = str(
        cfg.training.scheduler
    ).lower()

    if scheduler_name == "multistep_epoch":
        return "epoch"

    return "batch"


def build_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: DictConfig,
        total_steps: int,
):
    """
    Costruisce lo scheduler richiesto dalla configurazione.
    """

    scheduler_name = str(
        cfg.training.scheduler
    ).lower()

    if scheduler_name == "polynomial":
        return build_polynomial_scheduler(
            optimizer=optimizer,
            warmup_steps=int(
                cfg.training.warmup_steps
            ),
            total_steps=total_steps,
        )

    if scheduler_name == "multistep_epoch":
        return build_multistep_epoch_scheduler(
            optimizer=optimizer,
            cfg=cfg,
        )

    if scheduler_name in {
        "none",
        "constant",
    }:
        return LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0,
        )

    raise ValueError(
        f"Scheduler non supportato: {scheduler_name}"
    )


# -------------------------------------------------------------------------
# Logging locale
# -------------------------------------------------------------------------

def append_jsonl(
        path: str | Path,
        record: dict[str, Any],
) -> None:
    """
    Aggiunge un record a un file JSONL.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
            "a",
            encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


# -------------------------------------------------------------------------
# Checkpoint
# -------------------------------------------------------------------------


@dataclass
class TrainingCheckpointState:
    """
    Stato minimo necessario per riprendere una run.

    ``batch_in_epoch`` indica quanti batch dell'epoca corrente sono già
    stati completati. Se ``epoch_completed`` è True, la ripresa parte
    dall'epoca successiva.
    """

    epoch: int
    global_step: int
    best_metric: float
    batch_in_epoch: int = 0
    epoch_completed: bool = True


def capture_rng_state() -> dict[str, Any]:
    """
    Cattura gli stati RNG usati dal progetto.

    Sono inclusi Python, NumPy, PyTorch CPU e, quando disponibile, CUDA.
    """

    numpy_state = np.random.get_state()

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(
                numpy_state[1].copy()
            ),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": None,
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(
        state: dict[str, Any] | None,
) -> None:
    """
    Ripristina gli stati RNG salvati nel checkpoint.

    I checkpoint precedenti alla versione 2 possono non contenerli: in
    quel caso il resume resta possibile, ma non è bitwise-riproducibile.
    """

    if not state:
        return

    if state.get("python") is not None:
        random.setstate(state["python"])

    if state.get("numpy") is not None:
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                numpy_state["state"]
                .cpu()
                .numpy()
                .astype(np.uint32, copy=False),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )

    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])

    if (
            torch.cuda.is_available()
            and state.get("cuda") is not None
    ):
        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


def _atomic_torch_save(
        payload: dict[str, Any],
        path: Path,
) -> None:
    """
    Salva prima su un file temporaneo e poi esegue una sostituzione atomica.

    In caso di interruzione durante ``torch.save`` il checkpoint precedente
    rimane quindi intatto.
    """

    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        tmp_path,
    )

    os.replace(
        tmp_path,
        path,
    )


def save_checkpoint(
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        epoch: int,
        global_step: int,
        cfg: DictConfig,
        best_metric: float,
        best_metric_name: str,
        batch_in_epoch: int = 0,
        epoch_completed: bool = True,
) -> None:
    """
    Salva lo stato completo di una run.

    Oltre ai pesi vengono salvati optimizer, scheduler, AMP scaler,
    posizione nella run e stati RNG. Il salvataggio è atomico.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "checkpoint_version": 2,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        "scaler_state_dict":
            scaler.state_dict()
            if scaler is not None
            else None,

        "epoch":
            int(epoch),

        "global_step":
            int(global_step),

        "batch_in_epoch":
            int(batch_in_epoch),

        "epoch_completed":
            bool(epoch_completed),

        "best_metric":
            float(best_metric),

        "best_metric_name":
            best_metric_name,

        "optimizer_class":
            optimizer.__class__.__name__,

        "scheduler_class":
            scheduler.__class__.__name__
            if scheduler is not None
            else None,

        "rng_state":
            capture_rng_state(),

        "config":
            OmegaConf.to_container(
                cfg,
                resolve=True,
            ),
    }

    _atomic_torch_save(
        checkpoint,
        path,
    )


def _normalize_resume_config_defaults(
        cfg: DictConfig,
) -> DictConfig:
    """
    Normalizza i campi introdotti dopo i primi checkpoint V2.

    I checkpoint storici non contenevano ``dropout_input``, dropout separati,
    ``layerdrop``, ``norm_first`` o ``training.eps``. Per quei checkpoint
    ricostruiamo esclusivamente i default equivalenti al comportamento storico,
    così il guard di resume continua a rifiutare differenze reali ma non semplici
    aggiunte retrocompatibili alla configurazione.
    """

    normalized = OmegaConf.create(
        OmegaConf.to_container(
            cfg,
            resolve=True,
        )
    )

    if OmegaConf.select(
            normalized,
            "model",
    ) is not None:
        if OmegaConf.select(
                normalized,
                "model.dropout_input",
        ) is None:
            OmegaConf.update(
                normalized,
                "model.dropout_input",
                0.0,
                merge=False,
            )

        for section in (
                "encoder",
                "decoder",
        ):
            section_path = (
                f"model.{section}"
            )

            if OmegaConf.select(
                    normalized,
                    section_path,
            ) is None:
                continue

            dropout = OmegaConf.select(
                normalized,
                f"{section_path}.dropout",
                default=0.0,
            )

            defaults = {
                "attention_dropout": float(dropout),
                "activation_dropout": float(dropout),
                "layerdrop": 0.0,
                "norm_first": True,
            }

            for key, value in defaults.items():
                full_path = (
                    f"{section_path}.{key}"
                )

                if OmegaConf.select(
                        normalized,
                        full_path,
                ) is None:
                    OmegaConf.update(
                        normalized,
                        full_path,
                        value,
                        merge=False,
                    )

    if OmegaConf.select(
            normalized,
            "training",
    ) is not None:
        if OmegaConf.select(
                normalized,
                "training.eps",
        ) is None:
            OmegaConf.update(
                normalized,
                "training.eps",
                1.0e-8,
                merge=False,
            )

    return normalized


def _validate_resume_config(
        checkpoint_config: dict[str, Any] | None,
        current_cfg: DictConfig | None,
) -> None:
    """
    Verifica le parti della configurazione che non devono cambiare in resume.

    Sono volutamente esclusi path locali, logging e numero di worker,
    che possono cambiare quando la run viene spostata su un'altra macchina.

    ``epochs`` e ``max_steps`` fanno invece parte del piano di ottimizzazione:
    nel pretraining polynomial determinano l'orizzonte dello scheduler. Per un
    resume fedele devono quindi restare invariati. Estendere intenzionalmente
    una run va trattato come un nuovo esperimento, non come un resume neutro.
    """

    if checkpoint_config is None or current_cfg is None:
        return

    saved_cfg = _normalize_resume_config_defaults(
        OmegaConf.create(
            checkpoint_config
        )
    )

    normalized_current_cfg = (
        _normalize_resume_config_defaults(
            current_cfg
        )
    )

    critical_paths = [
        "model",
        "patching",
        "masking",
        "loss",
        "audio",
        "training.optimizer",
        "training.learning_rate",
        "training.weight_decay",
        "training.betas",
        "training.eps",
        "training.scheduler",
        "training.warmup_steps",
        "training.grad_clip_norm",
        "training.batch_size",
        "training.epochs",
        "training.max_steps",
    ]

    mismatches: list[str] = []

    for key in critical_paths:
        saved_value = OmegaConf.select(
            saved_cfg,
            key,
        )
        current_value = OmegaConf.select(
            normalized_current_cfg,
            key,
        )

        if OmegaConf.to_container(
                OmegaConf.create({"value": saved_value}),
                resolve=True,
        )["value"] != OmegaConf.to_container(
                OmegaConf.create({"value": current_value}),
                resolve=True,
        )["value"]:
            mismatches.append(key)

    if mismatches:
        raise ValueError(
            "Configurazione incompatibile con il checkpoint di resume. "
            "Campi critici modificati: "
            + ", ".join(mismatches)
        )


def load_training_checkpoint(
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        device: torch.device,
        expected_best_metric_name: str | None = None,
        current_cfg: DictConfig | None = None,
) -> TrainingCheckpointState:
    """
    Ripristina una run completa.

    Vengono ripristinati modello, optimizer, scheduler, AMP scaler e RNG.
    I checkpoint V1 restano leggibili; in quel caso ``batch_in_epoch`` vale
    zero e l'epoca viene considerata completata.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint di resume non trovato: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
    }

    missing = required_keys.difference(
        checkpoint.keys()
    )

    if missing:
        raise ValueError(
            "Checkpoint non valido per il resume. Campi mancanti: "
            + ", ".join(sorted(missing))
        )

    _validate_resume_config(
        checkpoint.get("config"),
        current_cfg,
    )

    saved_optimizer_class = checkpoint.get(
        "optimizer_class"
    )

    if (
            saved_optimizer_class is not None
            and saved_optimizer_class
            != optimizer.__class__.__name__
    ):
        raise ValueError(
            "Optimizer incompatibile con il checkpoint: "
            f"salvato={saved_optimizer_class}, "
            f"corrente={optimizer.__class__.__name__}."
        )

    saved_scheduler_class = checkpoint.get(
        "scheduler_class"
    )

    current_scheduler_class = (
        scheduler.__class__.__name__
        if scheduler is not None
        else None
    )

    if (
            saved_scheduler_class is not None
            and saved_scheduler_class
            != current_scheduler_class
    ):
        raise ValueError(
            "Scheduler incompatibile con il checkpoint: "
            f"salvato={saved_scheduler_class}, "
            f"corrente={current_scheduler_class}."
        )

    if expected_best_metric_name is not None:
        saved_metric_name = checkpoint.get(
            "best_metric_name"
        )

        if (
                saved_metric_name is not None
                and saved_metric_name
                != expected_best_metric_name
        ):
            raise ValueError(
                "Checkpoint con metrica best incompatibile: "
                f"salvata={saved_metric_name}, "
                f"attesa={expected_best_metric_name}."
            )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if (
            scheduler is not None
            and checkpoint.get(
        "scheduler_state_dict"
    ) is not None
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (
            scaler is not None
            and checkpoint.get(
        "scaler_state_dict"
    ) is not None
    ):
        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    restore_rng_state(
        checkpoint.get("rng_state")
    )

    checkpoint_version = int(
        checkpoint.get(
            "checkpoint_version",
            1,
        )
    )

    return TrainingCheckpointState(
        epoch=int(
            checkpoint.get("epoch", 0)
        ),
        global_step=int(
            checkpoint.get(
                "global_step",
                0,
            )
        ),
        best_metric=float(
            checkpoint.get(
                "best_metric",
                float("inf"),
            )
        ),
        batch_in_epoch=(
            int(
                checkpoint.get(
                    "batch_in_epoch",
                    0,
                )
            )
            if checkpoint_version >= 2
            else 0
        ),
        epoch_completed=(
            bool(
                checkpoint.get(
                    "epoch_completed",
                    True,
                )
            )
            if checkpoint_version >= 2
            else True
        ),
    )
