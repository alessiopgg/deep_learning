"""
Tracking opzionale degli esperimenti.

Il training locale JSONL rimane la sorgente di logging di base.
Questo modulo aggiunge, quando richiesto dalla configurazione,
un backend Weights & Biases senza introdurre dipendenze W&B
nel modello, nel dataset o nei loop per-batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class ExperimentTracker:
    """
    Interfaccia minima comune per il tracking degli esperimenti.
    """

    def log(
            self,
            metrics: dict[str, Any],
            step: int | None = None,
    ) -> None:
        """
        Registra un insieme di metriche.
        """

    def finish(self) -> None:
        """
        Chiude il tracker e forza la finalizzazione della run.
        """


class LocalTracker(ExperimentTracker):
    """
    Tracker nullo usato quando il backend è locale.

    Il logging JSONL esistente continua a essere gestito
    direttamente dai moduli di training.
    """

    def log(
            self,
            metrics: dict[str, Any],
            step: int | None = None,
    ) -> None:
        return None

    def finish(self) -> None:
        return None


class WandBTracker(ExperimentTracker):
    """
    Wrapper minimale attorno a Weights & Biases.
    """

    def __init__(
            self,
            cfg: DictConfig,
            output_dir: str | Path,
            job_type: str,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Il backend W&B è attivo ma il pacchetto 'wandb' "
                "non è installato. Installa con: pip install wandb"
            ) from exc

        logging_cfg = cfg.logging

        mode = str(
            logging_cfg.get(
                "wandb_mode",
                "offline",
            )
        ).lower()

        allowed_modes = {
            "online",
            "offline",
            "disabled",
        }

        if mode not in allowed_modes:
            raise ValueError(
                "logging.wandb_mode deve essere uno tra: "
                "online, offline, disabled. "
                f"Ricevuto: {mode}"
            )

        project = str(
            logging_cfg.get(
                "wandb_project",
                "mae-ast",
            )
        )

        entity = logging_cfg.get(
            "wandb_entity",
            None,
        )

        group = logging_cfg.get(
            "wandb_group",
            None,
        )

        tags = list(
            logging_cfg.get(
                "wandb_tags",
                [],
            )
        )

        configured_dir = logging_cfg.get(
            "wandb_dir",
            None,
        )

        if configured_dir is None:
            run_dir = Path(output_dir)
        else:
            run_dir = Path(
                str(configured_dir)
            )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_dict = OmegaConf.to_container(
            cfg,
            resolve=True,
        )

        self._run = wandb.init(
            project=project,
            entity=(
                None
                if entity is None
                else str(entity)
            ),
            name=str(
                cfg.experiment.name
            ),
            group=(
                None
                if group is None
                else str(group)
            ),
            tags=tags,
            job_type=job_type,
            mode=mode,
            dir=str(
                run_dir.resolve()
            ),
            config=config_dict,
        )

        print(
            "W&B attivo: "
            f"project={project} "
            f"mode={mode} "
            f"job_type={job_type}"
        )

    def log(
            self,
            metrics: dict[str, Any],
            step: int | None = None,
    ) -> None:
        self._run.log(
            metrics,
            step=step,
        )

    def finish(self) -> None:
        self._run.finish()


def build_experiment_tracker(
        cfg: DictConfig,
        output_dir: str | Path,
        job_type: str,
) -> ExperimentTracker:
    """
    Costruisce il tracker richiesto dalla configurazione.

    Backend supportati:
    - local: mantiene solo il logging locale già esistente;
    - wandb: abilita Weights & Biases.
    """

    backend = str(
        cfg.logging.get(
            "backend",
            "local",
        )
    ).lower()

    if backend == "local":
        return LocalTracker()

    if backend == "wandb":
        return WandBTracker(
            cfg=cfg,
            output_dir=output_dir,
            job_type=job_type,
        )

    raise ValueError(
        "Backend di logging non supportato: "
        f"{backend}"
    )
