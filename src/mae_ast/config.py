"""
Gestione delle configurazioni del progetto MAE-AST.

La configurazione di base viene individuata automaticamente
rispetto alla root del progetto, indipendentemente dalla
directory da cui viene lanciato uno script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


# config.py si trova in:
#
# project_root/src/mae_ast/config.py
#
# quindi risaliamo di tre livelli per ottenere la root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_CONFIG = (
        PROJECT_ROOT
        / "configs"
        / "base.yaml"
)


def _resolve_path(
        path: str | Path,
) -> Path:
    """
    Risolve un percorso.

    Se il path è assoluto viene usato direttamente.
    Se è relativo viene interpretato rispetto alla root
    del progetto MAE-AST.
    """

    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(
        config_path: str | Path | None = None,
        local_config_path: str | Path | None = None,
        base_path: str | Path | None = None,
        overrides: Sequence[str] | None = None,
) -> DictConfig:
    """
    Carica la configurazione base e, opzionalmente,
    quella specifica dell'esperimento e quella locale.

    Priorità:

        base.yaml
            ↓
        configurazione esperimento
            ↓
        configurazione locale
            ↓
        override da riga di comando
    """

    if base_path is None:
        resolved_base_path = DEFAULT_BASE_CONFIG
    else:
        resolved_base_path = _resolve_path(
            base_path
        )

    if not resolved_base_path.exists():
        raise FileNotFoundError(
            "Configurazione base non trovata: "
            f"{resolved_base_path}"
        )

    cfg = OmegaConf.load(
        resolved_base_path
    )

    if config_path is not None:

        resolved_config_path = _resolve_path(
            config_path
        )

        if not resolved_config_path.exists():
            raise FileNotFoundError(
                "Configurazione esperimento "
                f"non trovata: {resolved_config_path}"
            )

        experiment_cfg = OmegaConf.load(
            resolved_config_path
        )

        cfg = OmegaConf.merge(
            cfg,
            experiment_cfg,
        )

    if local_config_path is not None:

        resolved_local_config_path = _resolve_path(
            local_config_path
        )

        if not resolved_local_config_path.exists():
            raise FileNotFoundError(
                "Configurazione locale non trovata: "
                f"{resolved_local_config_path}"
            )

        local_cfg = OmegaConf.load(
            resolved_local_config_path
        )

        cfg = OmegaConf.merge(
            cfg,
            local_cfg,
        )

    if overrides:

        override_cfg = OmegaConf.from_dotlist(
            list(overrides)
        )

        cfg = OmegaConf.merge(
            cfg,
            override_cfg,
        )

    return cfg


def print_config(
        cfg: DictConfig,
) -> None:
    """Stampa la configurazione completa."""

    print(
        OmegaConf.to_yaml(
            cfg,
            resolve=True,
        )
    )


def save_config(
        cfg: DictConfig,
        output_path: str | Path,
) -> None:
    """
    Salva la configurazione effettivamente utilizzata.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OmegaConf.save(
        config=cfg,
        f=output_path,
    )