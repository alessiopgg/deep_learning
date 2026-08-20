"""
Entry point per il fine-tuning supervisionato di MAE-AST.

Questo script:
1. carica la configurazione;
2. costruisce i DataLoader downstream;
3. costruisce MAE-AST;
4. carica il checkpoint pretrained;
5. avvia il fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mae_ast.config import (
    load_config,
    print_config,
    save_config,
)
from mae_ast.data.dataset import build_dataloader
from mae_ast.models.mae_ast import MAEASTModel
from mae_ast.training.finetune import run_finetuning
from mae_ast.training.utils import (
    get_device,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning MAE-AST"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configurazione YAML dell'esperimento.",
    )

    parser.add_argument(
        "--local-config",
        type=str,
        default=None,
        help=(
            "Configurazione YAML locale per path e impostazioni "
            "dipendenti dalla macchina."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint ottenuto dal pretraining.",
    )

    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override OmegaConf.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(
        config_path=args.config,
        local_config_path=args.local_config,
        overrides=args.set,
    )

    if cfg.data.train_manifest is None:
        raise ValueError(
            "data.train_manifest non specificato."
        )

    if cfg.data.val_manifest is None:
        raise ValueError(
            "data.val_manifest non specificato."
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato: "
            f"{checkpoint_path}"
        )

    set_seed(
        int(cfg.experiment.seed)
    )

    device = get_device()

    print("\nConfigurazione utilizzata:\n")
    print_config(cfg)

    print(
        f"\nDevice selezionato: {device}"
    )

    # -------------------------------------------------------------
    # Dataset downstream
    # -------------------------------------------------------------

    validation_batch_size = (
        int(cfg.training.batch_size)
        if cfg.training.validation_batch_size is None
        else int(cfg.training.validation_batch_size)
    )

    train_loader = build_dataloader(
        manifest_path=cfg.data.train_manifest,
        audio_cfg=cfg.audio,
        mode="finetune",
        batch_size=int(
            cfg.training.batch_size
        ),
        dataset_root=cfg.data.dataset_root,
        stats_path=cfg.data.stats_file,
        shuffle=True,
        num_workers=int(
            cfg.data.num_workers
        ),
        pin_memory=bool(
            cfg.data.pin_memory
        ),
        # La pipeline SSAST ESC-50 scarta il batch train incompleto.
        drop_last=True,
        strict=bool(
            cfg.data.strict_loading
        ),
        # SpecAugment + noise SOLO sul training downstream.
        augment=True,
    )

    val_loader = build_dataloader(
        manifest_path=cfg.data.val_manifest,
        audio_cfg=cfg.audio,
        mode="finetune",
        batch_size=validation_batch_size,
        dataset_root=cfg.data.dataset_root,
        stats_path=cfg.data.stats_file,
        shuffle=False,
        num_workers=int(
            cfg.data.num_workers
        ),
        pin_memory=bool(
            cfg.data.pin_memory
        ),
        drop_last=False,
        strict=bool(
            cfg.data.strict_loading
        ),
        # Validation sempre pulita: nessuna augmentation.
        augment=False,
    )

    # -------------------------------------------------------------
    # Modello
    # -------------------------------------------------------------

    # Il downstream utilizza sempre il vero MAE-AST.
    model = MAEASTModel(cfg)

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    output_dir = (
            Path(
                cfg.checkpoint.output_dir
            )
            / str(
        cfg.experiment.name
    )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_config(
        cfg,
        output_dir
        / "config.yaml",
        )

    # -------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------

    result = run_finetuning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        pretrained_checkpoint=checkpoint_path,
    )

    print("\nFine-tuning completato.")

    print(
        f"Best validation accuracy: "
        f"{result['best_val_accuracy']:.4f}"
    )

    print(
        f"Output: "
        f"{result['output_dir']}"
    )


if __name__ == "__main__":
    main()