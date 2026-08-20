"""
Entry point per il pretraining self-supervised di MAE-AST.

Questo script:
1. carica la configurazione OmegaConf;
2. inizializza seed e device;
3. costruisce DataLoader e modello;
4. salva la configurazione effettiva della run;
5. avvia il pretraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mae_ast.config import (
    load_config,
    print_config,
    save_config,
)
from mae_ast.data.dataset import build_dataloader
from mae_ast.models.full_sequence_proxy import FullSequenceProxy
from mae_ast.models.mae_ast import MAEASTModel
from mae_ast.training.pretrain import run_pretraining
from mae_ast.training.utils import (
    get_device,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretraining MAE-AST"
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
        "--set",
        nargs="*",
        default=None,
        help=(
            "Override OmegaConf. Esempio: "
            "--set model.encoder.layers=12 training.batch_size=16"
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint di training completo da cui riprendere, "
            "tipicamente outputs/<esperimento>/last.pt."
        ),
    )

    return parser.parse_args()


def build_model(cfg):
    """
    Costruisce il modello richiesto dalla configurazione.
    """

    model_type = str(
        cfg.model.type
    ).lower()

    if model_type == "mae":
        return MAEASTModel(cfg)

    if model_type in {
        "full_sequence_proxy",
        "proxy",
    }:
        return FullSequenceProxy(cfg)

    raise ValueError(
        f"Tipo di modello non supportato: {model_type}"
    )


def main():
    args = parse_args()

    cfg = load_config(
        config_path=args.config,
        local_config_path=args.local_config,
        overrides=args.set,
    )

    if cfg.data.train_manifest is None:
        raise ValueError(
            "data.train_manifest non è stato specificato."
        )

    if cfg.data.val_manifest is None:
        raise ValueError(
            "data.val_manifest non è stato specificato."
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
    # DataLoader
    # -------------------------------------------------------------

    # Generatore separato dal RNG globale del modello. Viene riseminato
    # a ogni epoca dal loop di training per ricostruire lo stesso shuffle
    # anche dopo un resume intra-epoca.
    train_generator = torch.Generator()
    train_generator.manual_seed(
        int(cfg.experiment.seed)
    )

    train_loader = build_dataloader(
        manifest_path=cfg.data.train_manifest,
        audio_cfg=cfg.audio,
        mode="pretrain",
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
        drop_last=True,
        strict=bool(
            cfg.data.strict_loading
        ),
        generator=train_generator,
        resumable_shuffle=True,
    )

    val_loader = build_dataloader(
        manifest_path=cfg.data.val_manifest,
        audio_cfg=cfg.audio,
        mode="pretrain",
        batch_size=int(
            cfg.training.batch_size
        ),
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
    )

    # -------------------------------------------------------------
    # Modello
    # -------------------------------------------------------------

    model = build_model(cfg)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parametri modello: "
        f"{num_parameters:,}"
    )

    # -------------------------------------------------------------
    # Directory della run
    # -------------------------------------------------------------

    configured_output_dir = (
        Path(cfg.checkpoint.output_dir)
        / str(cfg.experiment.name)
    )

    resume_path = (
        None
        if args.resume is None
        else Path(args.resume)
    )

    # In resume continuiamo nella directory della run originale. Questo
    # evita di scrivere last.pt/best.pt in una cartella diversa per errore.
    output_dir = (
        configured_output_dir
        if resume_path is None
        else resume_path.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_config(
        cfg,
        output_dir / (
            "config.yaml"
            if resume_path is None
            else "config_resume.yaml"
        ),
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    result = run_pretraining(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        resume_checkpoint=resume_path,
    )

    print("\nPretraining completato.")

    print(
        f"Best validation loss: "
        f"{result['best_val_loss']:.6f}"
    )

    print(
        f"Global step finale: "
        f"{result['global_step']}"
    )

    print(
        f"Output: "
        f"{result['output_dir']}"
    )


if __name__ == "__main__":
    main()