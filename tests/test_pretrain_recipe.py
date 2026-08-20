from __future__ import annotations

import math

import torch

from mae_ast.config import load_config
from mae_ast.training.pretrain import (
    resolve_pretraining_total_epochs,
)
from mae_ast.training.utils import build_optimizer


def test_paper_recipe_12l_is_explicit_and_consistent():
    cfg = load_config(
        config_path=(
            "configs/experiments/"
            "pretrain_paper_12l.yaml"
        )
    )

    assert cfg.model.encoder.layers == 12
    assert cfg.model.decoder.layers == 2
    assert cfg.model.dropout_input == 0.1

    assert cfg.model.encoder.dropout == 0.1
    assert cfg.model.encoder.attention_dropout == 0.1
    assert cfg.model.encoder.activation_dropout == 0.0
    assert cfg.model.encoder.layerdrop == 0.05
    assert cfg.model.encoder.norm_first is False

    assert cfg.model.decoder.dropout == 0.1
    assert cfg.model.decoder.attention_dropout == 0.1
    assert cfg.model.decoder.activation_dropout == 0.0
    assert cfg.model.decoder.layerdrop == 0.0
    assert cfg.model.decoder.norm_first is False

    assert cfg.masking.strategy == "chunk"
    assert cfg.masking.ratio == 0.75
    assert cfg.masking.shared_across_batch is True

    assert cfg.training.optimizer == "adam"
    assert cfg.training.betas == [0.9, 0.98]
    assert math.isclose(cfg.training.eps, 1.0e-6)
    assert math.isclose(cfg.training.weight_decay, 0.01)
    assert cfg.training.warmup_steps == 32000
    assert cfg.training.max_steps == 550000
    assert cfg.training.grad_clip_norm == 10.0


def test_paper_recipe_6l_changes_only_encoder_depth_and_name():
    cfg_12 = load_config(
        config_path=(
            "configs/experiments/"
            "pretrain_paper_12l.yaml"
        )
    )
    cfg_6 = load_config(
        config_path=(
            "configs/experiments/"
            "pretrain_paper_6l.yaml"
        )
    )

    assert cfg_12.model.encoder.layers == 12
    assert cfg_6.model.encoder.layers == 6

    assert cfg_12.model.dropout_input == cfg_6.model.dropout_input
    assert cfg_12.model.decoder == cfg_6.model.decoder
    assert cfg_12.masking == cfg_6.masking
    assert cfg_12.training == cfg_6.training


def test_paper_optimizer_uses_adam_eps_and_betas():
    cfg = load_config(
        config_path=(
            "configs/experiments/"
            "pretrain_paper_12l.yaml"
        )
    )

    parameter = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    optimizer = build_optimizer(
        [parameter],
        cfg,
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )
    assert optimizer.defaults["betas"] == (0.9, 0.98)
    assert math.isclose(
        optimizer.defaults["eps"],
        1.0e-6,
    )
    assert math.isclose(
        optimizer.defaults["weight_decay"],
        0.01,
    )


def test_default_optimizer_keeps_historical_eps():
    cfg = load_config()

    parameter = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    optimizer = build_optimizer(
        [parameter],
        cfg,
    )

    assert math.isclose(
        optimizer.defaults["eps"],
        1.0e-8,
    )


def test_max_steps_becomes_primary_training_horizon():
    cfg = load_config(
        config_path=(
            "configs/experiments/"
            "pretrain_paper_12l.yaml"
        )
    )

    # 478284 / 32 con drop_last=True -> 14946 step/epoca.
    total_epochs = resolve_pretraining_total_epochs(
        cfg=cfg,
        steps_per_epoch=14946,
    )

    assert total_epochs == 37
    assert 36 * 14946 < 550000
    assert 37 * 14946 >= 550000


def test_epochs_remain_primary_when_max_steps_is_null():
    cfg = load_config()

    total_epochs = resolve_pretraining_total_epochs(
        cfg=cfg,
        steps_per_epoch=123,
    )

    assert total_epochs == 8


def test_run_pretraining_reaches_max_steps_beyond_configured_epochs(tmp_path):
    from types import SimpleNamespace

    from torch.utils.data import DataLoader, Dataset

    from mae_ast.training.pretrain import run_pretraining

    class TinyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {
                "spectrogram": torch.tensor(
                    [[float(index)]],
                    dtype=torch.float32,
                )
            }

    class TinyPretrainModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(
                torch.tensor(0.5)
            )

        def forward_pretrain(
                self,
                spectrogram,
                generator=None,
        ):
            batch_size = spectrogram.shape[0]
            target = torch.ones(
                batch_size,
                2,
                3,
                device=spectrogram.device,
            )
            prediction = (
                self.scale
                * torch.ones_like(target)
            )

            return SimpleNamespace(
                reconstruction_pred=prediction,
                classification_pred=prediction,
                target_masked=target,
            )

    cfg = load_config(
        overrides=[
            "training.batch_size=2",
            "training.epochs=1",
            "training.max_steps=3",
            "training.optimizer=adam",
            "training.learning_rate=0.001",
            "training.weight_decay=0.0",
            "training.betas=[0.9,0.999]",
            "training.eps=1e-8",
            "training.scheduler=polynomial",
            "training.warmup_steps=0",
            "training.grad_clip_norm=null",
            "training.amp=false",
            "training.log_interval=100",
            "checkpoint.save_best=false",
            "checkpoint.save_last=false",
            "checkpoint.save_every_steps=null",
            "logging.backend=local",
        ]
    )

    loader = DataLoader(
        TinyDataset(),
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    result = run_pretraining(
        model=TinyPretrainModel(),
        train_loader=loader,
        val_loader=loader,
        cfg=cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )

    # Con 2 step/epoca e epochs=1, la vecchia logica si sarebbe fermata a 2.
    # Ora max_steps=3 forza correttamente l'ingresso nella seconda epoca.
    assert result["global_step"] == 3
