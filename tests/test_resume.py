from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mae_ast.training.utils import (
    load_training_checkpoint,
    save_checkpoint,
    set_seed,
)


def _cfg(max_steps=100, epochs=8):
    return OmegaConf.create(
        {
            "experiment": {"name": "resume_test", "seed": 42},
            "model": {"type": "test"},
            "patching": {"patch_h": 16, "patch_w": 16},
            "masking": {"strategy": "chunk", "ratio": 0.75},
            "loss": {"reconstruction_weight": 10.0},
            "audio": {"sample_rate": 16000},
            "training": {
                "optimizer": "adamw",
                "learning_rate": 1.0e-3,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "scheduler": "polynomial",
                "warmup_steps": 10,
                "grad_clip_norm": None,
                "batch_size": 2,
                "epochs": epochs,
                "max_steps": max_steps,
            },
        }
    )



def test_checkpoint_roundtrip_restores_full_training_state(tmp_path):
    set_seed(123)

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    x = torch.randn(3, 4)
    loss = model(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    saved_weights = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    path = tmp_path / "last.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=2,
        global_step=17,
        cfg=_cfg(),
        best_metric=3.25,
        best_metric_name="val_loss",
        batch_in_epoch=7,
        epoch_completed=False,
    )

    assert path.exists()
    assert not (tmp_path / "last.pt.tmp").exists()

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(5)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    random.random()
    np.random.rand()
    torch.rand(5)

    state = load_training_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        device=torch.device("cpu"),
        expected_best_metric_name="val_loss",
    )

    assert state.epoch == 2
    assert state.global_step == 17
    assert state.batch_in_epoch == 7
    assert state.epoch_completed is False
    assert state.best_metric == pytest.approx(3.25)

    for key, value in model.state_dict().items():
        assert torch.equal(value, saved_weights[key])

    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.rand()) == pytest.approx(expected_numpy)
    assert torch.equal(torch.rand(5), expected_torch)


def test_resume_rejects_incompatible_optimizer(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    path = tmp_path / "last.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=1,
        cfg=_cfg(),
        best_metric=1.0,
        best_metric_name="val_loss",
    )

    other_model = torch.nn.Linear(2, 1)
    other_optimizer = torch.optim.Adam(
        other_model.parameters(),
        lr=1.0e-3,
    )
    other_scheduler = torch.optim.lr_scheduler.LambdaLR(
        other_optimizer,
        lr_lambda=lambda _: 1.0,
    )

    with pytest.raises(ValueError, match="Optimizer incompatibile"):
        load_training_checkpoint(
            path=path,
            model=other_model,
            optimizer=other_optimizer,
            scheduler=other_scheduler,
            scaler=None,
            device=torch.device("cpu"),
            expected_best_metric_name="val_loss",
        )


def test_resume_rejects_changed_max_steps(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    path = tmp_path / "last.pt"
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=3,
        cfg=_cfg(max_steps=100),
        best_metric=1.0,
        best_metric_name="val_loss",
        batch_in_epoch=3,
        epoch_completed=False,
    )

    other_model = torch.nn.Linear(2, 1)
    other_optimizer = torch.optim.AdamW(other_model.parameters(), lr=1.0e-3)
    other_scheduler = torch.optim.lr_scheduler.LambdaLR(
        other_optimizer,
        lr_lambda=lambda _: 1.0,
    )

    with pytest.raises(ValueError, match="training.max_steps"):
        load_training_checkpoint(
            path=path,
            model=other_model,
            optimizer=other_optimizer,
            scheduler=other_scheduler,
            scaler=None,
            device=torch.device("cpu"),
            expected_best_metric_name="val_loss",
            current_cfg=_cfg(max_steps=200),
        )


def test_resume_rejects_changed_epochs(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    path = tmp_path / "last.pt"
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=3,
        cfg=_cfg(epochs=8),
        best_metric=1.0,
        best_metric_name="val_loss",
        batch_in_epoch=3,
        epoch_completed=False,
    )

    other_model = torch.nn.Linear(2, 1)
    other_optimizer = torch.optim.AdamW(other_model.parameters(), lr=1.0e-3)
    other_scheduler = torch.optim.lr_scheduler.LambdaLR(
        other_optimizer,
        lr_lambda=lambda _: 1.0,
    )

    with pytest.raises(ValueError, match="training.epochs"):
        load_training_checkpoint(
            path=path,
            model=other_model,
            optimizer=other_optimizer,
            scheduler=other_scheduler,
            scaler=None,
            device=torch.device("cpu"),
            expected_best_metric_name="val_loss",
            current_cfg=_cfg(epochs=9),
        )


def test_resume_accepts_legacy_config_when_new_fields_match_historical_defaults(
        tmp_path,
):
    from mae_ast.config import load_config

    current_cfg = load_config()

    legacy_cfg = OmegaConf.create(
        OmegaConf.to_container(
            current_cfg,
            resolve=True,
        )
    )

    del legacy_cfg.model["dropout_input"]

    for section in (
            "encoder",
            "decoder",
    ):
        del legacy_cfg.model[section]["attention_dropout"]
        del legacy_cfg.model[section]["activation_dropout"]
        del legacy_cfg.model[section]["layerdrop"]
        del legacy_cfg.model[section]["norm_first"]

    del legacy_cfg.training["eps"]

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(current_cfg.training.learning_rate),
        betas=tuple(current_cfg.training.betas),
        eps=1.0e-8,
        weight_decay=float(current_cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    path = tmp_path / "legacy.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=1,
        cfg=legacy_cfg,
        best_metric=1.0,
        best_metric_name="val_loss",
    )

    other_model = torch.nn.Linear(2, 1)
    other_optimizer = torch.optim.AdamW(
        other_model.parameters(),
        lr=float(current_cfg.training.learning_rate),
        betas=tuple(current_cfg.training.betas),
        eps=float(current_cfg.training.eps),
        weight_decay=float(current_cfg.training.weight_decay),
    )
    other_scheduler = torch.optim.lr_scheduler.LambdaLR(
        other_optimizer,
        lr_lambda=lambda _: 1.0,
    )

    state = load_training_checkpoint(
        path=path,
        model=other_model,
        optimizer=other_optimizer,
        scheduler=other_scheduler,
        scaler=None,
        device=torch.device("cpu"),
        expected_best_metric_name="val_loss",
        current_cfg=current_cfg,
    )

    assert state.global_step == 1
