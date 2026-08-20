from __future__ import annotations

import sys
import types

import pytest
from omegaconf import OmegaConf

from mae_ast.training.tracking import (
    LocalTracker,
    WandBTracker,
    build_experiment_tracker,
)


def _base_cfg(backend: str):
    return OmegaConf.create(
        {
            "experiment": {
                "name": "tracking_test",
            },
            "logging": {
                "backend": backend,
                "wandb_project": "mae-ast-test",
                "wandb_entity": None,
                "wandb_mode": "offline",
                "wandb_group": None,
                "wandb_tags": ["test"],
                "wandb_dir": None,
            },
        }
    )


def test_local_tracker_is_noop(tmp_path):
    cfg = _base_cfg("local")

    tracker = build_experiment_tracker(
        cfg=cfg,
        output_dir=tmp_path,
        job_type="pretrain",
    )

    assert isinstance(tracker, LocalTracker)

    tracker.log(
        {"train/loss_total": 1.0},
        step=1,
    )
    tracker.finish()


def test_unknown_backend_raises(tmp_path):
    cfg = _base_cfg("unknown")

    with pytest.raises(ValueError):
        build_experiment_tracker(
            cfg=cfg,
            output_dir=tmp_path,
            job_type="pretrain",
        )


def test_wandb_tracker_uses_expected_api(tmp_path, monkeypatch):
    calls = {}

    class FakeRun:
        def log(self, metrics, step=None):
            calls["log"] = {
                "metrics": metrics,
                "step": step,
            }

        def finish(self):
            calls["finished"] = True

    fake_module = types.ModuleType("wandb")

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    fake_module.init = fake_init

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        fake_module,
    )

    cfg = _base_cfg("wandb")

    tracker = build_experiment_tracker(
        cfg=cfg,
        output_dir=tmp_path,
        job_type="pretrain",
    )

    assert isinstance(tracker, WandBTracker)
    assert calls["init"]["project"] == "mae-ast-test"
    assert calls["init"]["name"] == "tracking_test"
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["job_type"] == "pretrain"
    assert calls["init"]["tags"] == ["test"]

    tracker.log(
        {"val/loss_total": 2.0},
        step=7,
    )

    assert calls["log"]["step"] == 7
    assert calls["log"]["metrics"]["val/loss_total"] == 2.0

    tracker.finish()
    assert calls["finished"] is True
