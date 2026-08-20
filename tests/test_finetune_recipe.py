import math

import torch
from omegaconf import OmegaConf

from mae_ast.config import load_config
from mae_ast.data.audio import AudioPreprocessor
from mae_ast.training.utils import (
    build_optimizer,
    build_scheduler,
    scheduler_step_unit,
)


def test_esc50_recipe_config_6l():
    cfg = load_config(
        config_path="configs/experiments/esc50_finetune_6l.yaml"
    )

    assert cfg.model.encoder.layers == 6
    assert cfg.training.epochs == 50
    assert cfg.training.batch_size == 48
    assert cfg.training.validation_batch_size == 96
    assert cfg.training.optimizer == "adam"
    assert cfg.training.warmup_steps == 0
    assert cfg.training.scheduler == "multistep_epoch"
    assert cfg.training.lr_decay_start_epoch == 6
    assert math.isclose(cfg.training.lr_decay_gamma, 0.85)
    assert cfg.audio.finetune_freq_mask == 24
    assert cfg.audio.finetune_time_mask == 96
    assert cfg.audio.finetune_noise is True


def test_esc50_recipe_config_12l():
    cfg = load_config(
        config_path="configs/experiments/esc50_finetune_12l.yaml"
    )

    assert cfg.model.encoder.layers == 12
    assert cfg.training.optimizer == "adam"
    assert cfg.training.scheduler == "multistep_epoch"


def test_esc50_optimizer_is_adam():
    cfg = load_config(
        config_path="configs/experiments/esc50_finetune_6l.yaml"
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
    assert not isinstance(
        optimizer,
        torch.optim.AdamW,
    )
    assert optimizer.defaults["betas"] == (0.95, 0.999)
    assert math.isclose(
        optimizer.defaults["weight_decay"],
        5.0e-7,
    )


def test_multistep_scheduler_is_epoch_based():
    cfg = load_config(
        config_path="configs/experiments/esc50_finetune_6l.yaml"
    )

    parameter = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    optimizer = build_optimizer(
        [parameter],
        cfg,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        total_steps=100,
    )

    assert scheduler_step_unit(cfg) == "epoch"

    # Epoche 1..5: LR invariato. Dopo lo step dell'epoca 6
    # entra il primo decadimento 0.85, come MultiStepLR di SSAST.
    for _ in range(5):
        optimizer.step()
        scheduler.step()

    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        1.0e-4,
        rel_tol=1e-8,
    )

    optimizer.step()
    scheduler.step()

    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        8.5e-5,
        rel_tol=1e-8,
    )


def test_specaugment_and_noise_preserve_shape_and_finite_values():
    cfg = OmegaConf.create(
        {
            "sample_rate": 16000,
            "n_mels": 128,
            "frame_length_ms": 25,
            "frame_shift_ms": 10,
            "f_min": 20.0,
            "f_max": 8000.0,
            "pretrain_target_frames": 1024,
            "finetune_target_frames": 512,
            "resample": True,
            "normalize_to_half_std": True,
            "eps": 1.0e-8,
            "finetune_freq_mask": 24,
            "finetune_time_mask": 96,
            "finetune_noise": True,
            "finetune_noise_max_scale": 0.1,
            "finetune_time_roll_max": 10,
        }
    )

    preprocessor = AudioPreprocessor(
        cfg=cfg,
        norm_mean=0.0,
        norm_std=1.0,
    )

    torch.manual_seed(42)

    fbank = torch.ones(
        512,
        128,
    )

    augmented = preprocessor._apply_specaugment(
        fbank
    )
    augmented = preprocessor._normalize(
        augmented
    )
    augmented = preprocessor._apply_finetune_noise(
        augmented
    )

    assert augmented.shape == (512, 128)
    assert torch.isfinite(augmented).all()
    assert not torch.equal(
        augmented,
        torch.full_like(augmented, 0.5),
    )
