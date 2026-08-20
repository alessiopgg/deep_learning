from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from mae_ast.config import load_config
from mae_ast.models.full_sequence_proxy import FullSequenceProxy
from mae_ast.models.mae_ast import MAEASTModel
from mae_ast.models.transformer import (
    build_transformer_encoder,
    forward_transformer_encoder,
)


def _small_cfg(**overrides):
    dotlist = [
        "model.encoder.layers=1",
        "model.encoder.embed_dim=32",
        "model.encoder.num_heads=4",
        "model.encoder.mlp_ratio=2.0",
        "model.decoder.layers=1",
        "model.decoder.embed_dim=32",
        "model.decoder.num_heads=4",
        "model.decoder.mlp_ratio=2.0",
        "training.amp=false",
    ]

    for key, value in overrides.items():
        dotlist.append(
            f"{key}={value}"
        )

    return load_config(
        overrides=dotlist
    )


def test_transformer_dropouts_are_configured_separately():
    cfg = OmegaConf.create(
        {
            "layers": 1,
            "embed_dim": 32,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.11,
            "attention_dropout": 0.22,
            "activation_dropout": 0.33,
            "layerdrop": 0.0,
            "norm_first": False,
        }
    )

    encoder = build_transformer_encoder(
        cfg
    )

    layer = encoder.layers[0]

    assert layer.norm_first is False
    assert layer.self_attn.dropout == 0.22
    assert layer.dropout.p == 0.33
    assert layer.dropout1.p == 0.11
    assert layer.dropout2.p == 0.11


def test_layerdrop_skips_layers_only_during_training():
    cfg = OmegaConf.create(
        {
            "layers": 2,
            "embed_dim": 16,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "layerdrop": 1.0,
            "norm_first": True,
        }
    )

    encoder = build_transformer_encoder(
        cfg
    )
    norm = nn.LayerNorm(16)

    call_count = {"value": 0}

    def _count_call(*_):
        call_count["value"] += 1

    hooks = [
        layer.register_forward_hook(
            _count_call
        )
        for layer in encoder.layers
    ]

    tokens = torch.randn(
        2,
        5,
        16,
    )

    encoder.train()
    output_train = forward_transformer_encoder(
        encoder=encoder,
        norm=norm,
        tokens=tokens,
        cfg=cfg,
    )

    assert output_train.shape == tokens.shape
    assert call_count["value"] == 0

    encoder.eval()
    output_eval = forward_transformer_encoder(
        encoder=encoder,
        norm=norm,
        tokens=tokens,
        cfg=cfg,
    )

    assert output_eval.shape == tokens.shape
    assert call_count["value"] == 2

    for hook in hooks:
        hook.remove()


def test_mae_and_proxy_expose_paper_dropout_configuration():
    cfg = _small_cfg(
        **{
            "model.dropout_input": 0.1,
            "model.encoder.dropout": 0.1,
            "model.encoder.attention_dropout": 0.1,
            "model.encoder.activation_dropout": 0.0,
            "model.encoder.layerdrop": 0.05,
            "model.encoder.norm_first": "false",
            "model.decoder.dropout": 0.1,
            "model.decoder.attention_dropout": 0.1,
            "model.decoder.activation_dropout": 0.0,
            "model.decoder.layerdrop": 0.0,
            "model.decoder.norm_first": "false",
        }
    )

    mae = MAEASTModel(cfg)
    proxy = FullSequenceProxy(cfg)

    assert mae.dropout_input.p == 0.1
    assert proxy.dropout_input.p == 0.1

    for model in (mae, proxy):
        encoder_layer = model.encoder.layers[0]
        decoder_layer = model.decoder.layers[0]

        assert encoder_layer.norm_first is False
        assert encoder_layer.self_attn.dropout == 0.1
        assert encoder_layer.dropout.p == 0.0
        assert encoder_layer.dropout1.p == 0.1
        assert encoder_layer.dropout2.p == 0.1

        assert decoder_layer.norm_first is False
        assert decoder_layer.self_attn.dropout == 0.1
        assert decoder_layer.dropout.p == 0.0


def test_dropout_input_is_used_only_in_pretraining_path():
    cfg = _small_cfg()
    model = MAEASTModel(cfg)

    class CountingIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x

    counter = CountingIdentity()
    model.dropout_input = counter

    model.eval()

    pretrain_input = torch.randn(
        1,
        int(cfg.audio.pretrain_target_frames),
        int(cfg.audio.n_mels),
    )

    finetune_input = torch.randn(
        1,
        int(cfg.audio.finetune_target_frames),
        int(cfg.audio.n_mels),
    )

    with torch.no_grad():
        model.forward_pretrain(
            pretrain_input
        )

    assert counter.calls == 1

    with torch.no_grad():
        model.forward_finetune(
            finetune_input
        )

    assert counter.calls == 1


def test_checkpoint_parameter_names_remain_encoder_layers_compatible():
    cfg = _small_cfg()
    model = MAEASTModel(cfg)

    keys = set(
        model.state_dict().keys()
    )

    assert "encoder.layers.0.self_attn.in_proj_weight" in keys
    assert "decoder.layers.0.self_attn.in_proj_weight" in keys
    assert "encoder_norm.weight" in keys
    assert "decoder_norm.weight" in keys
