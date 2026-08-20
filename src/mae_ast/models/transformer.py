"""
Utility per costruire ed eseguire gli stack Transformer della V2.

Questo modulo mantiene ``nn.TransformerEncoderLayer`` di PyTorch, ma rende
espliciti i dettagli della recipe MAE-AST che il layer standard accorpa:

- dropout residuale;
- attention dropout;
- activation dropout del FFN;
- pre-norm / post-norm;
- layerdrop;
- dropout all'ingresso dello stack Transformer.

Lo stack continua a essere un ``nn.TransformerEncoder`` per preservare i nomi
dei parametri ``*.layers.N.*`` nei checkpoint esistenti.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


def _config_float(
        cfg: DictConfig,
        key: str,
        default: float,
) -> float:
    """Legge un float dalla config mantenendo retrocompatibilità."""

    return float(
        cfg.get(
            key,
            default,
        )
    )


def _config_bool(
        cfg: DictConfig,
        key: str,
        default: bool,
) -> bool:
    """Legge un booleano dalla config mantenendo retrocompatibilità."""

    return bool(
        cfg.get(
            key,
            default,
        )
    )


def build_transformer_encoder(
        cfg: DictConfig,
) -> nn.TransformerEncoder:
    """
    Costruisce uno stack Transformer configurabile in stile MAE-AST.

    ``nn.TransformerEncoderLayer`` usa un solo argomento ``dropout`` per più
    punti del blocco. Dopo la costruzione separiamo esplicitamente:

    - ``self_attn.dropout`` -> attention dropout;
    - ``dropout`` -> activation dropout dopo GELU nel FFN;
    - ``dropout1`` / ``dropout2`` -> dropout dei rami residuali.

    I campi aggiuntivi hanno default retrocompatibili con la V2 precedente.
    """

    embed_dim = int(
        cfg.embed_dim
    )

    residual_dropout = _config_float(
        cfg,
        "dropout",
        0.0,
    )

    attention_dropout = _config_float(
        cfg,
        "attention_dropout",
        residual_dropout,
    )

    activation_dropout = _config_float(
        cfg,
        "activation_dropout",
        residual_dropout,
    )

    norm_first = _config_bool(
        cfg,
        "norm_first",
        True,
    )

    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=int(
            cfg.num_heads
        ),
        dim_feedforward=int(
            embed_dim
            * float(
                cfg.mlp_ratio
            )
        ),
        dropout=residual_dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )

    # Dropout dei pesi di attenzione.
    layer.self_attn.dropout = (
        attention_dropout
    )

    # Dropout dopo GELU nel ramo FFN.
    layer.dropout.p = (
        activation_dropout
    )

    # ``dropout1`` e ``dropout2`` restano al valore residuale passato
    # al costruttore, quindi non serve modificarli manualmente.

    return nn.TransformerEncoder(
        layer,
        num_layers=int(
            cfg.layers
        ),
        enable_nested_tensor=False,
    )


def forward_transformer_encoder(
        encoder: nn.TransformerEncoder,
        norm: nn.LayerNorm,
        tokens: torch.Tensor,
        cfg: DictConfig,
) -> torch.Tensor:
    """
    Esegue lo stack Transformer con semantica compatibile MAE-AST/fairseq.

    Comportamento:

    - ``norm_first=False``: LayerNorm globale prima dello stack;
    - dropout ``cfg.dropout`` all'ingresso dello stack;
    - LayerDrop indipendente per ogni blocco solo in training;
    - ``norm_first=True``: LayerNorm globale dopo lo stack.

    Con i default storici V2 (``norm_first=True``, dropout/layerdrop a zero)
    il comportamento resta equivalente a quello precedente.
    """

    norm_first = _config_bool(
        cfg,
        "norm_first",
        True,
    )

    residual_dropout = _config_float(
        cfg,
        "dropout",
        0.0,
    )

    layerdrop = _config_float(
        cfg,
        "layerdrop",
        0.0,
    )

    if not 0.0 <= layerdrop <= 1.0:
        raise ValueError(
            "layerdrop deve essere compreso tra 0 e 1, "
            f"trovato {layerdrop}."
        )

    if not norm_first:
        tokens = norm(tokens)

    tokens = nn.functional.dropout(
        tokens,
        p=residual_dropout,
        training=encoder.training,
    )

    for layer in encoder.layers:
        if (
                encoder.training
                and layerdrop > 0.0
        ):
            keep_layer = bool(
                torch.rand(
                    ()
                ).item()
                >= layerdrop
            )

            if not keep_layer:
                continue

        tokens = layer(tokens)

    if norm_first:
        tokens = norm(tokens)

    return tokens
