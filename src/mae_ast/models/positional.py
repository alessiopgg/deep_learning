"""
Utility per i positional embedding sinusoidali 1D di MAE-AST.

Nel paper MAE-AST vengono utilizzati positional embedding sinusoidali
fissi per rappresentare la posizione dei token nella sequenza.

Gli embedding non sono parametri apprendibili:
vengono calcolati una volta e poi utilizzati dal modello.
"""

from __future__ import annotations

import math

import torch


def build_1d_sincos_pos_embed(
        embed_dim: int,
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Costruisce positional embedding sinusoidali 1D fissi.

    Args:
        embed_dim:
            Dimensione dell'embedding di ogni token.

        seq_len:
            Numero massimo di posizioni nella sequenza.

        device:
            Device su cui creare il tensore.
            Se non specificato viene utilizzata la CPU.

        dtype:
            Tipo numerico del tensore.

    Returns:
        Tensor di shape:

            (1, seq_len, embed_dim)

    Nota:
        embed_dim deve essere pari perché metà delle dimensioni
        utilizza il seno e metà utilizza il coseno.
    """

    if embed_dim <= 0:
        raise ValueError(
            f"embed_dim deve essere maggiore di 0, trovato {embed_dim}"
        )

    if embed_dim % 2 != 0:
        raise ValueError(
            f"embed_dim deve essere pari, trovato {embed_dim}"
        )

    if seq_len <= 0:
        raise ValueError(
            f"seq_len deve essere maggiore di 0, trovato {seq_len}"
        )

    if device is None:
        device = torch.device("cpu")

    # Posizioni della sequenza:
    #
    # 0
    # 1
    # 2
    # ...
    # seq_len - 1
    #
    # Shape: (seq_len, 1)
    position = torch.arange(
        seq_len,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)

    # Frequenze utilizzate dalle funzioni seno/coseno.
    #
    # Shape: (embed_dim / 2,)
    div_term = torch.exp(
        torch.arange(
            0,
            embed_dim,
            2,
            device=device,
            dtype=dtype,
        )
        * (-math.log(10000.0) / embed_dim)
    )

    # Tensor finale degli embedding posizionali.
    #
    # Shape: (seq_len, embed_dim)
    positional_embedding = torch.zeros(
        seq_len,
        embed_dim,
        device=device,
        dtype=dtype,
    )

    positional_embedding[:, 0::2] = torch.sin(
        position * div_term
    )

    positional_embedding[:, 1::2] = torch.cos(
        position * div_term
    )

    # Aggiungiamo la dimensione batch per poter sommare direttamente:
    #
    # token_embeddings:      (B, N, D)
    # positional_embedding:  (1, N, D)
    #
    # PyTorch applicherà il broadcasting sulla dimensione batch.
    return positional_embedding.unsqueeze(0)