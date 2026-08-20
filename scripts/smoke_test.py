"""
Smoke test end-to-end della struttura MAE-AST v2.

Il test utilizza dati sintetici e un Transformer molto piccolo
per verificare rapidamente che i diversi moduli siano collegati
correttamente.

Controlliamo:

- caricamento OmegaConf;
- costruzione MAE-AST;
- pretraining forward;
- reconstruction loss;
- classification loss;
- backward;
- fine-tuning forward;
- cross-entropy;
- backward;
- FullSequenceProxy;
- backward del proxy.

NON è un test delle prestazioni del modello.
"""

from __future__ import annotations

import torch

from mae_ast.config import load_config
from mae_ast.losses import (
    finetune_loss,
    pretrain_loss,
)
from mae_ast.models.full_sequence_proxy import (
    FullSequenceProxy,
)
from mae_ast.models.mae_ast import MAEASTModel
from mae_ast.training.utils import (
    get_device,
    set_seed,
)


def check_finite(
        name: str,
        tensor: torch.Tensor,
) -> None:
    """Controlla che un tensore non contenga NaN o inf."""

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contiene NaN o valori infiniti."
        )


def main() -> None:

    print("=" * 65)
    print("MAE-AST V2 — SMOKE TEST")
    print("=" * 65)

    # -------------------------------------------------------------
    # Configurazione ridotta
    # -------------------------------------------------------------
    #
    # Non utilizziamo il modello 6 × 768 completo perché qui
    # vogliamo controllare solamente che la pipeline funzioni.
    #
    # Manteniamo però:
    #
    #   1024 × 128
    #       ↓
    #   512 patch
    #       ↓
    #   masking 75%
    #
    # quindi la struttura della sequenza rimane quella reale.
    # -------------------------------------------------------------

    cfg = load_config(
        overrides=[
            "model.encoder.layers=1",
            "model.encoder.embed_dim=96",
            "model.encoder.num_heads=4",
            "model.encoder.mlp_ratio=2.0",
            "model.decoder.layers=1",
            "model.decoder.embed_dim=96",
            "model.decoder.num_heads=4",
            "model.decoder.mlp_ratio=2.0",
            "training.amp=false",
            "masking.strategy=chunk",
            "masking.ratio=0.75",
            "masking.shared_across_batch=true",
        ]
    )

    set_seed(
        int(cfg.experiment.seed)
    )

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    batch_size = 2

    # =============================================================
    # 1. PRETRAINING MAE-AST
    # =============================================================

    print(
        "\n[1/3] MAE-AST pretraining"
    )

    model = MAEASTModel(
        cfg
    ).to(device)

    pretrain_input = torch.randn(
        batch_size,
        int(
            cfg.audio.pretrain_target_frames
        ),
        int(
            cfg.audio.n_mels
        ),
        device=device,
    )

    output = model.forward_pretrain(
        pretrain_input
    )

    print(
        "  reconstruction_pred:",
        tuple(
            output.reconstruction_pred.shape
        ),
    )

    print(
        "  classification_pred:",
        tuple(
            output.classification_pred.shape
        ),
    )

    print(
        "  target_masked:",
        tuple(
            output.target_masked.shape
        ),
    )

    print(
        "  encoder_latent:",
        tuple(
            output.encoder_latent.shape
        ),
    )

    print(
        "  decoder_latent:",
        tuple(
            output.decoder_latent.shape
        ),
    )

    # Con 512 patch e masking 75%:
    #
    # visible = 128
    # masked  = 384

    assert (
            output.reconstruction_pred.shape
            == (
                batch_size,
                384,
                256,
            )
    )

    assert (
            output.classification_pred.shape
            == (
                batch_size,
                384,
                256,
            )
    )

    assert (
            output.target_masked.shape
            == (
                batch_size,
                384,
                256,
            )
    )

    assert (
            output.encoder_latent.shape
            == (
                batch_size,
                128,
                96,
            )
    )

    assert (
            output.decoder_latent.shape
            == (
                batch_size,
                512,
                96,
            )
    )

    losses = pretrain_loss(
        output,
        cfg.loss,
    )

    check_finite(
        "pretrain total loss",
        losses.total,
    )

    check_finite(
        "reconstruction loss",
        losses.reconstruction,
    )

    check_finite(
        "classification loss",
        losses.classification,
    )

    print(
        f"  reconstruction loss: "
        f"{losses.reconstruction.item():.6f}"
    )

    print(
        f"  classification loss: "
        f"{losses.classification.item():.6f}"
    )

    print(
        f"  total loss: "
        f"{losses.total.item():.6f}"
    )

    losses.total.backward()

    if model.patch_embed.weight.grad is None:
        raise RuntimeError(
            "Il gradiente non è arrivato alla patch embedding."
        )

    check_finite(
        "patch embedding gradient",
        model.patch_embed.weight.grad,
    )

    print(
        "  backward: OK"
    )

    # =============================================================
    # 2. FINE-TUNING
    # =============================================================

    print(
        "\n[2/3] MAE-AST fine-tuning"
    )

    model.zero_grad(
        set_to_none=True
    )

    finetune_input = torch.randn(
        batch_size,
        int(
            cfg.audio.finetune_target_frames
        ),
        int(
            cfg.audio.n_mels
        ),
        device=device,
    )

    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
        device=device,
    )

    finetune_output = (
        model.forward_finetune(
            finetune_input
        )
    )

    print(
        "  encoder_latent:",
        tuple(
            finetune_output
            .encoder_latent
            .shape
        ),
    )

    print(
        "  pooled:",
        tuple(
            finetune_output.pooled.shape
        ),
    )

    print(
        "  logits:",
        tuple(
            finetune_output.logits.shape
        ),
    )

    assert (
            finetune_output
            .encoder_latent
            .shape
            == (
                batch_size,
                256,
                96,
            )
    )

    assert (
            finetune_output.pooled.shape
            == (
                batch_size,
                96,
            )
    )

    assert (
            finetune_output.logits.shape
            == (
                batch_size,
                int(
                    cfg.model.num_classes
                ),
            )
    )

    ft_loss = finetune_loss(
        output=finetune_output,
        labels=labels,
        label_smoothing=float(
            cfg.loss.label_smoothing
        ),
    )

    check_finite(
        "finetune loss",
        ft_loss.total,
    )

    print(
        f"  cross entropy: "
        f"{ft_loss.total.item():.6f}"
    )

    ft_loss.total.backward()

    if (
            model.finetune_head
                    .weight
                    .grad
            is None
    ):
        raise RuntimeError(
            "Il gradiente non è arrivato "
            "alla finetune head."
        )

    print(
        "  backward: OK"
    )

    # =============================================================
    # 3. FULL-SEQUENCE PROXY
    # =============================================================

    print(
        "\n[3/3] FullSequenceProxy"
    )

    proxy = FullSequenceProxy(
        cfg
    ).to(device)

    proxy_output = (
        proxy.forward_pretrain(
            pretrain_input
        )
    )

    print(
        "  encoder_latent:",
        tuple(
            proxy_output.encoder_latent.shape
        ),
    )

    print(
        "  decoder_latent:",
        tuple(
            proxy_output.decoder_latent.shape
        ),
    )

    # Differenza fondamentale:
    #
    # MAE encoder:
    #     128 token
    #
    # Proxy encoder:
    #     512 token

    assert (
            proxy_output.encoder_latent.shape
            == (
                batch_size,
                512,
                96,
            )
    )

    assert (
            proxy_output.decoder_latent.shape
            == (
                batch_size,
                512,
                96,
            )
    )

    proxy_losses = pretrain_loss(
        proxy_output,
        cfg.loss,
    )

    check_finite(
        "proxy total loss",
        proxy_losses.total,
    )

    print(
        f"  total loss: "
        f"{proxy_losses.total.item():.6f}"
    )

    proxy_losses.total.backward()

    if (
            proxy.patch_embed
                    .weight
                    .grad
            is None
    ):
        raise RuntimeError(
            "Il gradiente non attraversa "
            "il FullSequenceProxy."
        )

    print(
        "  backward: OK"
    )

    # =============================================================
    # RISULTATO
    # =============================================================

    print("\n" + "=" * 65)
    print("SMOKE TEST SUPERATO")
    print("=" * 65)

    print(
        "\nPipeline verificata:"
    )

    print(
        "  patching -> masking -> encoder -> decoder "
        "-> heads -> losses -> backward"
    )

    print(
        "  encoder -> mean pooling -> downstream head "
        "-> cross entropy -> backward"
    )

    print(
        "  full-sequence proxy -> loss -> backward"
    )


if __name__ == "__main__":
    main()