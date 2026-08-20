"""
Funzioni di loss utilizzate nel progetto MAE-AST.

PRETRAINING
-----------
1. Reconstruction loss:
       MSE tra patch ricostruite e patch originali mascherate.

2. Classification / contrastive loss:
       confronto intra-clip tra le rappresentazioni predette
       e le patch originali mascherate.

3. Loss totale:
       reconstruction_weight * reconstruction_loss
       +
       classification_weight * classification_loss

FINE-TUNING
-----------
Cross-entropy supervisionata sulle classi downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from mae_ast.models.mae_ast import (
    FinetuneOutput,
    PretrainOutput,
)


@dataclass
class LossOutput:
    """
    Contiene la loss totale e le sue componenti.
    """

    total: torch.Tensor

    reconstruction: torch.Tensor | None = None
    classification: torch.Tensor | None = None


def reconstruction_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
) -> torch.Tensor:
    """
    Calcola la reconstruction loss tramite MSE.

    La loss viene calcolata solamente sulle patch mascherate.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Shape incompatibili nella reconstruction loss: "
            f"prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )

    return F.mse_loss(
        prediction,
        target,
        reduction="mean",
    )


def mae_ast_classification_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
) -> torch.Tensor:
    """
    Classification / contrastive loss intra-clip di MAE-AST.

    Input:

        prediction:
            (B, N_mask, patch_dim)

        target:
            (B, N_mask, patch_dim)

    Per ogni clip:

        query_i × target_j
                ↓
        matrice N_mask × N_mask
                ↓
        log_softmax
                ↓
        elementi diagonali = coppie positive

    Gli altri token mascherati della stessa clip
    rappresentano i negativi.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Shape incompatibili nella classification loss: "
            f"prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )

    if prediction.ndim != 3:
        raise ValueError(
            "La classification loss richiede tensori "
            "con shape (B, N_mask, D)."
        )

    # Similarità tra ogni predizione e ogni target
    # all'interno della stessa clip.
    similarities = (
            prediction
            @ target.transpose(-1, -2)
    )

    # Distribuzione di probabilità sui possibili target.
    log_probabilities = F.log_softmax(
        similarities,
        dim=-1,
    )

    # La diagonale contiene le coppie corrette:
    #
    # predizione patch i <-> patch originale i
    positive_log_probabilities = (
        torch.diagonal(
            log_probabilities,
            dim1=-2,
            dim2=-1,
        )
    )

    return (
        -positive_log_probabilities.mean()
    )


def pretrain_loss(
        output: PretrainOutput,
        loss_cfg: DictConfig,
) -> LossOutput:
    """
    Calcola la loss completa di pretraining.
    """

    rec_loss = reconstruction_loss(
        prediction=output.reconstruction_pred,
        target=output.target_masked,
    )

    cls_loss = mae_ast_classification_loss(
        prediction=output.classification_pred,
        target=output.target_masked,
    )

    total_loss = (
            float(
                loss_cfg.reconstruction_weight
            )
            * rec_loss
            +
            float(
                loss_cfg.classification_weight
            )
            * cls_loss
    )

    return LossOutput(
        total=total_loss,
        reconstruction=rec_loss,
        classification=cls_loss,
    )


def finetune_loss(
        output: FinetuneOutput,
        labels: torch.Tensor,
        label_smoothing: float = 0.0,
) -> LossOutput:
    """
    Calcola la cross-entropy utilizzata nel fine-tuning.
    """

    if output.logits.ndim != 2:
        raise ValueError(
            "I logits devono avere shape (B, C), "
            f"trovato {tuple(output.logits.shape)}"
        )

    if labels.ndim != 1:
        raise ValueError(
            "Le label devono avere shape (B,), "
            f"trovato {tuple(labels.shape)}"
        )

    classification = F.cross_entropy(
        output.logits,
        labels,
        label_smoothing=float(
            label_smoothing
        ),
    )

    return LossOutput(
        total=classification,
        classification=classification,
    )