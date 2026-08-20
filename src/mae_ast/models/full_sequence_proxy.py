"""
Baseline full-sequence utilizzata per il benchmark computazionale.

ATTENZIONE:
questo modello NON rappresenta una replica completa di SSAST.

La differenza controllata rispetto a MAE-AST è:

MAE-AST:
    l'encoder vede solamente i token visibili.

FullSequenceProxy:
    l'encoder vede l'intera sequenza e i token mascherati
    vengono sostituiti con un mask token apprendibile.

Il decoder e le head di pretraining rimangono presenti,
in modo da rendere il confronto compute/memory più corretto.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from mae_ast.models.mae_ast import (
    PretrainOutput,
)
from mae_ast.models.masking import (
    MaskingOutput,
    mask_patches,
)
from mae_ast.models.patching import (
    get_patch_metadata,
    patchify,
)
from mae_ast.models.positional import (
    build_1d_sincos_pos_embed,
)
from mae_ast.models.transformer import (
    build_transformer_encoder,
    forward_transformer_encoder,
)


class FullSequenceProxy(nn.Module):
    """
    Baseline full-sequence per il benchmark.
    """

    def __init__(
            self,
            cfg: DictConfig,
    ):
        super().__init__()

        self.cfg = cfg

        self.audio_cfg = cfg.audio
        self.patch_cfg = cfg.patching
        self.encoder_cfg = cfg.model.encoder
        self.decoder_cfg = cfg.model.decoder
        self.mask_cfg = cfg.masking

        self.pretrain_meta = get_patch_metadata(
            time_frames=int(
                self.audio_cfg.pretrain_target_frames
            ),
            n_mels=int(
                self.audio_cfg.n_mels
            ),
            patch_h=int(
                self.patch_cfg.patch_h
            ),
            patch_w=int(
                self.patch_cfg.patch_w
            ),
        )

        self.patch_dim = (
                int(self.patch_cfg.patch_h)
                * int(self.patch_cfg.patch_w)
        )

        encoder_dim = int(
            self.encoder_cfg.embed_dim
        )

        decoder_dim = int(
            self.decoder_cfg.embed_dim
        )

        # -------------------------------------------------------------
        # Patch embedding
        # -------------------------------------------------------------

        self.patch_embed = nn.Linear(
            self.patch_dim,
            encoder_dim,
        )

        self.dropout_input = nn.Dropout(
            float(
                cfg.model.get(
                    "dropout_input",
                    0.0,
                )
            )
        )

        # -------------------------------------------------------------
        # Mask token utilizzato direttamente nell'encoder
        # -------------------------------------------------------------

        self.encoder_mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                encoder_dim,
            )
        )

        # -------------------------------------------------------------
        # Encoder full-sequence
        # -------------------------------------------------------------

        self.encoder = build_transformer_encoder(
            self.encoder_cfg
        )

        self.encoder_norm = nn.LayerNorm(
            encoder_dim
        )

        # -------------------------------------------------------------
        # Proiezione encoder -> decoder
        # -------------------------------------------------------------

        if encoder_dim != decoder_dim:
            self.encoder_to_decoder = nn.Linear(
                encoder_dim,
                decoder_dim,
            )
        else:
            self.encoder_to_decoder = nn.Identity()

        # -------------------------------------------------------------
        # Decoder
        # -------------------------------------------------------------

        self.decoder = build_transformer_encoder(
            self.decoder_cfg
        )

        self.decoder_norm = nn.LayerNorm(
            decoder_dim
        )

        # -------------------------------------------------------------
        # Stesse head del modello MAE
        # -------------------------------------------------------------

        self.reconstruction_head = nn.Linear(
            decoder_dim,
            self.patch_dim,
        )

        self.pretrain_classification_head = nn.Linear(
            decoder_dim,
            self.patch_dim,
        )

        # -------------------------------------------------------------
        # Positional embedding
        # -------------------------------------------------------------

        self.register_buffer(
            "encoder_pos_embed",
            build_1d_sincos_pos_embed(
                embed_dim=encoder_dim,
                seq_len=self.pretrain_meta.num_patches,
            ),
            persistent=True,
        )

        self.register_buffer(
            "decoder_pos_embed",
            build_1d_sincos_pos_embed(
                embed_dim=decoder_dim,
                seq_len=self.pretrain_meta.num_patches,
            ),
            persistent=True,
        )

        self._initialize_weights()

    def _initialize_weights(
            self,
    ) -> None:
        """Inizializza i layer specifici del modello."""

        nn.init.xavier_uniform_(
            self.patch_embed.weight
        )

        nn.init.zeros_(
            self.patch_embed.bias
        )

        nn.init.normal_(
            self.encoder_mask_token,
            std=0.02,
        )

        if isinstance(
                self.encoder_to_decoder,
                nn.Linear,
        ):
            nn.init.xavier_uniform_(
                self.encoder_to_decoder.weight
            )

            nn.init.zeros_(
                self.encoder_to_decoder.bias
            )

        nn.init.xavier_uniform_(
            self.reconstruction_head.weight
        )

        nn.init.zeros_(
            self.reconstruction_head.bias
        )

        nn.init.xavier_uniform_(
            self.pretrain_classification_head.weight
        )

        nn.init.zeros_(
            self.pretrain_classification_head.bias
        )

    @staticmethod
    def _gather_tokens(
            tokens: torch.Tensor,
            indices: torch.Tensor,
    ) -> torch.Tensor:
        """Estrae token tramite gli indici indicati."""

        batch_size, _, dim = tokens.shape

        gather_indices = (
            indices
            .unsqueeze(-1)
            .expand(
                batch_size,
                indices.shape[1],
                dim,
            )
        )

        return torch.gather(
            tokens,
            dim=1,
            index=gather_indices,
        )

    def _create_mask(
            self,
            patches: torch.Tensor,
            generator: torch.Generator | None,
    ) -> MaskingOutput:
        """Genera la stessa tipologia di masking utilizzata da MAE."""

        return mask_patches(
            patches=patches,
            strategy=str(
                self.mask_cfg.strategy
            ),
            mask_ratio=float(
                self.mask_cfg.ratio
            ),
            grid_h=self.pretrain_meta.n_freq,
            grid_w=self.pretrain_meta.n_time,
            chunk_sizes=tuple(
                self.mask_cfg.chunk_sizes
            ),
            span_length=int(
                self.mask_cfg.span_length
            ),
            share_mask_across_batch=bool(
                self.mask_cfg.shared_across_batch
            ),
            generator=generator,
        )

    def _apply_mask_token(
            self,
            patch_tokens: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sostituisce le patch mascherate con il mask token.

        patch_tokens:
            (B, N, D)

        mask:
            (B, N)
            0 = visibile
            1 = mascherata
        """

        batch_size, num_tokens, dim = (
            patch_tokens.shape
        )

        mask_tokens = (
            self.encoder_mask_token
            .expand(
                batch_size,
                num_tokens,
                dim,
            )
        )

        mask_boolean = (
            mask
            .unsqueeze(-1)
            .bool()
        )

        return torch.where(
            mask_boolean,
            mask_tokens,
            patch_tokens,
        )

    def forward_pretrain(
            self,
            spectrogram: torch.Tensor,
            masking_output: MaskingOutput | None = None,
            generator: torch.Generator | None = None,
    ) -> PretrainOutput:
        """
        Forward pass della baseline full-sequence.
        """

        if spectrogram.ndim != 3:
            raise ValueError(
                "Lo spettrogramma deve avere shape (B, T, F), "
                f"trovato {tuple(spectrogram.shape)}"
            )

        # -------------------------------------------------------------
        # 1. Patchify
        # -------------------------------------------------------------

        patches = patchify(
            spectrogram,
            patch_h=int(
                self.patch_cfg.patch_h
            ),
            patch_w=int(
                self.patch_cfg.patch_w
            ),
        )

        _, num_patches, _ = (
            patches.shape
        )

        if (
                num_patches
                != self.pretrain_meta.num_patches
        ):
            raise ValueError(
                "Numero patch inatteso: "
                f"{num_patches}, atteso "
                f"{self.pretrain_meta.num_patches}."
            )

        # -------------------------------------------------------------
        # 2. Masking
        # -------------------------------------------------------------

        if masking_output is None:
            masking_output = self._create_mask(
                patches,
                generator,
            )

        # -------------------------------------------------------------
        # 3. Patch embedding di TUTTI i token
        # -------------------------------------------------------------

        patch_tokens = self.patch_embed(
            patches
        )

        patch_tokens = self.dropout_input(
            patch_tokens
        )

        # -------------------------------------------------------------
        # 4. Mask token direttamente nell'encoder
        # -------------------------------------------------------------

        encoder_input = self._apply_mask_token(
            patch_tokens=patch_tokens,
            mask=masking_output.mask,
        )

        encoder_pos = (
            self.encoder_pos_embed
            .to(
                device=encoder_input.device,
                dtype=encoder_input.dtype,
            )
        )

        encoder_input = (
                encoder_input
                + encoder_pos
        )

        # -------------------------------------------------------------
        # 5. Encoder su TUTTI i token
        # -------------------------------------------------------------

        encoder_latent = forward_transformer_encoder(
            encoder=self.encoder,
            norm=self.encoder_norm,
            tokens=encoder_input,
            cfg=self.encoder_cfg,
        )

        # -------------------------------------------------------------
        # 6. Decoder
        # -------------------------------------------------------------

        decoder_input = (
            self.encoder_to_decoder(
                encoder_latent
            )
        )

        decoder_pos = (
            self.decoder_pos_embed
            .to(
                device=decoder_input.device,
                dtype=decoder_input.dtype,
            )
        )

        decoder_input = (
                decoder_input
                + decoder_pos
        )

        decoder_latent = forward_transformer_encoder(
            encoder=self.decoder,
            norm=self.decoder_norm,
            tokens=decoder_input,
            cfg=self.decoder_cfg,
        )

        # -------------------------------------------------------------
        # 7. Token mascherati
        # -------------------------------------------------------------

        masked_decoder_latent = (
            self._gather_tokens(
                decoder_latent,
                masking_output.ids_mask,
            )
        )

        target_masked = self._gather_tokens(
            patches,
            masking_output.ids_mask,
        )

        # -------------------------------------------------------------
        # 8. Head
        # -------------------------------------------------------------

        reconstruction_pred = (
            self.reconstruction_head(
                masked_decoder_latent
            )
        )

        classification_pred = (
            self.pretrain_classification_head(
                masked_decoder_latent
            )
        )

        reconstruction_all = (
            self.reconstruction_head(
                decoder_latent
            )
        )

        return PretrainOutput(
            reconstruction_pred=reconstruction_pred,
            classification_pred=classification_pred,
            target_masked=target_masked,
            reconstruction_all=reconstruction_all,
            target_all=patches,
            mask=masking_output.mask,
            ids_keep=masking_output.ids_keep,
            ids_mask=masking_output.ids_mask,
            ids_restore=masking_output.ids_restore,
            encoder_latent=encoder_latent,
            decoder_latent=decoder_latent,
            masked_decoder_latent=masked_decoder_latent,
        )

    def forward(
            self,
            spectrogram: torch.Tensor,
    ) -> PretrainOutput:
        """Interfaccia standard del proxy."""

        return self.forward_pretrain(
            spectrogram
        )