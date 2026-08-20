"""
Implementazione principale del modello MAE-AST.

Il modello supporta due modalità:

PRETRAINING
-----------
spettrogramma
    -> patchify
    -> masking
    -> encoder sui soli token visibili
    -> inserimento dei mask token
    -> decoder sulla sequenza completa
    -> reconstruction head
    -> classification head

FINE-TUNING
-----------
spettrogramma
    -> patchify
    -> encoder su tutti i token
    -> mean pooling
    -> classification head

Il decoder viene utilizzato solamente durante il pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig

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


@dataclass
class PretrainOutput:
    """
    Output prodotto dal modello durante il pretraining.
    """

    # Output utilizzati dalle loss.
    reconstruction_pred: torch.Tensor
    classification_pred: torch.Tensor
    target_masked: torch.Tensor

    # Output completi utili per analisi e ricostruzioni.
    reconstruction_all: torch.Tensor
    target_all: torch.Tensor

    # Informazioni sul masking.
    mask: torch.Tensor
    ids_keep: torch.Tensor
    ids_mask: torch.Tensor
    ids_restore: torch.Tensor

    # Rappresentazioni interne.
    encoder_latent: torch.Tensor
    decoder_latent: torch.Tensor
    masked_decoder_latent: torch.Tensor


@dataclass
class FinetuneOutput:
    """
    Output prodotto dal modello durante il fine-tuning.
    """

    logits: torch.Tensor
    pooled: torch.Tensor
    patch_tokens: torch.Tensor
    encoder_latent: torch.Tensor


class MAEASTModel(nn.Module):
    """
    Masked Autoencoding Audio Spectrogram Transformer.
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

        # -------------------------------------------------------------
        # Metadata delle patch
        # -------------------------------------------------------------

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

        self.finetune_meta = get_patch_metadata(
            time_frames=int(
                self.audio_cfg.finetune_target_frames
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

        # Dropout applicato alle patch proiettate durante il pretraining,
        # prima di masking e positional embedding. Nel fine-tuning resta OFF.
        self.dropout_input = nn.Dropout(
            float(
                cfg.model.get(
                    "dropout_input",
                    0.0,
                )
            )
        )

        # -------------------------------------------------------------
        # Encoder Transformer
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
        # Mask token
        # -------------------------------------------------------------

        # Un singolo vettore apprendibile condiviso
        # tra tutte le patch mascherate.
        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                decoder_dim,
            )
        )

        # -------------------------------------------------------------
        # Decoder Transformer
        # -------------------------------------------------------------

        self.decoder = build_transformer_encoder(
            self.decoder_cfg
        )

        self.decoder_norm = nn.LayerNorm(
            decoder_dim
        )

        # -------------------------------------------------------------
        # Head di pretraining
        # -------------------------------------------------------------

        # Ricostruisce i 256 valori della patch originale.
        self.reconstruction_head = nn.Linear(
            decoder_dim,
            self.patch_dim,
        )

        # Produce la rappresentazione utilizzata dalla
        # classification / contrastive loss intra-clip.
        self.pretrain_classification_head = nn.Linear(
            decoder_dim,
            self.patch_dim,
        )

        # -------------------------------------------------------------
        # Head di fine-tuning
        # -------------------------------------------------------------

        self.finetune_head = nn.Linear(
            encoder_dim,
            int(cfg.model.num_classes),
        )

        # -------------------------------------------------------------
        # Positional embedding fissi
        # -------------------------------------------------------------

        self.register_buffer(
            "encoder_pos_embed_pretrain",
            build_1d_sincos_pos_embed(
                embed_dim=encoder_dim,
                seq_len=self.pretrain_meta.num_patches,
            ),
            persistent=True,
        )

        self.register_buffer(
            "encoder_pos_embed_finetune",
            build_1d_sincos_pos_embed(
                embed_dim=encoder_dim,
                seq_len=self.finetune_meta.num_patches,
            ),
            persistent=True,
        )

        self.register_buffer(
            "decoder_pos_embed_pretrain",
            build_1d_sincos_pos_embed(
                embed_dim=decoder_dim,
                seq_len=self.pretrain_meta.num_patches,
            ),
            persistent=True,
        )

        self._initialize_weights()

    # -----------------------------------------------------------------
    # Inizializzazione
    # -----------------------------------------------------------------

    def _initialize_weights(
            self,
    ) -> None:
        """
        Inizializza esplicitamente i layer aggiunti attorno
        ai Transformer.
        """

        nn.init.xavier_uniform_(
            self.patch_embed.weight
        )

        nn.init.zeros_(
            self.patch_embed.bias
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

        nn.init.normal_(
            self.mask_token,
            std=0.02,
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

        nn.init.xavier_uniform_(
            self.finetune_head.weight
        )

        nn.init.zeros_(
            self.finetune_head.bias
        )

    # -----------------------------------------------------------------
    # Utility interne
    # -----------------------------------------------------------------

    def _patchify(
            self,
            spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converte lo spettrogramma nelle patch utilizzate dal modello.
        """

        return patchify(
            spectrogram,
            patch_h=int(
                self.patch_cfg.patch_h
            ),
            patch_w=int(
                self.patch_cfg.patch_w
            ),
        )

    def _encode(
            self,
            tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Esegue l'encoder Transformer con norm/dropout/layerdrop configurabili."""

        return forward_transformer_encoder(
            encoder=self.encoder,
            norm=self.encoder_norm,
            tokens=tokens,
            cfg=self.encoder_cfg,
        )

    def _decode(
            self,
            tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Esegue il decoder Transformer con norm/dropout/layerdrop configurabili."""

        return forward_transformer_encoder(
            encoder=self.decoder,
            norm=self.decoder_norm,
            tokens=tokens,
            cfg=self.decoder_cfg,
        )

    @staticmethod
    def _gather_tokens(
            tokens: torch.Tensor,
            indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estrae token specifici utilizzando gli indici originali.
        """

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
        """
        Applica la strategia di masking definita nella configurazione.
        """

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

    # -----------------------------------------------------------------
    # Pretraining
    # -----------------------------------------------------------------

    def forward_pretrain(
            self,
            spectrogram: torch.Tensor,
            masking_output: MaskingOutput | None = None,
            generator: torch.Generator | None = None,
    ) -> PretrainOutput:
        """
        Forward pass di pretraining.

        La caratteristica fondamentale di MAE-AST è che
        l'encoder elabora solamente i token visibili.
        """

        if spectrogram.ndim != 3:
            raise ValueError(
                "Lo spettrogramma deve avere shape (B, T, F), "
                f"trovato {tuple(spectrogram.shape)}"
            )

        # -------------------------------------------------------------
        # 1. Patchify
        # -------------------------------------------------------------

        patches = self._patchify(
            spectrogram
        )

        batch_size, num_patches, _ = (
            patches.shape
        )

        if (
                num_patches
                != self.pretrain_meta.num_patches
        ):
            raise ValueError(
                "Numero di patch inatteso nel pretraining: "
                f"{num_patches}, atteso "
                f"{self.pretrain_meta.num_patches}."
            )

        # -------------------------------------------------------------
        # 2. Masking
        # -------------------------------------------------------------

        if masking_output is None:
            masking_output = self._create_mask(
                patches=patches,
                generator=generator,
            )

        # -------------------------------------------------------------
        # 3. Linear patch embedding
        # -------------------------------------------------------------

        patch_tokens = self.patch_embed(
            patches
        )

        # Dropout di input della recipe MAE-AST. Viene applicato solamente
        # nel pretraining, coerentemente con il percorso ``mask=True``
        # dell'implementazione pubblica.
        patch_tokens = self.dropout_input(
            patch_tokens
        )

        # -------------------------------------------------------------
        # 4. Positional embedding dell'encoder
        # -------------------------------------------------------------

        encoder_pos = (
            self.encoder_pos_embed_pretrain
            .to(
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
        )

        patch_tokens = (
                patch_tokens
                + encoder_pos
        )

        # -------------------------------------------------------------
        # 5. Teniamo solamente i token visibili
        # -------------------------------------------------------------

        visible_tokens = self._gather_tokens(
            patch_tokens,
            masking_output.ids_keep,
        )

        # -------------------------------------------------------------
        # 6. Encoder
        # -------------------------------------------------------------

        encoder_latent = self._encode(
            visible_tokens
        )

        # -------------------------------------------------------------
        # 7. Proiezione encoder -> decoder
        # -------------------------------------------------------------

        decoder_visible = (
            self.encoder_to_decoder(
                encoder_latent
            )
        )

        # -------------------------------------------------------------
        # 8. Introduzione dei mask token
        # -------------------------------------------------------------

        num_masked = (
            masking_output
            .ids_mask
            .shape[1]
        )

        mask_tokens = self.mask_token.expand(
            batch_size,
            num_masked,
            -1,
        )

        # Sequenza temporanea:
        #
        # [token visibili | mask token]
        decoder_tokens = torch.cat(
            [
                decoder_visible,
                mask_tokens,
            ],
            dim=1,
        )

        # Ripristiniamo la posizione originale
        # di tutti i token.
        decoder_tokens = self._gather_tokens(
            decoder_tokens,
            masking_output.ids_restore,
        )

        # -------------------------------------------------------------
        # 9. Positional embedding del decoder
        # -------------------------------------------------------------

        decoder_pos = (
            self.decoder_pos_embed_pretrain
            .to(
                device=decoder_tokens.device,
                dtype=decoder_tokens.dtype,
            )
        )

        decoder_tokens = (
                decoder_tokens
                + decoder_pos
        )

        # -------------------------------------------------------------
        # 10. Decoder
        # -------------------------------------------------------------

        decoder_latent = self._decode(
            decoder_tokens
        )

        # -------------------------------------------------------------
        # 11. Selezione dei soli token mascherati
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
        # 12. Head di pretraining
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

        # Output completo della reconstruction head.
        # Utile successivamente per visualizzare
        # spettrogrammi ricostruiti.
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

    # -----------------------------------------------------------------
    # Fine-tuning
    # -----------------------------------------------------------------

    def forward_finetune(
            self,
            spectrogram: torch.Tensor,
    ) -> FinetuneOutput:
        """
        Forward pass per il fine-tuning supervisionato.

        Durante il fine-tuning:
        - non viene applicato masking;
        - il decoder non viene utilizzato;
        - tutti i token passano attraverso l'encoder;
        - viene applicato mean pooling;
        - la classification head produce i logits.
        """

        if spectrogram.ndim != 3:
            raise ValueError(
                "Lo spettrogramma deve avere shape (B, T, F), "
                f"trovato {tuple(spectrogram.shape)}"
            )

        patches = self._patchify(
            spectrogram
        )

        _, num_patches, _ = (
            patches.shape
        )

        if (
                num_patches
                != self.finetune_meta.num_patches
        ):
            raise ValueError(
                "Numero di patch inatteso nel fine-tuning: "
                f"{num_patches}, atteso "
                f"{self.finetune_meta.num_patches}."
            )

        # Patch embedding.
        tokens = self.patch_embed(
            patches
        )

        # Positional embedding.
        positional_embedding = (
            self.encoder_pos_embed_finetune
            .to(
                device=tokens.device,
                dtype=tokens.dtype,
            )
        )

        tokens = (
                tokens
                + positional_embedding
        )

        # Encoder su tutti i token.
        encoder_latent = self._encode(
            tokens
        )

        # Mean pooling sulla dimensione dei token.
        pooled = encoder_latent.mean(
            dim=1
        )

        # Classificazione finale.
        logits = self.finetune_head(
            pooled
        )

        return FinetuneOutput(
            logits=logits,
            pooled=pooled,
            patch_tokens=patches,
            encoder_latent=encoder_latent,
        )

    def forward(
            self,
            spectrogram: torch.Tensor,
            mode: str = "pretrain",
    ):
        """
        Interfaccia generale del modello.
        """

        if mode == "pretrain":
            return self.forward_pretrain(
                spectrogram
            )

        if mode == "finetune":
            return self.forward_finetune(
                spectrogram
            )

        raise ValueError(
            f"Modalità non supportata: {mode}"
        )