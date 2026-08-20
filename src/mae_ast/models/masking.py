"""
Strategie di masking utilizzate in MAE-AST.

Strategie supportate:

- random:
    maschera token scelti casualmente.

- chunk:
    maschera blocchi bidimensionali contigui sulla griglia
    frequenza-tempo delle patch.

- span:
    maschera segmenti contigui lungo la sequenza 1D.

Convenzioni:

    patches:
        (N, D) oppure (B, N, D)

    mask:
        (N,) oppure (B, N)

        0 = token visibile
        1 = token mascherato

    ids_keep:
        indici originali dei token visibili.

    ids_mask:
        indici originali dei token mascherati.

    ids_restore:
        indici necessari per ricostruire l'ordine originale
        della sequenza dopo aver disposto prima i token visibili
        e poi quelli mascherati.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class MaskingOutput:
    """
    Output prodotto da una strategia di masking.
    """

    visible_patches: torch.Tensor
    mask: torch.Tensor
    ids_keep: torch.Tensor
    ids_mask: torch.Tensor
    ids_restore: torch.Tensor


def _as_batched_patches(
        patches: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """
    Converte le patch nella forma batch (B, N, D).

    Restituisce anche un booleano che indica se l'input originale
    rappresentava un singolo esempio.
    """

    if patches.ndim == 2:
        return patches.unsqueeze(0), True

    if patches.ndim != 3:
        raise ValueError(
            "Le patch devono avere shape (N, D) oppure (B, N, D), "
            f"trovato {tuple(patches.shape)}"
        )

    return patches, False


def _restore_singleton_output(
        output: MaskingOutput,
        squeeze: bool,
) -> MaskingOutput:
    """
    Rimuove la dimensione batch quando l'input originale
    era costituito da un singolo esempio.
    """

    if not squeeze:
        return output

    return MaskingOutput(
        visible_patches=output.visible_patches.squeeze(0),
        mask=output.mask.squeeze(0),
        ids_keep=output.ids_keep.squeeze(0),
        ids_mask=output.ids_mask.squeeze(0),
        ids_restore=output.ids_restore.squeeze(0),
    )


def _build_ids_from_mask(
        mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Costruisce gli indici associati a una maschera binaria.

    Args:
        mask:
            Tensor di shape (B, N), dove:

            0 = token visibile
            1 = token mascherato

    Returns:
        ids_keep:
            Indici originali dei token visibili.

        ids_mask:
            Indici originali dei token mascherati.

        ids_restore:
            Permutazione inversa necessaria per ripristinare
            l'ordine originale della sequenza.
    """

    batch_size, num_patches = mask.shape
    device = mask.device

    ids_keep_list = []
    ids_mask_list = []
    ids_restore_list = []

    for batch_idx in range(batch_size):

        ids_keep = torch.nonzero(
            mask[batch_idx] == 0,
            as_tuple=False,
            ).squeeze(1)

        ids_mask = torch.nonzero(
            mask[batch_idx] == 1,
            as_tuple=False,
            ).squeeze(1)

        # Sequenza temporanea:
        #
        # [token visibili | token mascherati]
        ids_shuffle = torch.cat(
            [ids_keep, ids_mask],
            dim=0,
        )

        # Costruiamo la permutazione inversa.
        ids_restore = torch.empty(
            num_patches,
            dtype=torch.long,
            device=device,
        )

        ids_restore[ids_shuffle] = torch.arange(
            num_patches,
            device=device,
        )

        ids_keep_list.append(ids_keep)
        ids_mask_list.append(ids_mask)
        ids_restore_list.append(ids_restore)

    return (
        torch.stack(ids_keep_list, dim=0),
        torch.stack(ids_mask_list, dim=0),
        torch.stack(ids_restore_list, dim=0),
    )


def apply_keep_indices(
        patches: torch.Tensor,
        ids_keep: torch.Tensor,
) -> torch.Tensor:
    """
    Estrae dalle patch originali solamente i token visibili.
    """

    x, unbatched = _as_batched_patches(patches)

    batch_size, _, patch_dim = x.shape

    # Se ids_keep arriva da un singolo esempio, aggiungiamo
    # temporaneamente la dimensione batch.
    if ids_keep.ndim == 1:
        ids_keep = ids_keep.unsqueeze(0)

    gather_idx = ids_keep.unsqueeze(-1).expand(
        batch_size,
        ids_keep.shape[1],
        patch_dim,
    )

    visible_patches = torch.gather(
        x,
        dim=1,
        index=gather_idx,
    )

    if unbatched:
        visible_patches = visible_patches.squeeze(0)

    return visible_patches


def _repeat_single_mask(
        mask_1d: torch.Tensor,
        batch_size: int,
) -> torch.Tensor:
    """
    Replica la stessa maschera su tutti gli esempi del batch.
    """

    if mask_1d.ndim != 1:
        raise ValueError(
            f"Era attesa una maschera 1D, trovata {tuple(mask_1d.shape)}"
        )

    return mask_1d.unsqueeze(0).expand(
        batch_size,
        -1,
    ).clone()


def _choose_chunk_size(
        chunk_sizes: Iterable[int],
        generator: torch.Generator | None,
        device: torch.device,
) -> int:
    """
    Estrae casualmente la dimensione del prossimo blocco da mascherare.
    """

    chunk_sizes = list(chunk_sizes)

    if not chunk_sizes:
        raise ValueError("chunk_sizes non può essere vuoto.")

    if any(chunk_size <= 0 for chunk_size in chunk_sizes):
        raise ValueError(
            "Tutte le dimensioni presenti in chunk_sizes devono essere positive."
        )

    index = torch.randint(
        low=0,
        high=len(chunk_sizes),
        size=(1,),
        generator=generator,
        device=device,
    ).item()

    return int(chunk_sizes[index])


def _sample_chunk_mask_1d(
        num_patches: int,
        grid_h: int,
        grid_w: int,
        num_to_mask: int,
        chunk_sizes: Iterable[int],
        generator: torch.Generator | None,
        device: torch.device,
) -> torch.Tensor:
    """
    Genera una singola maschera chunk sulla griglia delle patch.
    """

    flat_mask = torch.zeros(
        num_patches,
        dtype=torch.long,
        device=device,
    )

    while int(flat_mask.sum().item()) < num_to_mask:

        chunk_size = _choose_chunk_size(
            chunk_sizes,
            generator=generator,
            device=device,
        )

        chunk_h = min(chunk_size, grid_h)
        chunk_w = min(chunk_size, grid_w)

        top = torch.randint(
            0,
            grid_h - chunk_h + 1,
            (1,),
            generator=generator,
            device=device,
            ).item()

        left = torch.randint(
            0,
            grid_w - chunk_w + 1,
            (1,),
            generator=generator,
            device=device,
            ).item()

        for freq_idx in range(top, top + chunk_h):
            for time_idx in range(left, left + chunk_w):

                token_idx = (
                        freq_idx * grid_w
                        + time_idx
                )

                flat_mask[token_idx] = 1

    # A causa della sovrapposizione dei chunk possiamo superare
    # leggermente il numero desiderato di token mascherati.
    #
    # In quel caso ne manteniamo esattamente num_to_mask.
    masked_indices = torch.nonzero(
        flat_mask == 1,
        as_tuple=False,
        ).squeeze(1)

    if masked_indices.numel() > num_to_mask:

        permutation = torch.randperm(
            masked_indices.numel(),
            generator=generator,
            device=device,
        )

        selected_indices = masked_indices[
            permutation[:num_to_mask]
        ]

        flat_mask.zero_()
        flat_mask[selected_indices] = 1

    return flat_mask


def _sample_span_mask_1d(
        num_patches: int,
        num_to_mask: int,
        span_length: int,
        generator: torch.Generator | None,
        device: torch.device,
) -> torch.Tensor:
    """
    Genera una singola maschera costituita da span contigui.
    """

    flat_mask = torch.zeros(
        num_patches,
        dtype=torch.long,
        device=device,
    )

    while int(flat_mask.sum().item()) < num_to_mask:

        start = torch.randint(
            0,
            num_patches,
            (1,),
            generator=generator,
            device=device,
        ).item()

        end = min(
            start + span_length,
            num_patches,
            )

        flat_mask[start:end] = 1

    masked_indices = torch.nonzero(
        flat_mask == 1,
        as_tuple=False,
        ).squeeze(1)

    if masked_indices.numel() > num_to_mask:

        permutation = torch.randperm(
            masked_indices.numel(),
            generator=generator,
            device=device,
        )

        selected_indices = masked_indices[
            permutation[:num_to_mask]
        ]

        flat_mask.zero_()
        flat_mask[selected_indices] = 1

    return flat_mask


def random_mask(
        patches: torch.Tensor,
        mask_ratio: float,
        share_mask_across_batch: bool = False,
        generator: torch.Generator | None = None,
) -> MaskingOutput:
    """
    Applica masking casuale alle patch.

    Con mask_ratio=0.75 e 512 patch:

        384 patch vengono mascherate
        128 patch rimangono visibili
    """

    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(
            f"mask_ratio deve appartenere a [0, 1), trovato {mask_ratio}"
        )

    x, unbatched = _as_batched_patches(patches)

    batch_size, num_patches, patch_dim = x.shape

    num_keep = int(
        round(
            num_patches * (1.0 - mask_ratio)
        )
    )

    # L'encoder deve ricevere almeno un token.
    num_keep = max(
        1,
        min(num_keep, num_patches),
    )

    if share_mask_across_batch and batch_size > 1:

        noise = torch.rand(
            1,
            num_patches,
            device=x.device,
            generator=generator,
        )

        ids_shuffle_single = torch.argsort(
            noise,
            dim=1,
        )

        ids_restore_single = torch.argsort(
            ids_shuffle_single,
            dim=1,
        )

        ids_keep_single = ids_shuffle_single[
                          :, :num_keep
                          ]

        ids_mask_single = ids_shuffle_single[
                          :, num_keep:
                          ]

        ids_keep = ids_keep_single.expand(
            batch_size,
            -1,
        ).clone()

        ids_mask = ids_mask_single.expand(
            batch_size,
            -1,
        ).clone()

        ids_restore = ids_restore_single.expand(
            batch_size,
            -1,
        ).clone()

    else:

        noise = torch.rand(
            batch_size,
            num_patches,
            device=x.device,
            generator=generator,
        )

        ids_shuffle = torch.argsort(
            noise,
            dim=1,
        )

        ids_restore = torch.argsort(
            ids_shuffle,
            dim=1,
        )

        ids_keep = ids_shuffle[
                   :, :num_keep
                   ]

        ids_mask = ids_shuffle[
                   :, num_keep:
                   ]

    gather_idx = ids_keep.unsqueeze(-1).expand(
        batch_size,
        num_keep,
        patch_dim,
    )

    visible_patches = torch.gather(
        x,
        dim=1,
        index=gather_idx,
    )

    # Prima costruiamo la maschera nell'ordine:
    #
    # [token visibili | token mascherati]
    mask = torch.ones(
        batch_size,
        num_patches,
        device=x.device,
        dtype=torch.long,
    )

    mask[:, :num_keep] = 0

    # Poi la riportiamo nell'ordine originale.
    mask = torch.gather(
        mask,
        dim=1,
        index=ids_restore,
    )

    output = MaskingOutput(
        visible_patches=visible_patches,
        mask=mask,
        ids_keep=ids_keep,
        ids_mask=ids_mask,
        ids_restore=ids_restore,
    )

    return _restore_singleton_output(
        output,
        unbatched,
    )


def chunk_mask(
        patches: torch.Tensor,
        grid_h: int,
        grid_w: int,
        mask_ratio: float,
        chunk_sizes: Iterable[int] = (3, 4, 5),
        share_mask_across_batch: bool = False,
        generator: torch.Generator | None = None,
) -> MaskingOutput:
    """
    Applica masking a blocchi sulla griglia 2D delle patch.
    """

    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(
            f"mask_ratio deve appartenere a [0, 1), trovato {mask_ratio}"
        )

    x, unbatched = _as_batched_patches(patches)

    batch_size, num_patches, _ = x.shape

    if grid_h * grid_w != num_patches:
        raise ValueError(
            "Griglia incompatibile con il numero di patch: "
            f"{grid_h} * {grid_w} = {grid_h * grid_w}, "
            f"ma sono presenti {num_patches} patch."
        )

    num_to_mask = int(
        round(
            num_patches * mask_ratio
        )
    )

    num_to_mask = min(
        num_to_mask,
        num_patches - 1,
        )

    device = x.device

    if share_mask_across_batch and batch_size > 1:

        single_mask = _sample_chunk_mask_1d(
            num_patches=num_patches,
            grid_h=grid_h,
            grid_w=grid_w,
            num_to_mask=num_to_mask,
            chunk_sizes=chunk_sizes,
            generator=generator,
            device=device,
        )

        mask = _repeat_single_mask(
            single_mask,
            batch_size,
        )

    else:

        masks = []

        for _ in range(batch_size):

            single_mask = _sample_chunk_mask_1d(
                num_patches=num_patches,
                grid_h=grid_h,
                grid_w=grid_w,
                num_to_mask=num_to_mask,
                chunk_sizes=chunk_sizes,
                generator=generator,
                device=device,
            )

            masks.append(single_mask)

        mask = torch.stack(
            masks,
            dim=0,
        )

    ids_keep, ids_mask, ids_restore = _build_ids_from_mask(
        mask
    )

    visible_patches = apply_keep_indices(
        x,
        ids_keep,
    )

    if not torch.all(
            mask.sum(dim=1) == num_to_mask
    ):
        raise RuntimeError(
            "chunk_mask non ha prodotto il numero atteso "
            "di patch mascherate."
        )

    output = MaskingOutput(
        visible_patches=visible_patches,
        mask=mask,
        ids_keep=ids_keep,
        ids_mask=ids_mask,
        ids_restore=ids_restore,
    )

    return _restore_singleton_output(
        output,
        unbatched,
    )


def span_mask(
        patches: torch.Tensor,
        mask_ratio: float,
        span_length: int = 10,
        share_mask_across_batch: bool = False,
        generator: torch.Generator | None = None,
) -> MaskingOutput:
    """
    Applica masking a segmenti lungo la sequenza 1D.
    """

    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(
            f"mask_ratio deve appartenere a [0, 1), trovato {mask_ratio}"
        )

    if span_length <= 0:
        raise ValueError(
            f"span_length deve essere > 0, trovato {span_length}"
        )

    x, unbatched = _as_batched_patches(patches)

    batch_size, num_patches, _ = x.shape

    num_to_mask = int(
        round(
            num_patches * mask_ratio
        )
    )

    num_to_mask = min(
        num_to_mask,
        num_patches - 1,
        )

    device = x.device

    if share_mask_across_batch and batch_size > 1:

        single_mask = _sample_span_mask_1d(
            num_patches=num_patches,
            num_to_mask=num_to_mask,
            span_length=span_length,
            generator=generator,
            device=device,
        )

        mask = _repeat_single_mask(
            single_mask,
            batch_size,
        )

    else:

        masks = []

        for _ in range(batch_size):

            single_mask = _sample_span_mask_1d(
                num_patches=num_patches,
                num_to_mask=num_to_mask,
                span_length=span_length,
                generator=generator,
                device=device,
            )

            masks.append(single_mask)

        mask = torch.stack(
            masks,
            dim=0,
        )

    ids_keep, ids_mask, ids_restore = _build_ids_from_mask(
        mask
    )

    visible_patches = apply_keep_indices(
        x,
        ids_keep,
    )

    if not torch.all(
            mask.sum(dim=1) == num_to_mask
    ):
        raise RuntimeError(
            "span_mask non ha prodotto il numero atteso "
            "di patch mascherate."
        )

    output = MaskingOutput(
        visible_patches=visible_patches,
        mask=mask,
        ids_keep=ids_keep,
        ids_mask=ids_mask,
        ids_restore=ids_restore,
    )

    return _restore_singleton_output(
        output,
        unbatched,
    )


def mask_patches(
        patches: torch.Tensor,
        strategy: str = "random",
        mask_ratio: float = 0.75,
        grid_h: int | None = None,
        grid_w: int | None = None,
        chunk_sizes: Iterable[int] = (3, 4, 5),
        span_length: int = 10,
        share_mask_across_batch: bool = False,
        generator: torch.Generator | None = None,
) -> MaskingOutput:
    """
    Interfaccia generale per applicare una strategia di masking.

    Args:
        patches:
            Patch di input.

        strategy:
            Una tra:
            - random
            - chunk
            - span

        mask_ratio:
            Percentuale di token da mascherare.

        grid_h, grid_w:
            Dimensioni della griglia, necessarie per chunk masking.

        chunk_sizes:
            Possibili dimensioni dei blocchi chunk.

        span_length:
            Lunghezza degli span per span masking.

        share_mask_across_batch:
            Se True, tutti gli esempi del batch ricevono
            la stessa maschera.

        generator:
            Generatore PyTorch opzionale per rendere
            il campionamento riproducibile.
    """

    strategy = strategy.lower()

    if strategy == "random":
        return random_mask(
            patches=patches,
            mask_ratio=mask_ratio,
            share_mask_across_batch=share_mask_across_batch,
            generator=generator,
        )

    if strategy == "chunk":

        if grid_h is None or grid_w is None:
            raise ValueError(
                "Per utilizzare chunk masking "
                "devono essere specificati grid_h e grid_w."
            )

        return chunk_mask(
            patches=patches,
            grid_h=grid_h,
            grid_w=grid_w,
            mask_ratio=mask_ratio,
            chunk_sizes=chunk_sizes,
            share_mask_across_batch=share_mask_across_batch,
            generator=generator,
        )

    if strategy == "span":
        return span_mask(
            patches=patches,
            mask_ratio=mask_ratio,
            span_length=span_length,
            share_mask_across_batch=share_mask_across_batch,
            generator=generator,
        )

    raise ValueError(
        f"Strategia di masking non supportata: {strategy}"
    )