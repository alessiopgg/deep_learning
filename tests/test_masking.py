import pytest
import torch

from mae_ast.models.masking import (
    chunk_mask,
    mask_patches,
    random_mask,
    span_mask,
)


NUM_PATCHES = 512
PATCH_DIM = 256
MASK_RATIO = 0.75

NUM_MASKED = 384
NUM_VISIBLE = 128


def create_patches(batch_size: int = 2) -> torch.Tensor:
    """
    Crea un batch sintetico di patch per i test.
    """

    return torch.randn(
        batch_size,
        NUM_PATCHES,
        PATCH_DIM,
    )


def test_random_mask_shapes():
    patches = create_patches()

    output = random_mask(
        patches,
        mask_ratio=MASK_RATIO,
    )

    assert output.visible_patches.shape == (
        2,
        NUM_VISIBLE,
        PATCH_DIM,
    )

    assert output.mask.shape == (
        2,
        NUM_PATCHES,
    )

    assert output.ids_keep.shape == (
        2,
        NUM_VISIBLE,
    )

    assert output.ids_mask.shape == (
        2,
        NUM_MASKED,
    )

    assert output.ids_restore.shape == (
        2,
        NUM_PATCHES,
    )


def test_random_mask_exact_ratio():
    patches = create_patches()

    output = random_mask(
        patches,
        mask_ratio=MASK_RATIO,
    )

    masked_per_sample = output.mask.sum(dim=1)

    assert torch.all(
        masked_per_sample == NUM_MASKED
    )


def test_chunk_mask_exact_ratio():
    patches = create_patches()

    output = chunk_mask(
        patches,
        grid_h=8,
        grid_w=64,
        mask_ratio=MASK_RATIO,
        chunk_sizes=(3, 4, 5),
    )

    assert output.visible_patches.shape == (
        2,
        NUM_VISIBLE,
        PATCH_DIM,
    )

    assert torch.all(
        output.mask.sum(dim=1) == NUM_MASKED
    )


def test_span_mask_exact_ratio():
    patches = create_patches()

    output = span_mask(
        patches,
        mask_ratio=MASK_RATIO,
        span_length=10,
    )

    assert output.visible_patches.shape == (
        2,
        NUM_VISIBLE,
        PATCH_DIM,
    )

    assert torch.all(
        output.mask.sum(dim=1) == NUM_MASKED
    )


def test_shared_random_mask():
    patches = create_patches()

    output = random_mask(
        patches,
        mask_ratio=MASK_RATIO,
        share_mask_across_batch=True,
    )

    assert torch.equal(
        output.mask[0],
        output.mask[1],
    )

    assert torch.equal(
        output.ids_keep[0],
        output.ids_keep[1],
    )

    assert torch.equal(
        output.ids_mask[0],
        output.ids_mask[1],
    )


def test_shared_chunk_mask():
    patches = create_patches()

    output = chunk_mask(
        patches,
        grid_h=8,
        grid_w=64,
        mask_ratio=MASK_RATIO,
        chunk_sizes=(3, 4, 5),
        share_mask_across_batch=True,
    )

    assert torch.equal(
        output.mask[0],
        output.mask[1],
    )


def test_deterministic_random_mask():
    patches = create_patches()

    generator_1 = torch.Generator().manual_seed(42)
    generator_2 = torch.Generator().manual_seed(42)

    output_1 = random_mask(
        patches,
        mask_ratio=MASK_RATIO,
        generator=generator_1,
    )

    output_2 = random_mask(
        patches,
        mask_ratio=MASK_RATIO,
        generator=generator_2,
    )

    assert torch.equal(
        output_1.mask,
        output_2.mask,
    )


def test_visible_patches_correspond_to_ids_keep():
    patches = create_patches()

    output = random_mask(
        patches,
        mask_ratio=MASK_RATIO,
    )

    expected_visible = torch.gather(
        patches,
        dim=1,
        index=output.ids_keep.unsqueeze(-1).expand(
            -1,
            -1,
            PATCH_DIM,
        ),
    )

    assert torch.equal(
        output.visible_patches,
        expected_visible,
    )


def test_mask_patches_dispatcher():
    patches = create_patches()

    output = mask_patches(
        patches,
        strategy="chunk",
        mask_ratio=MASK_RATIO,
        grid_h=8,
        grid_w=64,
    )

    assert output.visible_patches.shape == (
        2,
        NUM_VISIBLE,
        PATCH_DIM,
    )


def test_invalid_mask_strategy():
    patches = create_patches()

    with pytest.raises(ValueError):
        mask_patches(
            patches,
            strategy="non_esiste",
            mask_ratio=MASK_RATIO,
        )


def test_invalid_chunk_grid():
    patches = create_patches()

    with pytest.raises(ValueError):
        chunk_mask(
            patches,
            grid_h=7,
            grid_w=64,
            mask_ratio=MASK_RATIO,
        )