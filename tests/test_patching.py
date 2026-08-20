import torch

from mae_ast.models.patching import (
    compute_patch_grid,
    get_patch_metadata,
    patchify,
    unpatchify,
)


def test_pretrain_patch_grid():
    n_freq, n_time = compute_patch_grid(
        time_frames=1024,
        n_mels=128,
        patch_h=16,
        patch_w=16,
    )

    assert n_freq == 8
    assert n_time == 64


def test_pretrain_patch_shape():
    x = torch.randn(2, 1024, 128)

    patches = patchify(
        x,
        patch_h=16,
        patch_w=16,
    )

    assert patches.shape == (2, 512, 256)


def test_finetune_patch_shape():
    x = torch.randn(2, 512, 128)

    patches = patchify(
        x,
        patch_h=16,
        patch_w=16,
    )

    assert patches.shape == (2, 256, 256)


def test_patchify_unpatchify_roundtrip():
    x = torch.randn(2, 1024, 128)

    patches = patchify(x)

    reconstructed = unpatchify(
        patches,
        time_frames=1024,
        n_mels=128,
    )

    assert reconstructed.shape == x.shape
    assert torch.equal(reconstructed, x)


def test_patch_metadata():
    metadata = get_patch_metadata(
        time_frames=1024,
        n_mels=128,
        patch_h=16,
        patch_w=16,
    )

    assert metadata.n_freq == 8
    assert metadata.n_time == 64
    assert metadata.num_patches == 512
    assert metadata.patch_dim == 256