"""
Patching utilities for MAE-AST.

The input spectrogram convention is:

    (T, F)       for a single sample
    (B, T, F)    for a batch

where:
    T = time frames
    F = frequency bins

Patches have shape:

    patch_h = frequency dimension
    patch_w = temporal dimension

Tokens are ordered frequency-major, then time-major:

    token_index = freq_index * n_time + time_index
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PatchMetadata:
    """Metadata describing the patch grid."""

    time_frames: int
    n_mels: int

    patch_h: int
    patch_w: int

    n_freq: int
    n_time: int

    num_patches: int
    patch_dim: int


def compute_patch_grid(
        time_frames: int,
        n_mels: int,
        patch_h: int,
        patch_w: int,
) -> tuple[int, int]:
    """
    Compute the spectrogram patch grid.

    Returns:
        (n_freq, n_time)
    """

    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("Patch dimensions must be positive.")

    if n_mels % patch_h != 0:
        raise ValueError(
            f"n_mels={n_mels} is not divisible by patch_h={patch_h}"
        )

    if time_frames % patch_w != 0:
        raise ValueError(
            f"time_frames={time_frames} is not divisible by patch_w={patch_w}"
        )

    n_freq = n_mels // patch_h
    n_time = time_frames // patch_w

    return n_freq, n_time


def get_patch_metadata(
        time_frames: int,
        n_mels: int,
        patch_h: int = 16,
        patch_w: int = 16,
) -> PatchMetadata:
    """Return metadata for a given spectrogram and patch configuration."""

    n_freq, n_time = compute_patch_grid(
        time_frames=time_frames,
        n_mels=n_mels,
        patch_h=patch_h,
        patch_w=patch_w,
    )

    return PatchMetadata(
        time_frames=time_frames,
        n_mels=n_mels,
        patch_h=patch_h,
        patch_w=patch_w,
        n_freq=n_freq,
        n_time=n_time,
        num_patches=n_freq * n_time,
        patch_dim=patch_h * patch_w,
    )


def _as_batched_spectrogram(
        spectrogram: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """
    Convert input to (B, T, F).

    Returns:
        batched tensor
        True if original input was unbatched
    """

    if spectrogram.ndim == 2:
        return spectrogram.unsqueeze(0), True

    if spectrogram.ndim == 3:
        return spectrogram, False

    raise ValueError(
        "Expected spectrogram with shape "
        f"(T, F) or (B, T, F), found {tuple(spectrogram.shape)}"
    )


def patchify(
        spectrogram: torch.Tensor,
        patch_h: int = 16,
        patch_w: int = 16,
) -> torch.Tensor:
    """
    Convert a spectrogram into flattened non-overlapping patches.

    Input:
        (T, F)
        or
        (B, T, F)

    Output:
        (N, patch_dim)
        or
        (B, N, patch_dim)
    """

    x, unbatched = _as_batched_spectrogram(spectrogram)

    batch_size, time_frames, n_mels = x.shape

    n_freq, n_time = compute_patch_grid(
        time_frames=time_frames,
        n_mels=n_mels,
        patch_h=patch_h,
        patch_w=patch_w,
    )

    # (B, T, F) -> (B, F, T)
    x = x.transpose(1, 2)

    # Split frequency and time into patch dimensions.
    #
    # (B, F, T)
    # ->
    # (B, n_freq, patch_h, n_time, patch_w)
    x = x.reshape(
        batch_size,
        n_freq,
        patch_h,
        n_time,
        patch_w,
    )

    # Put patch-grid coordinates together.
    #
    # ->
    # (B, n_freq, n_time, patch_h, patch_w)
    x = x.permute(0, 1, 3, 2, 4).contiguous()

    # Flatten both:
    #   patch grid
    #   patch contents
    patches = x.reshape(
        batch_size,
        n_freq * n_time,
        patch_h * patch_w,
        )

    if unbatched:
        patches = patches.squeeze(0)

    return patches


def unpatchify(
        patches: torch.Tensor,
        time_frames: int,
        n_mels: int,
        patch_h: int = 16,
        patch_w: int = 16,
) -> torch.Tensor:
    """
    Reconstruct a spectrogram from flattened patches.

    Input:
        (N, patch_dim)
        or
        (B, N, patch_dim)

    Output:
        (T, F)
        or
        (B, T, F)
    """

    unbatched = patches.ndim == 2

    if unbatched:
        patches = patches.unsqueeze(0)

    elif patches.ndim != 3:
        raise ValueError(
            "Expected patches with shape "
            f"(N, D) or (B, N, D), found {tuple(patches.shape)}"
        )

    batch_size, num_patches, patch_dim = patches.shape

    n_freq, n_time = compute_patch_grid(
        time_frames=time_frames,
        n_mels=n_mels,
        patch_h=patch_h,
        patch_w=patch_w,
    )

    expected_num_patches = n_freq * n_time
    expected_patch_dim = patch_h * patch_w

    if num_patches != expected_num_patches:
        raise ValueError(
            f"Found {num_patches} patches, "
            f"expected {expected_num_patches}"
        )

    if patch_dim != expected_patch_dim:
        raise ValueError(
            f"Patch dimension is {patch_dim}, "
            f"expected {expected_patch_dim}"
        )

    x = patches.reshape(
        batch_size,
        n_freq,
        n_time,
        patch_h,
        patch_w,
    )

    x = x.permute(0, 1, 3, 2, 4).contiguous()

    # (B, n_freq, patch_h, n_time, patch_w)
    # -> (B, F, T)
    x = x.reshape(
        batch_size,
        n_mels,
        time_frames,
    )

    # (B, F, T) -> (B, T, F)
    spectrogram = x.transpose(1, 2).contiguous()

    if unbatched:
        spectrogram = spectrogram.squeeze(0)

    return spectrogram