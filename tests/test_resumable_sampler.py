from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from mae_ast.data.dataset import ResumableRandomSampler


class _IndexDataset(Dataset):
    def __init__(self, size: int):
        self.size = size
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        self.accessed.append(int(index))
        return int(index)


def test_resumable_sampler_recreates_exact_suffix():
    dataset = _IndexDataset(20)
    sampler = ResumableRandomSampler(dataset)

    sampler.set_epoch(seed=1234, start_index=0)
    full_order = list(iter(sampler))

    sampler.set_epoch(seed=1234, start_index=6)
    resumed_order = list(iter(sampler))

    assert resumed_order == full_order[6:]


def test_resumable_sampler_does_not_load_skipped_samples():
    dataset = _IndexDataset(20)
    sampler = ResumableRandomSampler(dataset)

    sampler.set_epoch(seed=99, start_index=0)
    full_order = list(iter(sampler))

    # Tre batch da due campioni sono gia' stati completati.
    sampler.set_epoch(seed=99, start_index=6)

    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    first_resumed_batch = next(iter(loader)).tolist()

    assert first_resumed_batch == full_order[6:8]
    assert dataset.accessed == full_order[6:8]
