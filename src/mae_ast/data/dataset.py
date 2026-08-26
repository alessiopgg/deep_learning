"""
Dataset e DataLoader per MAE-AST.

Il dataset restituisce:

Pretraining:
    {
        "spectrogram": Tensor(T, F)
    }

Fine-tuning:
    {
        "spectrogram": Tensor(T, F),
        "label": Tensor()
    }

Patchify e masking NON vengono eseguiti nel Dataset.
Vengono applicati successivamente dal modello.
"""

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

from mae_ast.data.audio import AudioPreprocessor


def load_stats(
        stats_path: str | Path | None,
) -> tuple[float | None, float | None]:
    """Carica mean e std globali del dataset."""

    if stats_path is None:
        return None, None

    stats_path = Path(stats_path)

    if not stats_path.exists():
        raise FileNotFoundError(
            f"File statistiche non trovato: {stats_path}"
        )

    with stats_path.open(
            "r",
            encoding="utf-8",
    ) as file:
        stats = json.load(file)

    if "mean" not in stats or "std" not in stats:
        raise ValueError(
            "Il file delle statistiche deve contenere "
            "i campi 'mean' e 'std'."
        )

    return (
        float(stats["mean"]),
        float(stats["std"]),
    )


class ResumableRandomSampler(Sampler[int]):
    """
    Sampler casuale deterministico con offset di ripresa.

    A ogni epoca genera una permutazione completa usando un generatore
    locale dedicato, quindi non consuma il RNG globale del modello.
    In caso di resume puo' iniziare direttamente da ``start_index`` senza
    chiedere al Dataset di caricare i campioni gia' processati.
    """

    def __init__(
            self,
            data_source: Dataset,
            seed: int = 0,
    ):
        self.data_source = data_source
        self.seed = int(seed)
        self.start_index = 0

    def set_epoch(
            self,
            seed: int,
            start_index: int = 0,
    ) -> None:
        start_index = int(start_index)

        if start_index < 0 or start_index > len(self.data_source):
            raise ValueError(
                "start_index fuori intervallo per il sampler: "
                f"{start_index} (dataset={len(self.data_source)})."
            )

        self.seed = int(seed)
        self.start_index = start_index

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        permutation = torch.randperm(
            len(self.data_source),
            generator=generator,
        ).tolist()

        yield from permutation[self.start_index:]

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


class AudioDataset(Dataset):
    """
    Dataset PyTorch per pretraining e fine-tuning MAE-AST.
    """

    def __init__(
            self,
            manifest_path: str | Path,
            audio_cfg: DictConfig,
            mode: str = "pretrain",
            dataset_root: str | Path | None = None,
            stats_path: str | Path | None = None,
            strict: bool = True,
            augment: bool = False,
    ):
        self.manifest_path = Path(
            manifest_path
        )

        self.mode = mode
        self.strict = strict
        self.augment = bool(augment)

        if self.augment and self.mode != "finetune":
            raise ValueError(
                "Le augmentation downstream possono essere abilitate "
                "solo con mode='finetune'."
            )

        if dataset_root is not None:
            self.dataset_root = Path(
                dataset_root
            )
        else:
            self.dataset_root = None

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "Manifest non trovato: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
                "r",
                encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if (
                "data" not in manifest
                or not isinstance(
            manifest["data"],
            list,
        )
        ):
            raise ValueError(
                "Formato manifest non valido. "
                "È atteso {'data': [...]}."
            )

        self.entries: list[
            dict[str, Any]
        ] = manifest["data"]

        norm_mean, norm_std = load_stats(
            stats_path
        )

        self.preprocessor = AudioPreprocessor(
            cfg=audio_cfg,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_audio_path(
            self,
            wav_path: str,
    ) -> Path:
        """
        Risolve il path dell'audio.

        I path assoluti vengono utilizzati direttamente.

        I path relativi vengono interpretati rispetto
        a dataset_root.
        """

        path = Path(wav_path)

        if path.is_absolute():
            return path

        if self.dataset_root is None:
            return path

        return (
                self.dataset_root
                / path
        )

    def __getitem__(
            self,
            index: int,
    ) -> dict[str, torch.Tensor]:

        entry = self.entries[index]

        if "wav" not in entry:
            raise ValueError(
                f"Entry {index} priva del campo 'wav'."
            )

        wav_path = self._resolve_audio_path(
            entry["wav"]
        )

        try:
            spectrogram = self.preprocessor.process(
                wav_path=wav_path,
                mode=self.mode,
                augment=self.augment,
            )

        except Exception:
            if self.strict:
                raise

            # Modalità permissiva utilizzabile solo quando esplicitamente
            # richiesta. Per gli esperimenti finali useremo strict=True.
            time_frames, n_mels = (
                self.preprocessor.expected_shape(
                    self.mode
                )
            )

            spectrogram = torch.zeros(
                time_frames,
                n_mels,
                dtype=torch.float32,
            )

        sample = {
            "spectrogram": spectrogram,
        }

        if self.mode == "finetune":

            if "labels" not in entry:
                raise ValueError(
                    "Nel fine-tuning ogni entry deve "
                    "contenere il campo 'labels'."
                )

            sample["label"] = torch.tensor(
                int(entry["labels"]),
                dtype=torch.long,
            )

        return sample


def build_dataloader(
        manifest_path: str | Path,
        audio_cfg: DictConfig,
        mode: str,
        batch_size: int,
        dataset_root: str | Path | None = None,
        stats_path: str | Path | None = None,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        strict: bool = True,
        augment: bool = False,
        generator: torch.Generator | None = None,
        resumable_shuffle: bool = False,
) -> DataLoader:
    """
    Costruisce un DataLoader MAE-AST.
    """

    dataset = AudioDataset(
        manifest_path=manifest_path,
        audio_cfg=audio_cfg,
        mode=mode,
        dataset_root=dataset_root,
        stats_path=stats_path,
        strict=strict,
        augment=augment,
    )

    sampler = None

    if resumable_shuffle:
        if not shuffle:
            raise ValueError(
                "resumable_shuffle=True richiede shuffle=True."
            )

        sampler = ResumableRandomSampler(
            dataset,
            seed=0,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )
