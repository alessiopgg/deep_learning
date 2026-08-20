"""
Preparazione di ESC-50 per il fine-tuning MAE-AST.

ESC-50 contiene 5 fold ufficiali.

Per ogni fold k vengono generati:

    esc50_train_foldk.json
    esc50_eval_foldk.json

Il fold k viene utilizzato come validation/test,
mentre gli altri quattro fold vengono utilizzati
per il training.

I path audio salvati nei manifest sono relativi
alla root di ESC-50.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_metadata(
        csv_path: Path,
) -> list[dict]:
    """
    Legge il file meta/esc50.csv.
    """

    entries: list[dict] = []

    with csv_path.open(
            "r",
            encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            entries.append(
                {
                    "filename":
                        row["filename"],

                    "fold":
                        int(row["fold"]),

                    "target":
                        int(row["target"]),

                    "category":
                        row["category"],
                }
            )

    return entries


def validate_audio_files(
        entries: list[dict],
        dataset_root: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Controlla che tutti gli audio indicati nei metadata esistano.
    """

    found: list[dict] = []
    missing: list[dict] = []

    for entry in entries:

        relative_path = (
                Path("audio")
                / entry["filename"]
        )

        absolute_path = (
                dataset_root
                / relative_path
        )

        if absolute_path.is_file():

            enriched = dict(entry)

            enriched["wav"] = (
                relative_path.as_posix()
            )

            found.append(enriched)

        else:

            missing.append(entry)

    return found, missing


def build_fold_manifests(
        entries: list[dict],
        num_folds: int = 5,
) -> dict[str, dict]:
    """
    Costruisce train/eval manifest per ogni fold.
    """

    manifests: dict[
        str,
        dict,
    ] = {}

    entries = sorted(
        entries,
        key=lambda item: item["filename"],
    )

    for fold in range(
            1,
            num_folds + 1,
    ):

        train_data = []
        eval_data = []

        for entry in entries:

            sample = {
                "wav":
                    entry["wav"],

                "labels":
                    str(
                        entry["target"]
                    ),
            }

            if entry["fold"] == fold:
                eval_data.append(sample)
            else:
                train_data.append(sample)

        manifests[
            f"train_fold{fold}"
        ] = {
            "data": train_data
        }

        manifests[
            f"eval_fold{fold}"
        ] = {
            "data": eval_data
        }

    return manifests


def save_json(
        data: dict,
        output_path: Path,
) -> None:
    """Salva un manifest JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
            "w",
            encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Salvato: {output_path} "
        f"({len(data['data'])} clip)"
    )


def print_statistics(
        entries: list[dict],
) -> None:
    """
    Mostra alcune statistiche di controllo del dataset.
    """

    fold_counts = defaultdict(int)
    class_counts = defaultdict(int)

    for entry in entries:

        fold_counts[
            entry["fold"]
        ] += 1

        class_counts[
            entry["category"]
        ] += 1

    print("\nDistribuzione per fold:")

    for fold in sorted(
            fold_counts
    ):
        print(
            f"  Fold {fold}: "
            f"{fold_counts[fold]} clip"
        )

    counts = list(
        class_counts.values()
    )

    print(
        f"\nNumero classi: "
        f"{len(class_counts)}"
    )

    print(
        "Clip per classe: "
        f"min={min(counts)}, "
        f"max={max(counts)}, "
        f"media={sum(counts) / len(counts):.1f}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Preparazione ESC-50 per MAE-AST"
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root del dataset ESC-50.",
    )

    parser.add_argument(
        "--output-dir",
        default="datafiles",
        help="Directory output dei manifest.",
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset_root
    ).resolve()

    output_dir = Path(
        args.output_dir
    )

    audio_dir = (
            dataset_root
            / "audio"
    )

    metadata_csv = (
            dataset_root
            / "meta"
            / "esc50.csv"
    )

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Directory audio non trovata: "
            f"{audio_dir}"
        )

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata ESC-50 non trovati: "
            f"{metadata_csv}"
        )

    print("=" * 60)
    print("PREPARAZIONE ESC-50")
    print("=" * 60)

    entries = read_metadata(
        metadata_csv
    )

    found, missing = (
        validate_audio_files(
            entries,
            dataset_root,
        )
    )

    print(
        f"Audio trovati: {len(found)}"
    )

    print(
        f"Audio mancanti: {len(missing)}"
    )

    print_statistics(found)

    manifests = (
        build_fold_manifests(
            found
        )
    )

    print("\nGenerazione manifest:")

    for name, manifest in (
            manifests.items()
    ):

        save_json(
            manifest,
            output_dir
            / f"esc50_{name}.json",
            )

    print("=" * 60)
    print("RIEPILOGO")
    print("=" * 60)

    print(
        f"Clip totali valide: "
        f"{len(found)}"
    )

    print(
        "Manifest generati: 10"
    )

    print(
        "5 train + 5 eval"
    )

    print(
        f"Dataset root corrente: "
        f"{dataset_root}"
    )


if __name__ == "__main__":
    main()