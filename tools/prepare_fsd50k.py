"""
Preparazione di FSD50K per il pretraining MAE-AST.

Il tool genera:

    datafiles/fsd50k_train.json
    datafiles/fsd50k_eval.json

Nei manifest vengono salvati path RELATIVI rispetto alla root
del dataset, in modo da poter utilizzare gli stessi JSON
su macchine differenti.

Esempio:

    {
        "wav": "FSD50K.dev_audio/64760.wav",
        "labels": "dummy"
    }

In questo modo sarà sufficiente modificare dataset_root
nella configurazione OmegaConf.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_ground_truth_csv(
        csv_path: Path,
) -> list[str]:
    """
    Legge gli identificativi audio dal ground truth FSD50K.
    """

    fnames: list[str] = []

    with csv_path.open(
            "r",
            encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            fnames.append(
                row["fname"]
            )

    return fnames


def collect_audio_files(
        fnames: list[str],
        dataset_root: Path,
        audio_subdir: str,
) -> tuple[list[str], list[str]]:
    """
    Verifica l'esistenza degli audio.

    Returns:
        path relativi dei file trovati;
        path relativi dei file mancanti.
    """

    found: list[str] = []
    missing: list[str] = []

    for fname in fnames:

        relative_path = (
                Path(audio_subdir)
                / f"{fname}.wav"
        )

        absolute_path = (
                dataset_root
                / relative_path
        )

        if absolute_path.is_file():
            found.append(
                relative_path.as_posix()
            )
        else:
            missing.append(
                relative_path.as_posix()
            )

    return found, missing


def build_manifest(
        wav_paths: list[str],
) -> dict:
    """
    Costruisce il manifest utilizzato dal Dataset MAE-AST.
    """

    return {
        "data": [
            {
                "wav": wav_path,
                "labels": "dummy",
            }
            for wav_path in sorted(
                wav_paths
            )
        ]
    }


def save_json(
        data: dict,
        output_path: Path,
) -> None:
    """Salva un file JSON."""

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


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Preparazione FSD50K per MAE-AST"
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root del dataset FSD50K.",
    )

    parser.add_argument(
        "--output-dir",
        default="datafiles",
        help="Directory in cui salvare i manifest.",
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset_root
    ).resolve()

    output_dir = Path(
        args.output_dir
    )

    dev_audio_dir = (
            dataset_root
            / "FSD50K.dev_audio"
    )

    eval_audio_dir = (
            dataset_root
            / "FSD50K.eval_audio"
    )

    ground_truth_dir = (
            dataset_root
            / "FSD50K.ground_truth"
    )

    dev_csv = (
            ground_truth_dir
            / "dev.csv"
    )

    eval_csv = (
            ground_truth_dir
            / "eval.csv"
    )

    required_paths = [
        dev_audio_dir,
        eval_audio_dir,
        dev_csv,
        eval_csv,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        print(
            "\nERRORE: mancano alcuni file "
            "o directory FSD50K:"
        )

        for path in missing_paths:
            print(f"  {path}")

        raise SystemExit(1)

    print("=" * 60)
    print("PREPARAZIONE FSD50K")
    print("=" * 60)

    dev_fnames = read_ground_truth_csv(
        dev_csv
    )

    eval_fnames = read_ground_truth_csv(
        eval_csv
    )

    dev_found, dev_missing = (
        collect_audio_files(
            fnames=dev_fnames,
            dataset_root=dataset_root,
            audio_subdir="FSD50K.dev_audio",
        )
    )

    eval_found, eval_missing = (
        collect_audio_files(
            fnames=eval_fnames,
            dataset_root=dataset_root,
            audio_subdir="FSD50K.eval_audio",
        )
    )

    print(
        f"Dev  - trovati: {len(dev_found)}, "
        f"mancanti: {len(dev_missing)}"
    )

    print(
        f"Eval - trovati: {len(eval_found)}, "
        f"mancanti: {len(eval_missing)}"
    )

    train_manifest = build_manifest(
        dev_found
    )

    eval_manifest = build_manifest(
        eval_found
    )

    save_json(
        train_manifest,
        output_dir
        / "fsd50k_train.json",
        )

    save_json(
        eval_manifest,
        output_dir
        / "fsd50k_eval.json",
        )

    print("=" * 60)
    print("RIEPILOGO")
    print("=" * 60)

    print(
        f"Train clip: {len(dev_found)}"
    )

    print(
        f"Eval clip:  {len(eval_found)}"
    )

    print(
        f"File mancanti: "
        f"{len(dev_missing) + len(eval_missing)}"
    )

    print(
        "\nNei manifest sono stati salvati "
        "path relativi."
    )

    print(
        f"Dataset root corrente: {dataset_root}"
    )


if __name__ == "__main__":
    main()