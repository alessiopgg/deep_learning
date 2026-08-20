"""
Confronto train/validation/test per AudioSet MAE-AST V2.

Lo script riusa gli indici CSV prodotti da ``explore_dataset.py`` per evitare
una nuova lettura di centinaia di migliaia di WAV. Train e validation sono
sottoinsiemi del train ufficiale AudioSet e vengono ricostruiti tramite
``source_id``. Il test usa il proprio indice EDA.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Carica un manifest nel formato MAE-AST V2."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "data" not in data or not isinstance(data["data"], list):
        raise ValueError(f"Manifest non valido: {path}")

    return data["data"]


def load_index(path: str | Path) -> dict[str, dict[str, str]]:
    """Indicizza dataset_index.csv per source_id."""

    path = Path(path)
    result: dict[str, dict[str, str]] = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                continue
            if source_id in result:
                raise ValueError(
                    f"source_id duplicato nell'indice {path}: {source_id}"
                )
            result[source_id] = row

    return result


def labels_of(entry: dict[str, Any]) -> list[str]:
    """Normalizza le label multi-label."""

    labels = entry.get("labels", [])
    if isinstance(labels, list):
        return [str(label) for label in labels]
    if labels is None:
        return []
    return [part.strip() for part in str(labels).split(",") if part.strip()]


def numeric(row: dict[str, str], key: str) -> float:
    """Legge un campo numerico dall'indice CSV."""

    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def summarize_split(
        entries: list[dict[str, Any]],
        index: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Riassume geometria e label di uno split."""

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    label_counts: Counter[str] = Counter()

    for entry in entries:
        source_id = str(entry.get("source_id", ""))
        row = index.get(source_id)
        if row is None:
            missing.append(source_id)
            continue
        rows.append(row)
        label_counts.update(labels_of(entry))

    if missing:
        raise ValueError(
            f"Mancano {len(missing)} source_id nell'indice. "
            f"Esempi: {missing[:5]}"
        )

    durations = np.asarray([numeric(row, "duration_sec") for row in rows])
    padding = np.asarray([numeric(row, "padding_ratio") for row in rows])

    full_real = sum(int(float(row.get("patches_full_real", 0) or 0)) for row in rows)
    partial = sum(int(float(row.get("patches_partial_padding", 0) or 0)) for row in rows)
    full_pad = sum(int(float(row.get("patches_full_padding", 0) or 0)) for row in rows)
    total_patches = full_real + partial + full_pad

    duration_thresholds = {}
    for threshold in (0.25, 1.0, 5.0, 9.0, 9.5, 9.9):
        count = int((durations < threshold).sum())
        duration_thresholds[str(threshold)] = {
            "count": count,
            "fraction": count / len(rows),
        }

    return {
        "num_entries": len(entries),
        "duration_sec": {
            "min": float(durations.min()),
            "mean": float(durations.mean()),
            "median": float(np.median(durations)),
            "max": float(durations.max()),
            "p05": float(np.percentile(durations, 5)),
            "p95": float(np.percentile(durations, 95)),
        },
        "duration_thresholds": duration_thresholds,
        "padding_ratio": {
            "mean": float(padding.mean()),
            "median": float(np.median(padding)),
            "p95": float(np.percentile(padding, 95)),
        },
        "patches": {
            "full_real_fraction": full_real / total_patches,
            "partial_padding_fraction": partial / total_patches,
            "full_padding_fraction": full_pad / total_patches,
        },
        "num_unique_labels": len(label_counts),
        "label_counts": dict(label_counts),
        "top_labels": label_counts.most_common(20),
    }


def prevalence_deltas(
        reference: dict[str, Any],
        other: dict[str, Any],
        top_k: int = 20,
) -> dict[str, Any]:
    """Confronta le prevalenze multi-label tra due split."""

    ref_n = reference["num_entries"]
    other_n = other["num_entries"]
    ref_counts = reference["label_counts"]
    other_counts = other["label_counts"]

    labels = sorted(set(ref_counts) | set(other_counts))
    rows = []

    for label in labels:
        ref_prev = ref_counts.get(label, 0) / ref_n
        other_prev = other_counts.get(label, 0) / other_n
        rows.append(
            {
                "label": label,
                "reference_prevalence": ref_prev,
                "other_prevalence": other_prev,
                "delta": other_prev - ref_prev,
                "absolute_delta": abs(other_prev - ref_prev),
            }
        )

    rows.sort(key=lambda item: item["absolute_delta"], reverse=True)

    return {
        "max_absolute_delta": rows[0]["absolute_delta"] if rows else 0.0,
        "mean_absolute_delta": mean(item["absolute_delta"] for item in rows) if rows else 0.0,
        "largest_differences": rows[:top_k],
    }


def require_matplotlib():
    """Importa matplotlib solo se vengono richiesti grafici."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            'Installare l\'extra analysis: pip install -e ".[analysis]"'
        ) from error
    return plt


def plot_comparison(
        summaries: dict[str, dict[str, Any]],
        output_dir: Path,
) -> None:
    """Crea pochi grafici sintetici di confronto tra split."""

    plt = require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(summaries)

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [summaries[name]["duration_sec"]["mean"] for name in names]
    medians = [summaries[name]["duration_sec"]["median"] for name in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, means, width, label="media")
    ax.bar(x + width / 2, medians, width, label="mediana")
    ax.set_xticks(x, names)
    ax.set_ylabel("Durata [s]")
    ax.set_title("Durata media/mediana per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "split_duration_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    values = [summaries[name]["padding_ratio"]["mean"] for name in names]
    ax.bar(names, values)
    ax.set_ylabel("Padding / target frames")
    ax.set_title("Padding medio per split")
    fig.tight_layout()
    fig.savefig(output_dir / "split_padding_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronta train/validation/test AudioSet senza rileggere i WAV."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument(
        "--official-train-index",
        required=True,
        help="dataset_index.csv dell'EDA sul train ufficiale prima dello split.",
    )
    parser.add_argument(
        "--test-index",
        required=True,
        help="dataset_index.csv dell'EDA sul test ufficiale.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    train_entries = load_manifest(args.train_manifest)
    val_entries = load_manifest(args.val_manifest)
    test_entries = load_manifest(args.test_manifest)

    official_train_index = load_index(args.official_train_index)
    test_index = load_index(args.test_index)

    summaries = {
        "train": summarize_split(train_entries, official_train_index),
        "validation": summarize_split(val_entries, official_train_index),
        "test": summarize_split(test_entries, test_index),
    }

    report = {
        "splits": summaries,
        "label_prevalence": {
            "validation_vs_train": prevalence_deltas(
                summaries["train"], summaries["validation"]
            ),
            "test_vs_train": prevalence_deltas(
                summaries["train"], summaries["test"]
            ),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "split_comparison.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    if not args.no_plots:
        plot_comparison(summaries, output_dir / "plots")

    print("\nConfronto split completato.")
    for name, summary in summaries.items():
        print(
            f"  {name:10s}: n={summary['num_entries']}, "
            f"durata media={summary['duration_sec']['mean']:.4f}s, "
            f"padding medio={summary['padding_ratio']['mean']:.3%}"
        )

    val_delta = report["label_prevalence"]["validation_vs_train"]["max_absolute_delta"]
    test_delta = report["label_prevalence"]["test_vs_train"]["max_absolute_delta"]
    print(f"  max delta label val/train:  {val_delta:.3%}")
    print(f"  max delta label test/train: {test_delta:.3%}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
