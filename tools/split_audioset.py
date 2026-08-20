"""
Creazione dello split train/validation interno per AudioSet 500k.

Il dataset Hugging Face fornisce già uno split test/eval ufficiale, che non
viene modificato. Questo script divide soltanto il train ufficiale in:

    - train per il pretraining;
    - validation interna per monitoraggio e scelta del checkpoint.

La selezione è deterministica e indipendente dall'ordine del manifest: ogni
``source_id`` riceve un hash SHA-256 stabile e le entry con hash più piccolo
formano la validation. In questo modo si ottiene esattamente la frazione
richiesta senza introdurre una dipendenza dall'ordinamento del JSON.

Prima di scrivere gli output vengono verificati:

    - presenza e unicità dei ``source_id`` nel train;
    - assenza di overlap train/test, se viene fornito il manifest test;
    - disgiunzione completa tra train e validation.

Le label AudioSet sono multi-label e NON vengono usate per costruire lo split
(self-supervised pretraining). Viene però prodotto un riepilogo della loro
distribuzione per controllare che la validation rimanga rappresentativa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_manifest(path: str | Path) -> dict[str, Any]:
    """Carica e valida il formato base di un manifest MAE-AST."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if "data" not in manifest or not isinstance(manifest["data"], list):
        raise ValueError(
            f"Formato manifest non valido in {path}. "
            "È atteso {'data': [...]} ."
        )

    return manifest


def _source_ids(entries: list[dict[str, Any]], name: str) -> list[str]:
    """Estrae i source_id e verifica che siano presenti e unici."""

    ids: list[str] = []
    missing = 0

    for entry in entries:
        source_id = entry.get("source_id")

        if source_id is None or str(source_id).strip() == "":
            missing += 1
            continue

        ids.append(str(source_id))

    if missing:
        raise ValueError(
            f"Il manifest {name} contiene {missing} entry senza source_id."
        )

    counts = Counter(ids)
    duplicates = [source_id for source_id, count in counts.items() if count > 1]

    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"Il manifest {name} contiene {len(duplicates)} source_id duplicati. "
            f"Esempi: {preview}"
        )

    return ids


def _stable_score(source_id: str, seed: int) -> str:
    """Restituisce un hash stabile usato per ordinare gli esempi."""

    payload = f"{seed}:{source_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _label_counts(entries: list[dict[str, Any]]) -> Counter[str]:
    """Conta le occorrenze multi-label in una collezione di entry."""

    counter: Counter[str] = Counter()

    for entry in entries:
        labels = entry.get("labels", [])

        if isinstance(labels, str):
            labels = [labels]

        for label in labels:
            counter[str(label)] += 1

    return counter


def _label_distribution_report(
        train_entries: list[dict[str, Any]],
        val_entries: list[dict[str, Any]],
        top_k: int = 20,
) -> dict[str, Any]:
    """Confronta le prevalenze label train/validation."""

    train_counts = _label_counts(train_entries)
    val_counts = _label_counts(val_entries)

    labels = sorted(set(train_counts) | set(val_counts))

    rows: list[dict[str, Any]] = []

    train_den = max(len(train_entries), 1)
    val_den = max(len(val_entries), 1)

    for label in labels:
        train_prev = train_counts[label] / train_den
        val_prev = val_counts[label] / val_den

        rows.append(
            {
                "label": label,
                "train_count": train_counts[label],
                "val_count": val_counts[label],
                "train_prevalence": train_prev,
                "val_prevalence": val_prev,
                "absolute_prevalence_delta": abs(val_prev - train_prev),
            }
        )

    rows.sort(
        key=lambda row: row["absolute_prevalence_delta"],
        reverse=True,
    )

    return {
        "num_unique_labels": len(labels),
        "max_absolute_prevalence_delta": (
            rows[0]["absolute_prevalence_delta"] if rows else 0.0
        ),
        "mean_absolute_prevalence_delta": (
            sum(row["absolute_prevalence_delta"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "largest_differences": rows[:top_k],
    }


def _write_manifest(
        path: Path,
        entries: list[dict[str, Any]],
        metadata: dict[str, Any],
) -> None:
    """Scrive un manifest preservando le entry originali."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "data": entries,
                "metadata": metadata,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def split_audioset_train_val(
        input_manifest: str | Path,
        train_output: str | Path,
        val_output: str | Path,
        val_ratio: float = 0.025,
        seed: int = 42,
        test_manifest: str | Path | None = None,
        summary_output: str | Path | None = None,
) -> dict[str, Any]:
    """
    Divide il train ufficiale AudioSet in train e validation interna.

    Lo split ha dimensione esatta ``round(N * val_ratio)`` ed è deterministico
    rispetto a ``source_id`` e ``seed``.
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio deve essere compreso tra 0 e 1.")

    input_manifest = Path(input_manifest)
    train_output = Path(train_output)
    val_output = Path(val_output)

    manifest = _load_manifest(input_manifest)
    entries: list[dict[str, Any]] = manifest["data"]

    if len(entries) < 2:
        raise ValueError("Servono almeno due entry per creare train/validation.")

    source_ids = _source_ids(entries, "train ufficiale")

    test_entries: list[dict[str, Any]] = []
    test_ids: set[str] = set()

    if test_manifest is not None:
        test_data = _load_manifest(test_manifest)
        test_entries = test_data["data"]
        test_ids = set(_source_ids(test_entries, "test ufficiale"))

        overlap = set(source_ids) & test_ids

        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                "Rilevato overlap tra train ufficiale e test ufficiale: "
                f"{len(overlap)} source_id. Esempi: {preview}"
            )

    num_val = round(len(entries) * val_ratio)
    num_val = max(1, min(num_val, len(entries) - 1))

    ranked_ids = sorted(
        source_ids,
        key=lambda source_id: _stable_score(source_id, seed),
    )
    val_ids = set(ranked_ids[:num_val])

    # Preserviamo l'ordine originale del manifest nei due output.
    train_entries = [
        entry
        for entry in entries
        if str(entry["source_id"]) not in val_ids
    ]
    val_entries = [
        entry
        for entry in entries
        if str(entry["source_id"]) in val_ids
    ]

    train_ids = {str(entry["source_id"]) for entry in train_entries}
    validation_ids = {str(entry["source_id"]) for entry in val_entries}

    if train_ids & validation_ids:
        raise RuntimeError("Errore interno: train e validation non sono disgiunti.")

    if len(train_entries) + len(val_entries) != len(entries):
        raise RuntimeError("Errore interno: alcune entry sono state perse nello split.")

    split_metadata = {
        "source": manifest.get("metadata", {}).get(
            "source",
            "confit/audioset-16khz-wds",
        ),
        "parent_manifest": str(input_manifest),
        "split_method": "sha256_source_id_rank",
        "split_seed": seed,
        "validation_ratio_requested": val_ratio,
        "validation_ratio_actual": len(val_entries) / len(entries),
        "official_test_untouched": test_manifest is not None,
    }

    _write_manifest(
        train_output,
        train_entries,
        {
            **split_metadata,
            "split": "train",
            "num_entries": len(train_entries),
        },
    )
    _write_manifest(
        val_output,
        val_entries,
        {
            **split_metadata,
            "split": "validation",
            "num_entries": len(val_entries),
        },
    )

    label_report = _label_distribution_report(
        train_entries,
        val_entries,
    )

    summary = {
        "input_manifest": str(input_manifest),
        "test_manifest": str(test_manifest) if test_manifest is not None else None,
        "split_method": "sha256_source_id_rank",
        "seed": seed,
        "validation_ratio_requested": val_ratio,
        "validation_ratio_actual": len(val_entries) / len(entries),
        "num_input_entries": len(entries),
        "num_train_entries": len(train_entries),
        "num_validation_entries": len(val_entries),
        "num_test_entries": len(test_entries),
        "num_unique_train_source_ids": len(train_ids),
        "num_unique_validation_source_ids": len(validation_ids),
        "num_unique_test_source_ids": len(test_ids),
        "train_validation_overlap": len(train_ids & validation_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "validation_test_overlap": len(validation_ids & test_ids),
        "label_distribution": label_report,
        "train_manifest": str(train_output),
        "validation_manifest": str(val_output),
    }

    if summary_output is None:
        summary_path = val_output.with_name("audioset_500k_split_summary.json")
    else:
        summary_path = Path(summary_output)

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nSplit AudioSet completato.")
    print(f"  Input ufficiale train: {len(entries)}")
    print(f"  Train pretraining:     {len(train_entries)}")
    print(f"  Validation interna:    {len(val_entries)}")
    print(f"  Test ufficiale:        {len(test_entries)}")
    print(
        "  Validation ratio:     "
        f"{100.0 * len(val_entries) / len(entries):.4f}%"
    )
    print(f"  Train/val overlap:     {len(train_ids & validation_ids)}")
    print(f"  Train/test overlap:    {len(train_ids & test_ids)}")
    print(f"  Val/test overlap:      {len(validation_ids & test_ids)}")
    print(
        "  Max delta prevalenza label: "
        f"{100.0 * label_report['max_absolute_prevalence_delta']:.3f} punti %"
    )
    print(f"  Train manifest: {train_output}")
    print(f"  Val manifest:   {val_output}")
    print(f"  Summary:        {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Divide il train ufficiale AudioSet in train e validation "
            "deterministici, lasciando intatto il test ufficiale."
        )
    )

    parser.add_argument(
        "--input-manifest",
        required=True,
        help="Manifest V2 del train ufficiale AudioSet.",
    )
    parser.add_argument(
        "--train-output",
        required=True,
        help="Manifest di output per il train di pretraining.",
    )
    parser.add_argument(
        "--val-output",
        required=True,
        help="Manifest di output per la validation interna.",
    )
    parser.add_argument(
        "--test-manifest",
        default=None,
        help=(
            "Manifest del test ufficiale. Se fornito viene verificata "
            "l'assenza di overlap tramite source_id."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.025,
        help="Frazione del train ufficiale assegnata alla validation (default 0.025).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed incorporato nell'hash deterministico (default 42).",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="JSON opzionale con il riepilogo dello split.",
    )

    args = parser.parse_args()

    split_audioset_train_val(
        input_manifest=args.input_manifest,
        train_output=args.train_output,
        val_output=args.val_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        test_manifest=args.test_manifest,
        summary_output=args.summary_output,
    )


if __name__ == "__main__":
    main()
