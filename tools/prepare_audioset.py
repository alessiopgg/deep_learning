"""
Preparazione di AudioSet WebDataset per MAE-AST V2.

Lo script legge gli shard ``.tar`` del dataset Hugging Face e costruisce
un manifest V2 portabile preservando i metadata originali disponibili:

    - source_id;
    - label testuali;
    - label_id numeriche;
    - shard e key originali.

Può inoltre estrarre i WAV mancanti senza riestrarre quelli già presenti.
Questo permette, ad esempio, di:

    - ricostruire un manifest arricchito per il train già estratto;
    - estrarre soltanto il piccolo split test ufficiale.

I file audio vengono nominati come nella precedente pipeline del progetto:

    shard-00000_sample-000000000.wav
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json_member(
        tar: tarfile.TarFile,
        member: tarfile.TarInfo,
) -> dict[str, Any]:
    """Legge un metadata JSON direttamente dallo shard."""

    file_obj = tar.extractfile(member)

    if file_obj is None:
        raise RuntimeError(
            f"Impossibile leggere il membro {member.name}."
        )

    return json.loads(
        file_obj.read().decode("utf-8")
    )


def _portable_wav_path(
        wav_path: Path,
        dataset_root: Path | None,
) -> str:
    """Restituisce un path portabile da salvare nel manifest."""

    if dataset_root is None:
        return str(wav_path.resolve())

    try:
        return wav_path.resolve().relative_to(
            dataset_root.resolve()
        ).as_posix()
    except ValueError as error:
        raise ValueError(
            "audio_dir deve essere contenuta in dataset_root "
            "quando si richiedono path relativi."
        ) from error


def prepare_audioset(
        tar_dir: str | Path,
        audio_dir: str | Path,
        output_manifest: str | Path,
        dataset_root: str | Path | None = None,
        extract_missing: bool = False,
        overwrite_audio: bool = False,
        max_shards: int | None = None,
) -> dict[str, Any]:
    """
    Costruisce un manifest MAE-AST a partire dagli shard AudioSet.

    Args:
        tar_dir: directory contenente gli shard ``.tar``.
        audio_dir: directory dei WAV già estratti o da estrarre.
        output_manifest: JSON di output.
        dataset_root: root rispetto alla quale salvare i path relativi.
        extract_missing: estrae i WAV mancanti dagli shard.
        overwrite_audio: sovrascrive WAV già presenti.
        max_shards: limita il numero di shard, utile per smoke test.
    """

    tar_dir = Path(tar_dir)
    audio_dir = Path(audio_dir)
    output_manifest = Path(output_manifest)
    dataset_root_path = (
        Path(dataset_root)
        if dataset_root is not None
        else None
    )

    if not tar_dir.is_dir():
        raise FileNotFoundError(
            f"Directory shard non trovata: {tar_dir}"
        )

    tar_paths = sorted(tar_dir.glob("*.tar"))

    if max_shards is not None:
        tar_paths = tar_paths[:max_shards]

    if not tar_paths:
        raise RuntimeError(
            f"Nessuno shard .tar trovato in {tar_dir}."
        )

    audio_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    source_id_counts: Counter[str] = Counter()

    for shard_index, tar_path in enumerate(
            tar_paths,
            start=1,
    ):
        shard_name = tar_path.stem

        print(
            f"[{shard_index}/{len(tar_paths)}] "
            f"{tar_path.name}"
        )

        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()

            json_members = {
                Path(member.name).stem: member
                for member in members
                if member.isfile()
                and member.name.lower().endswith(".json")
            }

            wav_members = {
                Path(member.name).stem: member
                for member in members
                if member.isfile()
                and member.name.lower().endswith(".wav")
            }

            keys = sorted(
                set(json_members)
                & set(wav_members)
            )

            missing_pairs = (
                set(json_members)
                ^ set(wav_members)
            )

            if missing_pairs:
                print(
                    f"  [WARN] {len(missing_pairs)} key senza coppia "
                    "WAV/JSON completa."
                )

            for key in keys:
                metadata = _load_json_member(
                    tar,
                    json_members[key],
                )

                output_wav = (
                    audio_dir
                    / f"{shard_name}_{key}.wav"
                )

                if (
                        overwrite_audio
                        or not output_wav.exists()
                ):
                    if extract_missing:
                        source = tar.extractfile(
                            wav_members[key]
                        )

                        if source is None:
                            status_counts[
                                "extract_error"
                            ] += 1
                            continue

                        with output_wav.open("wb") as target:
                            shutil.copyfileobj(
                                source,
                                target,
                            )

                        status_counts["extracted"] += 1
                    else:
                        status_counts[
                            "audio_missing"
                        ] += 1
                        continue
                else:
                    status_counts[
                        "already_exists"
                    ] += 1

                source_id = metadata.get("id")

                if source_id is not None:
                    source_id_counts[
                        str(source_id)
                    ] += 1

                labels = metadata.get(
                    "label",
                    [],
                )
                label_ids = metadata.get(
                    "label_id",
                    [],
                )

                if not isinstance(labels, list):
                    labels = [labels]

                if not isinstance(label_ids, list):
                    label_ids = [label_ids]

                data.append(
                    {
                        "wav": _portable_wav_path(
                            output_wav,
                            dataset_root_path,
                        ),
                        "source_id": source_id,
                        "labels": labels,
                        "label_ids": label_ids,
                        "shard": shard_name,
                        "key": key,
                    }
                )

    data.sort(
        key=lambda item: (
            str(item.get("shard", "")),
            str(item.get("key", "")),
        )
    )

    manifest = {
        "data": data,
        "metadata": {
            "source": "confit/audioset-16khz-wds",
            "num_shards": len(tar_paths),
            "num_entries": len(data),
        },
    }

    output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_manifest.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    duplicate_source_ids = sum(
        1
        for count in source_id_counts.values()
        if count > 1
    )

    summary = {
        "num_shards": len(tar_paths),
        "num_entries": len(data),
        "status_counts": dict(status_counts),
        "num_unique_source_ids": len(
            source_id_counts
        ),
        "num_source_ids_with_duplicates": (
            duplicate_source_ids
        ),
        "manifest": str(output_manifest),
        "audio_dir": str(audio_dir),
    }

    summary_path = output_manifest.with_suffix(
        ".summary.json"
    )

    with summary_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nPreparazione completata.")
    print(
        f"  Entry manifest: {len(data)}"
    )
    print(
        f"  Source ID unici: {len(source_id_counts)}"
    )
    print(
        "  Source ID duplicati: "
        f"{duplicate_source_ids}"
    )
    print(
        f"  Manifest: {output_manifest}"
    )
    print(
        f"  Summary: {summary_path}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce manifest AudioSet V2 dagli shard "
            "WebDataset e, opzionalmente, estrae i WAV mancanti."
        )
    )

    parser.add_argument(
        "--tar-dir",
        required=True,
        help="Directory contenente gli shard .tar.",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory dei WAV estratti.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Manifest JSON di output.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Root rispetto alla quale salvare path relativi. "
            "Se omessa vengono salvati path assoluti."
        ),
    )
    parser.add_argument(
        "--extract-missing",
        action="store_true",
        help="Estrae dagli shard i WAV mancanti.",
    )
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Sovrascrive WAV già presenti.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Numero massimo di shard, utile per smoke test.",
    )

    args = parser.parse_args()

    prepare_audioset(
        tar_dir=args.tar_dir,
        audio_dir=args.audio_dir,
        output_manifest=args.output,
        dataset_root=args.dataset_root,
        extract_missing=args.extract_missing,
        overwrite_audio=args.overwrite_audio,
        max_shards=args.max_shards,
    )


if __name__ == "__main__":
    main()
