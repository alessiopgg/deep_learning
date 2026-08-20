"""
EDA e audit del dataset per MAE-AST V2.

Lo script è pensato prima di tutto per AudioSet, ma funziona con qualunque
manifest nel formato ``{"data": [...]}`` usato dal progetto.

Analisi prodotte:

    - integrità dei file;
    - durata, sample rate e numero di canali;
    - stima esatta dei frame Kaldi fbank quando il sample rate è già target;
    - padding e troncamento rispetto alla lunghezza MAE-AST;
    - impatto del padding a livello di patch;
    - statistiche log-Mel su un campione deterministico, con e senza padding;
    - distribuzione delle label AudioSet;
    - controllo dei source_id duplicati;
    - grafici e alcuni esempi waveform/spettrogramma;
    - conteggio esatto degli outlier di durata;
    - analisi opzionale di silenzio/clipping su un campione deterministico;
    - dataset_index.csv e summary.json per analisi successive rapide.

Lo script NON modifica gli audio e NON crea gli split train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import wave
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
import soundfile as sf

from mae_ast.config import load_config
from mae_ast.data.audio import AudioPreprocessor


def load_manifest(
        manifest_path: str | Path,
) -> list[dict[str, Any]]:
    """Carica un manifest MAE-AST."""

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest non trovato: {manifest_path}"
        )

    with manifest_path.open(
            "r",
            encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if (
            "data" not in manifest
            or not isinstance(manifest["data"], list)
    ):
        raise ValueError(
            "Formato manifest non valido: è atteso {'data': [...]}."
        )

    return manifest["data"]


def resolve_audio_path(
        wav: str,
        dataset_root: str | Path | None,
) -> Path:
    """Risolve un path assoluto o relativo del manifest."""

    path = Path(wav)

    if path.is_absolute():
        return path

    if dataset_root is None:
        return path

    return Path(dataset_root) / path


def read_wav_metadata(
        wav_path: Path,
) -> dict[str, Any]:
    """
    Legge i metadata WAV senza decodificare l'intero segnale.

    In caso di WAV non gestibile dal modulo standard ``wave`` viene usato
    SoundFile come fallback, così da distinguere un file realmente corrotto
    da un file semplicemente non PCM-standard.
    """

    try:
        with wave.open(str(wav_path), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_rate = int(handle.getframerate())
            num_samples = int(handle.getnframes())

    except Exception:
        data, sample_rate = sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(
            data.T.copy()
        )
        channels = int(waveform.shape[0])
        num_samples = int(waveform.shape[-1])

    duration_sec = (
        num_samples / sample_rate
        if sample_rate > 0
        else 0.0
    )

    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "duration_sec": duration_sec,
    }


def estimate_kaldi_frames(
        num_samples: int,
        sample_rate: int,
        target_sample_rate: int,
        frame_length_ms: float,
        frame_shift_ms: float,
) -> int:
    """
    Stima i frame prodotti da Kaldi fbank con ``snip_edges=True``.

    Per AudioSet, già a 16 kHz, la formula coincide con la geometria reale
    usata da ``torchaudio.compliance.kaldi.fbank``. Se serve resampling,
    viene stimata prima la nuova lunghezza del segnale.
    """

    if sample_rate <= 0:
        return 0

    if sample_rate != target_sample_rate:
        num_samples = int(round(
            num_samples
            * target_sample_rate
            / sample_rate
        ))
        sample_rate = target_sample_rate

    frame_length = int(round(
        sample_rate
        * frame_length_ms
        / 1000.0
    ))
    frame_shift = int(round(
        sample_rate
        * frame_shift_ms
        / 1000.0
    ))

    if (
            num_samples < frame_length
            or frame_shift <= 0
    ):
        return 0

    return 1 + (
        num_samples - frame_length
    ) // frame_shift


def patch_padding_counts(
        real_frames: int,
        target_frames: int,
        n_mels: int,
        patch_h: int,
        patch_w: int,
) -> dict[str, int]:
    """Conta patch reali, parziali e completamente di padding."""

    if (
            target_frames % patch_w != 0
            or n_mels % patch_h != 0
    ):
        raise ValueError(
            "La geometria target deve essere divisibile per la patch."
        )

    clipped_real = min(
        max(real_frames, 0),
        target_frames,
    )

    time_patches = target_frames // patch_w
    freq_patches = n_mels // patch_h

    full_real_time = 0
    partial_time = 0
    full_padding_time = 0

    for time_index in range(time_patches):
        start = time_index * patch_w
        end = start + patch_w

        if end <= clipped_real:
            full_real_time += 1
        elif start < clipped_real < end:
            partial_time += 1
        else:
            full_padding_time += 1

    return {
        "patches_full_real": full_real_time * freq_patches,
        "patches_partial_padding": partial_time * freq_patches,
        "patches_full_padding": full_padding_time * freq_patches,
    }


def normalize_labels(
        labels: Any,
) -> list[str]:
    """Converte i diversi formati storici delle label in una lista."""

    if labels is None:
        return []

    if isinstance(labels, list):
        return [
            str(label).strip()
            for label in labels
            if str(label).strip()
        ]

    text = str(labels).strip()

    if not text:
        return []

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


DURATION_THRESHOLDS_SEC = (0.25, 1.0, 5.0, 9.0, 9.5, 9.9)


def summarize_duration_outliers(
        ok_rows: list[dict[str, Any]],
        shortest_k: int = 20,
) -> dict[str, Any]:
    """Riassume la coda corta della distribuzione delle durate."""

    if not ok_rows:
        return {}

    total = len(ok_rows)
    thresholds: dict[str, dict[str, float | int]] = {}

    for threshold in DURATION_THRESHOLDS_SEC:
        count = sum(
            1
            for row in ok_rows
            if float(row["duration_sec"]) < threshold
        )
        key = f"lt_{str(threshold).replace('.', '_')}_sec"
        thresholds[key] = {
            "threshold_sec": threshold,
            "count": count,
            "fraction": count / total,
        }

    shortest = sorted(
        ok_rows,
        key=lambda row: float(row["duration_sec"]),
    )[:shortest_k]

    return {
        "threshold_counts": thresholds,
        "shortest_examples": [
            {
                "source_id": row.get("source_id"),
                "wav": row.get("wav"),
                "duration_sec": float(row["duration_sec"]),
                "mel_frames_real": int(row["mel_frames_real"]),
                "padding_ratio": float(row["padding_ratio"]),
            }
            for row in shortest
        ],
    }


def _select_deterministic_rows(
        rows: list[dict[str, Any]],
        max_files: int | None,
        seed: int,
        include_shortest: int = 20,
) -> list[dict[str, Any]]:
    """
    Seleziona un campione riproducibile includendo sempre gli outlier corti.

    Questo evita che clip anomale di pochi millisecondi vengano perse da un
    campionamento puramente casuale.
    """

    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    if max_files is None or max_files < 0 or len(valid_rows) <= max_files:
        return valid_rows

    shortest = sorted(
        valid_rows,
        key=lambda row: float(row["duration_sec"]),
    )[:min(include_shortest, max_files)]

    selected_ids = {id(row) for row in shortest}
    remaining = [
        row for row in valid_rows
        if id(row) not in selected_ids
    ]

    rng = random.Random(seed)
    random_count = max_files - len(shortest)
    random_rows = rng.sample(remaining, random_count) if random_count > 0 else []

    return shortest + random_rows


def analyze_waveform_quality_sample(
        rows: list[dict[str, Any]],
        max_files: int | None,
        seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Analizza RMS, silenzio digitale e clipping su un campione deterministico.

    Il controllo richiede la decodifica completa del WAV e viene quindi tenuto
    separato dall'audit dei metadata. Con ``max_files < 0`` viene analizzato
    l'intero dataset.
    """

    selected = _select_deterministic_rows(
        rows,
        max_files=max_files,
        seed=seed,
    )

    quality_rows: list[dict[str, Any]] = []
    skipped = 0

    print(f"File per quality audit waveform: {len(selected)}")

    for index, row in enumerate(selected, start=1):
        wav_path = Path(row["resolved_path"])

        try:
            data, sample_rate = sf.read(
                str(wav_path),
                dtype="float32",
                always_2d=True,
            )

            flat = np.asarray(data, dtype=np.float32).reshape(-1)

            if flat.size == 0:
                raise ValueError("Waveform vuota")

            abs_values = np.abs(flat)
            rms = float(np.sqrt(np.mean(np.square(flat, dtype=np.float64))))
            peak_abs = float(abs_values.max())
            clip_fraction = float(np.mean(abs_values >= 0.999))
            near_zero_fraction = float(np.mean(abs_values <= 1.0e-6))

            quality_rows.append(
                {
                    "source_id": row.get("source_id"),
                    "wav": row.get("wav"),
                    "resolved_path": str(wav_path),
                    "sample_rate": int(sample_rate),
                    "duration_sec": float(row["duration_sec"]),
                    "rms": rms,
                    "peak_abs": peak_abs,
                    "clip_fraction_ge_0_999": clip_fraction,
                    "near_zero_fraction_le_1e_6": near_zero_fraction,
                }
            )

        except Exception as error:
            skipped += 1
            print(f"  [WARN quality] {wav_path}: {error}")

        if index % 1000 == 0 or index == len(selected):
            print(f"  Quality waveform {index}/{len(selected)}")

    if not quality_rows:
        return (
            {
                "num_files_requested": len(selected),
                "num_files_used": 0,
                "num_files_skipped": skipped,
            },
            [],
        )

    rms_values = np.asarray(
        [row["rms"] for row in quality_rows],
        dtype=np.float64,
    )
    peak_values = np.asarray(
        [row["peak_abs"] for row in quality_rows],
        dtype=np.float64,
    )
    clipping = np.asarray(
        [row["clip_fraction_ge_0_999"] for row in quality_rows],
        dtype=np.float64,
    )
    near_zero = np.asarray(
        [row["near_zero_fraction_le_1e_6"] for row in quality_rows],
        dtype=np.float64,
    )

    def count_fraction(mask: np.ndarray) -> dict[str, float | int]:
        count = int(mask.sum())
        return {
            "count": count,
            "fraction": count / len(quality_rows),
        }

    summary = {
        "num_files_requested": len(selected),
        "num_files_used": len(quality_rows),
        "num_files_skipped": skipped,
        "rms": {
            "min": float(rms_values.min()),
            "median": float(np.median(rms_values)),
            "mean": float(rms_values.mean()),
            "max": float(rms_values.max()),
            "p01": float(np.percentile(rms_values, 1)),
            "p99": float(np.percentile(rms_values, 99)),
        },
        "peak_abs": {
            "min": float(peak_values.min()),
            "median": float(np.median(peak_values)),
            "mean": float(peak_values.mean()),
            "max": float(peak_values.max()),
        },
        "near_silence": {
            "rms_le_1e_5": count_fraction(rms_values <= 1.0e-5),
            "rms_le_1e_4": count_fraction(rms_values <= 1.0e-4),
            "rms_le_1e_3": count_fraction(rms_values <= 1.0e-3),
            "near_zero_fraction_ge_99pct": count_fraction(near_zero >= 0.99),
        },
        "clipping": {
            "any_sample_ge_0_999": count_fraction(clipping > 0.0),
            "clip_fraction_ge_0_1pct": count_fraction(clipping >= 0.001),
            "clip_fraction_ge_1pct": count_fraction(clipping >= 0.01),
        },
    }

    quietest = sorted(quality_rows, key=lambda row: row["rms"])[:20]
    most_clipped = sorted(
        quality_rows,
        key=lambda row: row["clip_fraction_ge_0_999"],
        reverse=True,
    )[:20]

    summary["quietest_examples"] = quietest
    summary["most_clipped_examples"] = most_clipped

    return summary, quality_rows


def audit_entries(
        entries: list[dict[str, Any]],
        dataset_root: str | Path | None,
        cfg,
        mode: str,
        max_files: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scansiona i WAV e costruisce l'indice descrittivo."""

    if max_files is not None:
        entries = entries[:max_files]

    target_frames = (
        int(cfg.audio.pretrain_target_frames)
        if mode == "pretrain"
        else int(cfg.audio.finetune_target_frames)
    )

    target_sample_rate = int(
        cfg.audio.sample_rate
    )
    n_mels = int(cfg.audio.n_mels)
    patch_h = int(cfg.patching.patch_h)
    patch_w = int(cfg.patching.patch_w)

    rows: list[dict[str, Any]] = []
    sample_rate_counts: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    source_id_counts: Counter[str] = Counter()

    print(
        f"File da auditare: {len(entries)}"
    )

    for index, entry in enumerate(entries):
        wav_value = entry.get("wav")

        row: dict[str, Any] = {
            "entry_index": index,
            "wav": wav_value,
            "source_id": entry.get("source_id"),
            "status": "ok",
            "error": "",
        }

        labels = normalize_labels(
            entry.get("labels")
        )
        row["num_labels"] = len(labels)

        for label in labels:
            label_counts[label] += 1

        source_id = entry.get("source_id")
        if source_id:
            source_id_counts[
                str(source_id)
            ] += 1

        if not wav_value:
            row["status"] = "missing_wav_field"
            status_counts[row["status"]] += 1
            rows.append(row)
            continue

        wav_path = resolve_audio_path(
            str(wav_value),
            dataset_root,
        )
        row["resolved_path"] = str(wav_path)

        if not wav_path.is_file():
            row["status"] = "missing_file"
            status_counts[row["status"]] += 1
            rows.append(row)
            continue

        try:
            metadata = read_wav_metadata(
                wav_path
            )

            sample_rate = int(
                metadata["sample_rate"]
            )
            channels = int(
                metadata["channels"]
            )
            num_samples = int(
                metadata["num_samples"]
            )
            duration_sec = float(
                metadata["duration_sec"]
            )

            mel_frames_real = estimate_kaldi_frames(
                num_samples=num_samples,
                sample_rate=sample_rate,
                target_sample_rate=target_sample_rate,
                frame_length_ms=float(
                    cfg.audio.frame_length_ms
                ),
                frame_shift_ms=float(
                    cfg.audio.frame_shift_ms
                ),
            )

            padding_frames = max(
                target_frames - mel_frames_real,
                0,
            )
            truncated_frames = max(
                mel_frames_real - target_frames,
                0,
            )

            patch_counts = patch_padding_counts(
                real_frames=mel_frames_real,
                target_frames=target_frames,
                n_mels=n_mels,
                patch_h=patch_h,
                patch_w=patch_w,
            )

            row.update(
                {
                    "duration_sec": duration_sec,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "num_samples": num_samples,
                    "mel_frames_real": mel_frames_real,
                    "padding_frames": padding_frames,
                    "truncated_frames": truncated_frames,
                    "padding_ratio": (
                        padding_frames / target_frames
                    ),
                    "truncated_ratio_vs_target": (
                        truncated_frames / target_frames
                    ),
                    **patch_counts,
                }
            )

            sample_rate_counts[sample_rate] += 1
            channel_counts[channels] += 1
            status_counts["ok"] += 1

        except Exception as error:
            row["status"] = "corrupt_or_unreadable"
            row["error"] = str(error)
            status_counts[row["status"]] += 1

        rows.append(row)

        if (
                (index + 1) % 5000 == 0
                or index + 1 == len(entries)
        ):
            print(
                f"  Processati {index + 1}/{len(entries)}"
            )

    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    durations = [
        float(row["duration_sec"])
        for row in ok_rows
    ]
    padding_ratios = [
        float(row["padding_ratio"])
        for row in ok_rows
    ]
    truncation_ratios = [
        float(row["truncated_ratio_vs_target"])
        for row in ok_rows
    ]

    num_padding = sum(
        1
        for row in ok_rows
        if int(row["padding_frames"]) > 0
    )
    num_truncated = sum(
        1
        for row in ok_rows
        if int(row["truncated_frames"]) > 0
    )

    total_full_real_patches = sum(
        int(row["patches_full_real"])
        for row in ok_rows
    )
    total_partial_padding_patches = sum(
        int(row["patches_partial_padding"])
        for row in ok_rows
    )
    total_full_padding_patches = sum(
        int(row["patches_full_padding"])
        for row in ok_rows
    )
    total_patches = (
        total_full_real_patches
        + total_partial_padding_patches
        + total_full_padding_patches
    )
    padding_affected_patches = (
        total_partial_padding_patches
        + total_full_padding_patches
    )

    summary: dict[str, Any] = {
        "num_entries_scanned": len(rows),
        "status_counts": dict(status_counts),
        "sample_rate_counts": {
            str(key): value
            for key, value in sorted(
                sample_rate_counts.items()
            )
        },
        "channel_counts": {
            str(key): value
            for key, value in sorted(
                channel_counts.items()
            )
        },
        "num_unique_labels": len(label_counts),
        "top_labels": label_counts.most_common(30),
        "num_unique_source_ids": len(source_id_counts),
        "num_source_ids_with_duplicates": sum(
            1
            for count in source_id_counts.values()
            if count > 1
        ),
        "target_frames": target_frames,
        "duration_sec": {},
        "duration_outliers": {},
        "padding": {},
        "patch_padding": {},
    }

    if durations:
        summary["duration_sec"] = {
            "min": min(durations),
            "mean": mean(durations),
            "median": median(durations),
            "max": max(durations),
            "p05": float(np.percentile(durations, 5)),
            "p95": float(np.percentile(durations, 95)),
        }

        summary["duration_outliers"] = summarize_duration_outliers(ok_rows)

        summary["padding"] = {
            "num_clips_with_padding": num_padding,
            "fraction_clips_with_padding": (
                num_padding / len(ok_rows)
            ),
            "mean_padding_ratio": mean(padding_ratios),
            "num_clips_truncated": num_truncated,
            "fraction_clips_truncated": (
                num_truncated / len(ok_rows)
            ),
            "mean_truncated_ratio_vs_target": mean(
                truncation_ratios
            ),
        }

        num_clips_partial_padding = sum(
            1
            for row in ok_rows
            if int(row["patches_partial_padding"]) > 0
        )
        num_clips_full_padding = sum(
            1
            for row in ok_rows
            if int(row["patches_full_padding"]) > 0
        )

        summary["patch_padding"] = {
            "patches_full_real": total_full_real_patches,
            "patches_partial_padding": total_partial_padding_patches,
            "patches_full_padding": total_full_padding_patches,
            "padding_affected_patches": padding_affected_patches,
            "total_patches": total_patches,
            "fraction_full_real": (
                total_full_real_patches / total_patches
                if total_patches > 0 else 0.0
            ),
            "fraction_partial_padding": (
                total_partial_padding_patches / total_patches
                if total_patches > 0 else 0.0
            ),
            "fraction_full_padding": (
                total_full_padding_patches / total_patches
                if total_patches > 0 else 0.0
            ),
            "fraction_padding_affected": (
                padding_affected_patches / total_patches
                if total_patches > 0
                else 0.0
            ),
            "num_clips_with_partial_padding_patch": num_clips_partial_padding,
            "fraction_clips_with_partial_padding_patch": (
                num_clips_partial_padding / len(ok_rows)
            ),
            "num_clips_with_full_padding_patch": num_clips_full_padding,
            "fraction_clips_with_full_padding_patch": (
                num_clips_full_padding / len(ok_rows)
            ),
        }

    return rows, summary


def _running_stats_update(
        total_sum: float,
        total_squared_sum: float,
        total_count: int,
        tensor: torch.Tensor,
) -> tuple[float, float, int]:
    """Aggiorna statistiche streaming per un tensore."""

    values = tensor.to(torch.float64)

    return (
        total_sum + float(values.sum().item()),
        total_squared_sum
        + float((values * values).sum().item()),
        total_count + int(values.numel()),
    )


def _finalize_stats(
        total_sum: float,
        total_squared_sum: float,
        total_count: int,
) -> dict[str, float | int]:
    """Converte somme streaming in mean/std."""

    if total_count == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "num_values": 0,
        }

    value_mean = total_sum / total_count
    variance = max(
        total_squared_sum / total_count
        - value_mean ** 2,
        0.0,
    )

    return {
        "mean": value_mean,
        "std": variance ** 0.5,
        "num_values": total_count,
    }


@torch.no_grad()
def analyze_logmel_sample(
        rows: list[dict[str, Any]],
        cfg,
        mode: str,
        max_files: int,
        seed: int,
        histogram_values: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    Confronta statistiche log-Mel con e senza padding su un campione fisso.

    Per un confronto corretto, i frame oltre ``target_frames`` vengono esclusi
    anche dalle statistiche ``real_only``, perché il modello non li vede.
    """

    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    rng = random.Random(seed)

    if len(valid_rows) > max_files:
        selected = rng.sample(
            valid_rows,
            max_files,
        )
    else:
        selected = valid_rows

    preprocessor = AudioPreprocessor(
        cfg=cfg.audio,
        norm_mean=None,
        norm_std=None,
    )
    target_frames = preprocessor.target_frames(
        mode
    )

    real_sum = 0.0
    real_squared_sum = 0.0
    real_count = 0

    padded_sum = 0.0
    padded_squared_sum = 0.0
    padded_count = 0

    real_hist: list[float] = []
    padded_hist: list[float] = []

    num_ok = 0
    num_skipped = 0

    per_file_hist_cap = max(
        histogram_values // max(len(selected), 1),
        1,
    )

    np_rng = np.random.default_rng(seed)

    for index, row in enumerate(selected, start=1):
        wav_path = Path(
            row["resolved_path"]
        )

        try:
            raw_fbank = preprocessor.process_fbank_raw(
                wav_path
            )

            real_visible = raw_fbank[
                :target_frames
            ]
            padded = preprocessor._pad_or_truncate(
                raw_fbank,
                target_frames,
            )

            (
                real_sum,
                real_squared_sum,
                real_count,
            ) = _running_stats_update(
                real_sum,
                real_squared_sum,
                real_count,
                real_visible,
            )

            (
                padded_sum,
                padded_squared_sum,
                padded_count,
            ) = _running_stats_update(
                padded_sum,
                padded_squared_sum,
                padded_count,
                padded,
            )

            for tensor, target in (
                    (real_visible, real_hist),
                    (padded, padded_hist),
            ):
                flat = tensor.detach().cpu().numpy().reshape(-1)
                if flat.size > per_file_hist_cap:
                    indices = np_rng.choice(
                        flat.size,
                        size=per_file_hist_cap,
                        replace=False,
                    )
                    flat = flat[indices]
                target.extend(
                    float(value)
                    for value in flat
                )

            num_ok += 1

        except Exception as error:
            num_skipped += 1
            print(
                f"  [WARN log-Mel] {wav_path}: {error}"
            )

        if (
                index % 250 == 0
                or index == len(selected)
        ):
            print(
                f"  Log-Mel {index}/{len(selected)}"
            )

    real_stats = _finalize_stats(
        real_sum,
        real_squared_sum,
        real_count,
    )
    padded_stats = _finalize_stats(
        padded_sum,
        padded_squared_sum,
        padded_count,
    )

    summary = {
        "num_files_requested": len(selected),
        "num_files_used": num_ok,
        "num_files_skipped": num_skipped,
        "real_visible_frames": real_stats,
        "after_pad_truncate": padded_stats,
        "delta_mean_padding_minus_real": (
            float(padded_stats["mean"])
            - float(real_stats["mean"])
        ),
        "delta_std_padding_minus_real": (
            float(padded_stats["std"])
            - float(real_stats["std"])
        ),
    }

    return (
        summary,
        np.asarray(real_hist, dtype=np.float32),
        np.asarray(padded_hist, dtype=np.float32),
    )


def write_index_csv(
        rows: list[dict[str, Any]],
        output_path: Path,
) -> None:
    """Salva l'indice del dataset in CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def _require_matplotlib():
    """Importa matplotlib solo quando servono i grafici."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Per generare i grafici installare l'extra analysis: "
            "pip install -e \".[analysis]\""
        ) from error

    return plt


def plot_basic_distributions(
        rows: list[dict[str, Any]],
        summary: dict[str, Any],
        output_dir: Path,
        real_hist: np.ndarray | None,
        padded_hist: np.ndarray | None,
) -> None:
    """Genera i grafici principali dell'EDA."""

    plt = _require_matplotlib()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    if ok_rows:
        durations = [
            float(row["duration_sec"])
            for row in ok_rows
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(durations, bins=60)
        ax.set_title("Distribuzione durata audio")
        ax.set_xlabel("Durata [s]")
        ax.set_ylabel("Numero clip")
        fig.tight_layout()
        fig.savefig(
            output_dir / "duration_distribution.png",
            dpi=160,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(durations, bins=80)
        ax.set_yscale("log")
        ax.set_title("Distribuzione durata audio — scala log")
        ax.set_xlabel("Durata [s]")
        ax.set_ylabel("Numero clip")
        fig.tight_layout()
        fig.savefig(
            output_dir / "duration_distribution_log.png",
            dpi=160,
        )
        plt.close(fig)

        short_durations = [value for value in durations if value < 9.5]
        if short_durations:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(short_durations, bins=60)
            ax.set_yscale("log")
            ax.set_title("Coda corta delle durate (< 9.5 s)")
            ax.set_xlabel("Durata [s]")
            ax.set_ylabel("Numero clip")
            fig.tight_layout()
            fig.savefig(
                output_dir / "duration_short_tail.png",
                dpi=160,
            )
            plt.close(fig)

        padding_ratios = [
            float(row["padding_ratio"])
            for row in ok_rows
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(padding_ratios, bins=50)
        ax.set_title("Quantità di padding per clip")
        ax.set_xlabel("Padding / target frames")
        ax.set_ylabel("Numero clip")
        fig.tight_layout()
        fig.savefig(
            output_dir / "padding_ratio_distribution.png",
            dpi=160,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(padding_ratios, bins=60)
        ax.set_yscale("log")
        ax.set_title("Quantità di padding per clip — scala log")
        ax.set_xlabel("Padding / target frames")
        ax.set_ylabel("Numero clip")
        fig.tight_layout()
        fig.savefig(
            output_dir / "padding_ratio_distribution_log.png",
            dpi=160,
        )
        plt.close(fig)

    patch_padding = summary.get("patch_padding", {})
    if patch_padding and patch_padding.get("total_patches", 0):
        labels = ["reali", "parziali", "padding"]
        values = [
            patch_padding.get("patches_full_real", 0),
            patch_padding.get("patches_partial_padding", 0),
            patch_padding.get("patches_full_padding", 0),
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values)
        ax.set_title("Composizione delle patch rispetto al padding")
        ax.set_ylabel("Numero patch")
        fig.tight_layout()
        fig.savefig(
            output_dir / "patch_padding_composition.png",
            dpi=160,
        )
        plt.close(fig)

    for key, filename, title, xlabel in (
            (
                "sample_rate_counts",
                "sample_rate_distribution.png",
                "Distribuzione sample rate",
                "Sample rate [Hz]",
            ),
            (
                "channel_counts",
                "channel_distribution.png",
                "Distribuzione numero canali",
                "Canali",
            ),
    ):
        counts = summary.get(key, {})
        if counts:
            labels = list(counts.keys())
            values = list(counts.values())
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(labels, values)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Numero clip")
            fig.tight_layout()
            fig.savefig(
                output_dir / filename,
                dpi=160,
            )
            plt.close(fig)

    if (
            real_hist is not None
            and padded_hist is not None
            and real_hist.size > 0
            and padded_hist.size > 0
    ):
        combined = np.concatenate(
            [real_hist, padded_hist]
        )
        low, high = np.percentile(
            combined,
            [1, 99],
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(
            real_hist,
            bins=80,
            range=(low, high),
            density=True,
            alpha=0.55,
            label="solo frame reali",
        )
        ax.hist(
            padded_hist,
            bins=80,
            range=(low, high),
            density=True,
            alpha=0.55,
            label="dopo pad/truncate",
        )
        ax.set_title("Distribuzione log-Mel: impatto del padding")
        ax.set_xlabel("Valore log-Mel")
        ax.set_ylabel("Densità")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "logmel_padding_comparison.png",
            dpi=160,
        )
        plt.close(fig)

    top_labels = summary.get("top_labels", [])[:20]
    if top_labels:
        labels = [item[0] for item in top_labels][::-1]
        values = [item[1] for item in top_labels][::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(labels, values)
        ax.set_title("20 label più frequenti")
        ax.set_xlabel("Numero clip")
        fig.tight_layout()
        fig.savefig(
            output_dir / "top_labels.png",
            dpi=160,
        )
        plt.close(fig)



def plot_quality_distributions(
        quality_rows: list[dict[str, Any]],
        output_dir: Path,
) -> None:
    """Visualizza RMS e clipping del campione quality-audit."""

    if not quality_rows:
        return

    plt = _require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    rms_values = np.asarray(
        [max(float(row["rms"]), 1.0e-12) for row in quality_rows],
        dtype=np.float64,
    )
    clipping = np.asarray(
        [float(row["clip_fraction_ge_0_999"]) for row in quality_rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log10(rms_values), bins=60)
    ax.set_title("Distribuzione RMS waveform (campione)")
    ax.set_xlabel("log10(RMS)")
    ax.set_ylabel("Numero clip")
    fig.tight_layout()
    fig.savefig(output_dir / "waveform_rms_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    positive = clipping[clipping > 0]
    if positive.size > 0:
        ax.hist(positive, bins=60)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "Nessun campione clippato", ha="center", va="center")
    ax.set_title("Frazione campioni |x| ≥ 0.999 (campione)")
    ax.set_xlabel("Frazione campioni clippati")
    ax.set_ylabel("Numero clip")
    fig.tight_layout()
    fig.savefig(output_dir / "waveform_clipping_distribution.png", dpi=160)
    plt.close(fig)


def plot_examples(
        rows: list[dict[str, Any]],
        cfg,
        mode: str,
        output_dir: Path,
        example_count: int,
) -> None:
    """Salva esempi waveform + log-Mel per clip rappresentative."""

    if example_count <= 0:
        return

    plt = _require_matplotlib()

    valid_rows = sorted(
        (
            row
            for row in rows
            if row.get("status") == "ok"
        ),
        key=lambda row: float(row["duration_sec"]),
    )

    if not valid_rows:
        return

    if example_count == 1:
        positions = [len(valid_rows) // 2]
    else:
        positions = np.linspace(
            0,
            len(valid_rows) - 1,
            num=example_count,
            dtype=int,
        ).tolist()

    preprocessor = AudioPreprocessor(
        cfg=cfg.audio,
        norm_mean=None,
        norm_std=None,
    )
    target_frames = preprocessor.target_frames(mode)

    examples_dir = output_dir / "examples"
    examples_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for example_index, position in enumerate(
            positions,
            start=1,
    ):
        row = valid_rows[position]
        wav_path = Path(row["resolved_path"])

        data, sample_rate = sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(
            data.T.copy()
        )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        raw_fbank = preprocessor.process_fbank_raw(
            wav_path
        )
        padded = preprocessor._pad_or_truncate(
            raw_fbank,
            target_frames,
        )

        time_axis = np.arange(
            waveform.shape[-1]
        ) / sample_rate

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11, 9),
        )

        axes[0].plot(
            time_axis,
            waveform[0].numpy(),
            linewidth=0.6,
        )
        axes[0].set_title(
            f"Waveform — {wav_path.name}"
        )
        axes[0].set_xlabel("Tempo [s]")
        axes[0].set_ylabel("Ampiezza")

        raw_numpy = raw_fbank.T.numpy()
        padded_numpy = padded.T.numpy()
        vmin, vmax = np.percentile(raw_numpy, [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.min(raw_numpy))
            vmax = float(np.max(raw_numpy))

        axes[1].imshow(
            raw_numpy,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title(
            f"Log-Mel reale ({raw_fbank.shape[0]} frame)"
        )
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Mel bin")

        axes[2].imshow(
            padded_numpy,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        real_visible_frames = min(int(raw_fbank.shape[0]), target_frames)
        if real_visible_frames < target_frames:
            axes[2].axvline(
                real_visible_frames - 0.5,
                linestyle="--",
                linewidth=1.2,
                label="inizio padding",
            )
            axes[2].legend(loc="upper right")
        axes[2].set_title(
            f"Log-Mel dopo pad/truncate ({target_frames} frame)"
        )
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("Mel bin")

        fig.tight_layout()
        fig.savefig(
            examples_dir
            / f"example_{example_index:02d}.png",
            dpi=160,
        )
        plt.close(fig)


def print_summary(
        summary: dict[str, Any],
        mel_summary: dict[str, Any] | None,
        quality_summary: dict[str, Any] | None,
) -> None:
    """Stampa una sintesi leggibile dell'audit."""

    print("\n" + "=" * 70)
    print("EDA DATASET — SUMMARY")
    print("=" * 70)

    print(
        f"Entry analizzate: {summary['num_entries_scanned']}"
    )
    print(
        f"Stati: {summary['status_counts']}"
    )
    print(
        f"Sample rate: {summary['sample_rate_counts']}"
    )
    print(
        f"Canali: {summary['channel_counts']}"
    )

    duration = summary.get("duration_sec", {})
    if duration:
        print(
            "Durata [s]: "
            f"min={duration['min']:.3f}, "
            f"media={duration['mean']:.3f}, "
            f"mediana={duration['median']:.3f}, "
            f"max={duration['max']:.3f}"
        )

    duration_outliers = summary.get("duration_outliers", {})
    thresholds = duration_outliers.get("threshold_counts", {})
    if thresholds:
        print("Clip corte:")
        for item in thresholds.values():
            print(
                f"  < {item['threshold_sec']:.2f} s: "
                f"{item['count']} ({item['fraction']:.4%})"
            )

    padding = summary.get("padding", {})
    if padding:
        print(
            "Clip con padding: "
            f"{padding['fraction_clips_with_padding']:.2%}"
        )
        print(
            "Padding medio sul target: "
            f"{padding['mean_padding_ratio']:.2%}"
        )
        print(
            "Clip troncate: "
            f"{padding['fraction_clips_truncated']:.2%}"
        )

    patch_padding = summary.get(
        "patch_padding",
        {},
    )
    if patch_padding:
        print(
            "Patch toccate dal padding: "
            f"{patch_padding['fraction_padding_affected']:.2%}"
        )
        print(
            "  parzialmente padding: "
            f"{patch_padding.get('fraction_partial_padding', 0.0):.2%}"
        )
        print(
            "  completamente padding: "
            f"{patch_padding.get('fraction_full_padding', 0.0):.2%}"
        )

    print(
        f"Label uniche: {summary['num_unique_labels']}"
    )
    print(
        "Source ID duplicati: "
        f"{summary['num_source_ids_with_duplicates']}"
    )

    if mel_summary is not None:
        real = mel_summary["real_visible_frames"]
        padded = mel_summary["after_pad_truncate"]
        print("\nLog-Mel sul campione:")
        print(
            "  solo frame reali: "
            f"mean={real['mean']:.6f}, "
            f"std={real['std']:.6f}"
        )
        print(
            "  con pad/truncate: "
            f"mean={padded['mean']:.6f}, "
            f"std={padded['std']:.6f}"
        )
        print(
            "  delta mean: "
            f"{mel_summary['delta_mean_padding_minus_real']:.6f}"
        )
        print(
            "  delta std: "
            f"{mel_summary['delta_std_padding_minus_real']:.6f}"
        )


    if quality_summary is not None:
        print("\nQuality audit waveform:")
        print(
            f"  file usati: {quality_summary.get('num_files_used', 0)}"
        )
        rms = quality_summary.get("rms", {})
        if rms:
            print(
                "  RMS: "
                f"min={rms['min']:.6g}, "
                f"mediana={rms['median']:.6g}, "
                f"p99={rms['p99']:.6g}"
            )
        silence = quality_summary.get("near_silence", {})
        if silence:
            item = silence.get("rms_le_1e_4", {})
            print(
                "  quasi silenziose (RMS <= 1e-4): "
                f"{item.get('count', 0)} ({item.get('fraction', 0.0):.3%})"
            )
        clipping = quality_summary.get("clipping", {})
        if clipping:
            item = clipping.get("clip_fraction_ge_0_1pct", {})
            print(
                "  clipping >=0.1% campioni: "
                f"{item.get('count', 0)} ({item.get('fraction', 0.0):.3%})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA e audit dataset per MAE-AST V2"
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifest JSON da analizzare.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root per risolvere i path relativi.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in cui salvare indice, summary e grafici.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config esperimento opzionale, unita a base.yaml.",
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune"],
        default="pretrain",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limita l'audit a N file, utile per smoke test.",
    )
    parser.add_argument(
        "--mel-max-files",
        type=int,
        default=2000,
        help=(
            "Numero massimo di clip usate per il confronto log-Mel. "
            "Usare 0 per disabilitarlo."
        ),
    )
    parser.add_argument(
        "--histogram-values",
        type=int,
        default=250000,
        help="Numero massimo indicativo di valori per gli istogrammi log-Mel.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=3,
        help="Numero di esempi waveform/spettrogramma da salvare.",
    )
    parser.add_argument(
        "--quality-max-files",
        type=int,
        default=5000,
        help=(
            "Numero di WAV per quality audit (RMS/clipping). "
            "Usare 0 per disabilitare, -1 per tutto il dataset."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per il campionamento deterministico.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Non genera grafici.",
    )

    args = parser.parse_args()

    cfg = load_config(
        config_path=args.config,
    )

    entries = load_manifest(
        args.manifest
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows, summary = audit_entries(
        entries=entries,
        dataset_root=args.dataset_root,
        cfg=cfg,
        mode=args.mode,
        max_files=args.max_files,
    )

    write_index_csv(
        rows,
        output_dir / "dataset_index.csv",
    )

    mel_summary = None
    real_hist = None
    padded_hist = None

    if args.mel_max_files > 0:
        (
            mel_summary,
            real_hist,
            padded_hist,
        ) = analyze_logmel_sample(
            rows=rows,
            cfg=cfg,
            mode=args.mode,
            max_files=args.mel_max_files,
            seed=args.seed,
            histogram_values=args.histogram_values,
        )

        with (
                output_dir / "logmel_stats_sample.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                mel_summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

    quality_summary = None
    quality_rows: list[dict[str, Any]] = []

    if args.quality_max_files != 0:
        quality_summary, quality_rows = analyze_waveform_quality_sample(
            rows=rows,
            max_files=args.quality_max_files,
            seed=args.seed + 1009,
        )
        write_index_csv(
            quality_rows,
            output_dir / "waveform_quality_sample.csv",
        )
        with (
                output_dir / "waveform_quality_sample.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                quality_summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

    summary["logmel_sample"] = mel_summary
    summary["waveform_quality_sample"] = quality_summary

    with (
            output_dir / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if not args.no_plots:
        plot_basic_distributions(
            rows=rows,
            summary=summary,
            output_dir=output_dir / "plots",
            real_hist=real_hist,
            padded_hist=padded_hist,
        )

        plot_quality_distributions(
            quality_rows=quality_rows,
            output_dir=output_dir / "plots",
        )

        plot_examples(
            rows=rows,
            cfg=cfg,
            mode=args.mode,
            output_dir=output_dir / "plots",
            example_count=args.example_count,
        )

    print_summary(
        summary,
        mel_summary,
        quality_summary,
    )

    print("\nOutput:")
    print(
        f"  {output_dir / 'dataset_index.csv'}"
    )
    print(
        f"  {output_dir / 'summary.json'}"
    )
    if mel_summary is not None:
        print(
            f"  {output_dir / 'logmel_stats_sample.json'}"
        )
    if quality_summary is not None:
        print(
            f"  {output_dir / 'waveform_quality_sample.json'}"
        )
        print(
            f"  {output_dir / 'waveform_quality_sample.csv'}"
        )
    if not args.no_plots:
        print(
            f"  {output_dir / 'plots'}"
        )


if __name__ == "__main__":
    main()
