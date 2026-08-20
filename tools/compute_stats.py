"""
Calcolo streaming delle statistiche globali log-Mel del dataset.

Per il pretraining V2 le statistiche definitive vengono calcolate SOLO sui
frame realmente derivati dall'audio e appartenenti alla porzione che il modello
può vedere. I frame artificiali aggiunti per raggiungere ``target_frames`` non
contribuiscono quindi a mean e std.

Nella stessa passata viene comunque calcolata anche la statistica equivalente
``con padding a zero``. Questo permette di quantificare l'effetto del padding
senza rileggere una seconda volta l'intero dataset.

Pipeline per ogni file:

    caricamento audio
        ↓
    mono
        ↓
    eventuale resampling
        ↓
    rimozione componente DC
        ↓
    Kaldi fbank / log-Mel RAW
        ↓
    eventuale truncation a target_frames
        ↓
    accumulo statistiche real-only
        ↓
    conteggio virtuale degli zeri di padding
        ↓
    accumulo statistica di confronto padded-input

Le statistiche principali ``mean`` e ``std`` salvate nel JSON corrispondono
per default ai frame reali. Il formato rimane quindi compatibile con
``AudioDataset.load_stats``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mae_ast.config import load_config
from mae_ast.data.audio import AudioPreprocessor


def _mean_std_from_accumulators(
        total_sum: float,
        total_squared_sum: float,
        total_count: int,
) -> tuple[float, float]:
    """Converte gli accumulatori streaming in mean e std di popolazione."""

    if total_count <= 0:
        raise RuntimeError("Nessun valore disponibile per le statistiche.")

    mean = total_sum / total_count

    variance = max(
        (total_squared_sum / total_count) - mean ** 2,
        0.0,
    )

    return mean, variance ** 0.5


@torch.no_grad()
def compute_dataset_stats(
        manifest_path: str | Path,
        dataset_root: str | Path | None,
        audio_cfg,
        mode: str = "pretrain",
        max_files: int | None = None,
        stats_source: str = "real_frames",
) -> dict:
    """
    Calcola mean e std globali in streaming.

    ``real_frames``
        Usa solo i frame realmente derivati dall'audio, dopo l'eventuale
        truncation richiesta dal modello. Il padding artificiale è escluso.

    ``padded_input``
        Usa come statistiche principali la sequenza a lunghezza fissa che il
        modello riceverebbe prima della normalizzazione, includendo gli zeri di
        padding. Viene mantenuto soprattutto per confronti e riproducibilità.

    In entrambi i casi il JSON contiene anche entrambe le versioni complete.
    """

    if stats_source not in {"real_frames", "padded_input"}:
        raise ValueError(
            "stats_source deve essere 'real_frames' oppure 'padded_input'."
        )

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    if dataset_root is not None:
        dataset_root = Path(dataset_root)

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if "data" not in manifest or not isinstance(manifest["data"], list):
        raise ValueError(
            "Manifest non valido: è atteso il campo 'data' contenente una lista."
        )

    entries = manifest["data"]

    if max_files is not None:
        entries = entries[:max_files]

    preprocessor = AudioPreprocessor(
        cfg=audio_cfg,
        norm_mean=None,
        norm_std=None,
    )

    target_frames = preprocessor.target_frames(mode)
    n_mels = int(audio_cfg.n_mels)

    # Accumulatori real-only.
    real_sum = 0.0
    real_squared_sum = 0.0
    real_count = 0

    # Per la variante padded gli zeri non modificano somma e somma dei
    # quadrati: cambia soltanto il numero totale di valori.
    padded_sum = 0.0
    padded_squared_sum = 0.0
    padded_count = 0

    num_ok = 0
    num_skipped = 0
    real_frames_used = 0
    padding_frames_excluded = 0
    truncated_frames = 0
    files_with_padding = 0
    files_truncated = 0

    print(f"File da processare: {len(entries)}")
    print(
        "Statistiche principali: "
        + (
            "frame reali (padding escluso)"
            if stats_source == "real_frames"
            else "input padded a lunghezza fissa"
        )
    )

    for index, entry in enumerate(entries, start=1):
        wav_path = Path(entry["wav"])

        if not wav_path.is_absolute() and dataset_root is not None:
            wav_path = dataset_root / wav_path

        try:
            fbank_raw = preprocessor.process_fbank_raw(wav_path).to(torch.float64)

            raw_frames = int(fbank_raw.shape[0])

            # Il modello non vede eventuali frame oltre target_frames.
            if raw_frames > target_frames:
                fbank_effective = fbank_raw[:target_frames]
                truncated_frames += raw_frames - target_frames
                files_truncated += 1
            else:
                fbank_effective = fbank_raw

            effective_frames = int(fbank_effective.shape[0])
            padding_frames = max(target_frames - effective_frames, 0)

            if padding_frames > 0:
                files_with_padding += 1
                padding_frames_excluded += padding_frames

            current_sum = float(fbank_effective.sum().item())
            current_squared_sum = float(
                (fbank_effective * fbank_effective).sum().item()
            )
            current_count = int(fbank_effective.numel())

            # Real-only.
            real_sum += current_sum
            real_squared_sum += current_squared_sum
            real_count += current_count
            real_frames_used += effective_frames

            # Input padded: gli zeri aggiunti non cambiano i due accumulatori
            # numerici, ma aumentano il denominatore fino a target_frames.
            padded_sum += current_sum
            padded_squared_sum += current_squared_sum
            padded_count += target_frames * n_mels

            num_ok += 1

        except Exception as error:
            num_skipped += 1
            print(f"[WARN] File ignorato: {wav_path}")
            print(f"       Motivo: {error}")

        if index % 1000 == 0 or index == len(entries):
            print(f"Processati {index}/{len(entries)} file...")

    if num_ok == 0:
        raise RuntimeError("Nessun file valido trovato.")

    real_mean, real_std = _mean_std_from_accumulators(
        real_sum,
        real_squared_sum,
        real_count,
    )

    padded_mean, padded_std = _mean_std_from_accumulators(
        padded_sum,
        padded_squared_sum,
        padded_count,
    )

    selected_mean = real_mean if stats_source == "real_frames" else padded_mean
    selected_std = real_std if stats_source == "real_frames" else padded_std

    total_target_frames = num_ok * target_frames

    padding_ratio = (
        padding_frames_excluded / total_target_frames
        if total_target_frames > 0
        else 0.0
    )

    return {
        # Campi principali letti dal training.
        "mean": selected_mean,
        "std": selected_std,
        "stats_source": stats_source,

        # Entrambe le versioni restano salvate per audit scientifico.
        "real_frames": {
            "mean": real_mean,
            "std": real_std,
            "num_values": real_count,
            "num_frames": real_frames_used,
        },
        "padded_input": {
            "mean": padded_mean,
            "std": padded_std,
            "num_values": padded_count,
            "num_frames": total_target_frames,
        },
        "padding_effect": {
            "delta_mean_padded_minus_real": padded_mean - real_mean,
            "delta_std_padded_minus_real": padded_std - real_std,
            "padding_frames": padding_frames_excluded,
            "padding_ratio": padding_ratio,
            "files_with_padding": files_with_padding,
            "files_truncated": files_truncated,
            "truncated_frames": truncated_frames,
        },

        "mode": mode,
        "num_files_used": num_ok,
        "num_files_skipped": num_skipped,
        "target_frames": target_frames,
        "n_mels": n_mels,
        "sample_rate": int(audio_cfg.sample_rate),
        "normalize_to_half_std": bool(audio_cfg.normalize_to_half_std),
    }


def save_stats(
        stats: dict,
        output_path: str | Path,
) -> None:
    """Salva le statistiche in formato JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcolo delle statistiche globali per MAE-AST"
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifest JSON da analizzare.",
    )

    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root del dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="File JSON di output.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Configurazione esperimento opzionale. "
            "base.yaml viene sempre caricato."
        ),
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
        help="Numero massimo di file da utilizzare. Utile per smoke test.",
    )

    parser.add_argument(
        "--stats-source",
        choices=["real_frames", "padded_input"],
        default="real_frames",
        help=(
            "Sorgente dei campi mean/std principali. Default: real_frames, "
            "quindi padding artificiale escluso."
        ),
    )

    args = parser.parse_args()

    cfg = load_config(config_path=args.config)

    stats = compute_dataset_stats(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        audio_cfg=cfg.audio,
        mode=args.mode,
        max_files=args.max_files,
        stats_source=args.stats_source,
    )

    save_stats(stats, args.output)

    real = stats["real_frames"]
    padded = stats["padded_input"]
    effect = stats["padding_effect"]

    print("\n" + "=" * 70)
    print("STATISTICHE DATASET")
    print("=" * 70)

    print(f"File usati:              {stats['num_files_used']}")
    print(f"File skippati:           {stats['num_files_skipped']}")
    print(f"Stats principali:        {stats['stats_source']}")

    print("\nFrame reali:")
    print(f"  Mean:                   {real['mean']:.6f}")
    print(f"  Std:                    {real['std']:.6f}")

    print("\nCon padding a zero:")
    print(f"  Mean:                   {padded['mean']:.6f}")
    print(f"  Std:                    {padded['std']:.6f}")

    print("\nEffetto padding:")
    print(
        f"  Delta mean:             "
        f"{effect['delta_mean_padded_minus_real']:+.6f}"
    )
    print(
        f"  Delta std:              "
        f"{effect['delta_std_padded_minus_real']:+.6f}"
    )
    print(f"  Padding ratio:          {effect['padding_ratio']:.4%}")
    print(f"  File con padding:       {effect['files_with_padding']}")
    print(f"  File troncati:          {effect['files_truncated']}")

    print("\nStatistiche usate dal training:")
    print(f"  Mean:                   {stats['mean']:.6f}")
    print(f"  Std:                    {stats['std']:.6f}")
    print(f"  Output:                 {args.output}")


if __name__ == "__main__":
    main()
