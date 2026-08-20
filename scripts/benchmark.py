"""
Benchmark compute/memory:

    MAE-AST
        VS
    FullSequenceProxy

Il benchmark utilizza un batch sintetico già residente in memoria per isolare
il costo del modello da:

- accesso al disco;
- decoding degli audio;
- preprocessing;
- DataLoader.

Vengono misurati e salvati:

- tempo medio per step;
- examples/sec;
- memoria GPU allocata e riservata;
- picco di memoria GPU allocata e riservata;
- numero di parametri;
- numero di token elaborati da encoder e decoder;
- speedup MAE rispetto al FullSequenceProxy;
- riduzione percentuale di memoria MAE;
- metadata hardware/software utili alla riproducibilità.

Ogni esecuzione produce:

- un JSON completo della singola run;
- due righe appendibili in benchmark_results.csv, una per ciascun modello.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from mae_ast.config import load_config
from mae_ast.losses import pretrain_loss
from mae_ast.models.full_sequence_proxy import FullSequenceProxy
from mae_ast.models.mae_ast import MAEASTModel
from mae_ast.training.utils import (
    amp_enabled,
    build_grad_scaler,
    build_optimizer,
    get_device,
    reset_peak_gpu_memory,
    set_seed,
    sync_device,
)


CSV_FIELDS = [
    "run_id",
    "timestamp",
    "model",
    "encoder_layers",
    "decoder_layers",
    "mask_strategy",
    "mask_ratio",
    "batch_size",
    "warmup_steps",
    "steps",
    "seed",
    "amp_enabled",
    "total_parameters",
    "trainable_parameters",
    "encoder_tokens",
    "decoder_tokens",
    "total_time_sec",
    "step_time_sec_avg",
    "examples_per_sec",
    "baseline_gpu_mem_allocated_mb",
    "baseline_gpu_mem_reserved_mb",
    "peak_gpu_mem_allocated_mb",
    "peak_gpu_mem_reserved_mb",
    "speedup_mae_vs_proxy",
    "memory_reduction_allocated_percent",
    "memory_reduction_reserved_percent",
    "device",
    "gpu_name",
    "gpu_total_memory_mb",
    "torch_version",
    "cuda_version",
    "python_version",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark compute MAE-AST"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmarks",
        help="Directory in cui salvare JSON e CSV del benchmark.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nome opzionale della run. Se assente viene generato automaticamente.",
    )

    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
    )

    return parser.parse_args()


def current_gpu_memory_allocated_mb(
        device: torch.device,
):
    """Memoria GPU attualmente allocata da PyTorch."""

    if device.type != "cuda":
        return None

    return (
        torch.cuda.memory_allocated(device)
        / (1024 ** 2)
    )


def current_gpu_memory_reserved_mb(
        device: torch.device,
):
    """Memoria GPU attualmente riservata dal CUDA allocator di PyTorch."""

    if device.type != "cuda":
        return None

    return (
        torch.cuda.memory_reserved(device)
        / (1024 ** 2)
    )


def peak_gpu_memory_allocated_mb(
        device: torch.device,
):
    """Picco di memoria GPU allocata da PyTorch."""

    if device.type != "cuda":
        return None

    return (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 2)
    )


def peak_gpu_memory_reserved_mb(
        device: torch.device,
):
    """Picco di memoria GPU riservata dal CUDA allocator di PyTorch."""

    if device.type != "cuda":
        return None

    return (
        torch.cuda.max_memory_reserved(device)
        / (1024 ** 2)
    )


def count_parameters(model) -> tuple[int, int]:
    """Restituisce numero totale e numero trainabile di parametri."""

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def extract_token_counts(output) -> tuple[int | None, int | None]:
    """
    Legge direttamente dalle rappresentazioni interne quanti token vengono
    elaborati da encoder e decoder.

    Questo evita di ricostruire il numero di token a partire dal mask ratio e
    rende il risultato robusto alle diverse strategie di masking.
    """

    encoder_latent = getattr(
        output,
        "encoder_latent",
        None,
    )

    decoder_latent = getattr(
        output,
        "decoder_latent",
        None,
    )

    encoder_tokens = (
        int(encoder_latent.shape[1])
        if encoder_latent is not None
        else None
    )

    decoder_tokens = (
        int(decoder_latent.shape[1])
        if decoder_latent is not None
        else None
    )

    return encoder_tokens, decoder_tokens


def hardware_metadata(
        device: torch.device,
) -> dict[str, Any]:
    """Metadata hardware/software necessari per interpretare il benchmark."""

    gpu_name = None
    gpu_total_memory_mb = None

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        properties = torch.cuda.get_device_properties(device)
        gpu_total_memory_mb = (
            properties.total_memory
            / (1024 ** 2)
        )

    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_memory_mb,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def benchmark_model(
        name: str,
        model,
        spectrogram: torch.Tensor,
        cfg,
        device: torch.device,
        warmup_steps: int,
        benchmark_steps: int,
):
    """
    Esegue warmup e benchmark forward + backward + optimizer.
    """

    model = model.to(device)
    model.train()

    total_parameters, trainable_parameters = count_parameters(model)

    optimizer = build_optimizer(
        model.parameters(),
        cfg,
    )

    use_amp = amp_enabled(
        cfg,
        device,
    )

    scaler = build_grad_scaler(
        use_amp
    )

    encoder_tokens = None
    decoder_tokens = None

    # -------------------------------------------------------------
    # Singolo step
    # -------------------------------------------------------------

    def run_step():
        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
        ):
            output = model.forward_pretrain(
                spectrogram
            )

            losses = pretrain_loss(
                output,
                cfg.loss,
            )

        if scaler is not None:
            scaler.scale(
                losses.total
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:
            losses.total.backward()
            optimizer.step()

        return extract_token_counts(output)

    # -------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------

    print(
        f"\nWarmup {name}: "
        f"{warmup_steps} step"
    )

    for _ in range(warmup_steps):
        current_encoder_tokens, current_decoder_tokens = run_step()

        if encoder_tokens is None:
            encoder_tokens = current_encoder_tokens
            decoder_tokens = current_decoder_tokens

    sync_device(device)

    # Gli stati dell'optimizer sono ormai inizializzati.
    baseline_allocated = current_gpu_memory_allocated_mb(
        device
    )

    baseline_reserved = current_gpu_memory_reserved_mb(
        device
    )

    # reset_peak_memory_stats azzera sia allocated sia reserved peak.
    reset_peak_gpu_memory(
        device
    )

    # -------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------

    sync_device(device)
    start = time.perf_counter()

    for _ in range(benchmark_steps):
        current_encoder_tokens, current_decoder_tokens = run_step()

        if encoder_tokens is None:
            encoder_tokens = current_encoder_tokens
            decoder_tokens = current_decoder_tokens

    sync_device(device)

    elapsed = time.perf_counter() - start

    batch_size = int(
        spectrogram.shape[0]
    )

    step_time = (
        elapsed
        / benchmark_steps
    )

    examples_per_sec = (
        batch_size
        * benchmark_steps
        / elapsed
    )

    peak_allocated = peak_gpu_memory_allocated_mb(
        device
    )

    peak_reserved = peak_gpu_memory_reserved_mb(
        device
    )

    result = {
        "model": name,
        "steps": benchmark_steps,
        "batch_size": batch_size,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "encoder_tokens": encoder_tokens,
        "decoder_tokens": decoder_tokens,
        "amp_enabled": use_amp,
        "total_time_sec": elapsed,
        "step_time_sec_avg": step_time,
        "examples_per_sec": examples_per_sec,
        "baseline_gpu_mem_allocated_mb": baseline_allocated,
        "baseline_gpu_mem_reserved_mb": baseline_reserved,
        "peak_gpu_mem_allocated_mb": peak_allocated,
        "peak_gpu_mem_reserved_mb": peak_reserved,
    }

    print(
        f"\n{name}"
    )

    print(
        f"  parametri: "
        f"{total_parameters:,}"
    )

    if encoder_tokens is not None:
        print(
            f"  token encoder: "
            f"{encoder_tokens}"
        )

    if decoder_tokens is not None:
        print(
            f"  token decoder: "
            f"{decoder_tokens}"
        )

    print(
        f"  step time: "
        f"{step_time:.6f} s"
    )

    print(
        f"  examples/s: "
        f"{examples_per_sec:.2f}"
    )

    if baseline_allocated is not None:
        print(
            f"  GPU baseline allocated: "
            f"{baseline_allocated:.1f} MiB"
        )

    if baseline_reserved is not None:
        print(
            f"  GPU baseline reserved: "
            f"{baseline_reserved:.1f} MiB"
        )

    if peak_allocated is not None:
        print(
            f"  GPU peak allocated: "
            f"{peak_allocated:.1f} MiB"
        )

    if peak_reserved is not None:
        print(
            f"  GPU peak reserved: "
            f"{peak_reserved:.1f} MiB"
        )

    return result


def cleanup():
    """Libera gli oggetti CUDA tra i due benchmark."""

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_percent_reduction(
        mae_value: float | None,
        proxy_value: float | None,
) -> float | None:
    """Riduzione percentuale di MAE rispetto al proxy."""

    if (
            mae_value is None
            or proxy_value is None
            or proxy_value == 0
    ):
        return None

    return (
        1.0
        - mae_value / proxy_value
    ) * 100.0


def make_run_id(
        run_name: str | None,
        timestamp: str,
) -> str:
    """Crea un identificatore file-safe per la run."""

    if run_name is None:
        return f"benchmark_{timestamp}"

    safe_name = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in run_name.strip()
    )

    if not safe_name:
        safe_name = "benchmark"

    return f"{safe_name}_{timestamp}"


def build_csv_rows(
        run_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Converte il record completo in due righe flat per il CSV master."""

    common = {
        "run_id": run_record["run_id"],
        "timestamp": run_record["timestamp"],
        "encoder_layers": run_record["config"]["encoder_layers"],
        "decoder_layers": run_record["config"]["decoder_layers"],
        "mask_strategy": run_record["config"]["mask_strategy"],
        "mask_ratio": run_record["config"]["mask_ratio"],
        "batch_size": run_record["config"]["batch_size"],
        "warmup_steps": run_record["config"]["warmup_steps"],
        "steps": run_record["config"]["steps"],
        "seed": run_record["config"]["seed"],
        "speedup_mae_vs_proxy": run_record["comparison"]["speedup_mae_vs_proxy"],
        "memory_reduction_allocated_percent": run_record["comparison"][
            "memory_reduction_allocated_percent"
        ],
        "memory_reduction_reserved_percent": run_record["comparison"][
            "memory_reduction_reserved_percent"
        ],
        **run_record["hardware"],
    }

    rows = []

    for result in (
            run_record["mae"],
            run_record["proxy"],
    ):
        row = {
            **common,
            **result,
        }

        # Il CSV usa un sottoinsieme stabile e ordinato di campi.
        rows.append({
            field: row.get(field)
            for field in CSV_FIELDS
        })

    return rows


def save_benchmark_outputs(
        output_dir: str | Path,
        run_record: dict[str, Any],
) -> tuple[Path, Path]:
    """
    Salva il JSON completo della run e appende le due righe al CSV master.

    Il JSON viene scritto atomicamente per evitare file parziali in caso di
    interruzione durante la scrittura.
    """

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_path / f"{run_record['run_id']}.json"
    json_tmp_path = json_path.with_suffix(".json.tmp")

    with json_tmp_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            run_record,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    json_tmp_path.replace(
        json_path
    )

    csv_path = output_path / "benchmark_results.csv"
    csv_exists = csv_path.exists()

    with csv_path.open(
            "a",
            encoding="utf-8",
            newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if not csv_exists:
            writer.writeheader()

        writer.writerows(
            build_csv_rows(run_record)
        )

    return json_path, csv_path


def main():
    args = parse_args()

    if args.steps <= 0:
        raise ValueError(
            "--steps deve essere maggiore di zero."
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup non può essere negativo."
        )

    cfg = load_config(
        config_path=args.config,
        overrides=args.set,
    )

    set_seed(
        int(cfg.experiment.seed)
    )

    device = get_device()

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(cfg.training.batch_size)
    )

    if batch_size <= 0:
        raise ValueError(
            "Il batch size deve essere maggiore di zero."
        )

    print(
        f"Device: {device}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Encoder layers: "
        f"{cfg.model.encoder.layers}"
    )

    print(
        f"Mask strategy: "
        f"{cfg.masking.strategy}"
    )

    print(
        f"Mask ratio: "
        f"{cfg.masking.ratio}"
    )

    # -------------------------------------------------------------
    # Batch identico per entrambi i modelli
    # -------------------------------------------------------------

    spectrogram = torch.randn(
        batch_size,
        int(cfg.audio.pretrain_target_frames),
        int(cfg.audio.n_mels),
        device=device,
    )

    # -------------------------------------------------------------
    # MAE
    # -------------------------------------------------------------

    mae_model = MAEASTModel(
        cfg
    )

    mae_result = benchmark_model(
        name="MAE-AST",
        model=mae_model,
        spectrogram=spectrogram,
        cfg=cfg,
        device=device,
        warmup_steps=args.warmup,
        benchmark_steps=args.steps,
    )

    del mae_model
    cleanup()

    # -------------------------------------------------------------
    # Full-sequence proxy
    # -------------------------------------------------------------

    proxy_model = FullSequenceProxy(
        cfg
    )

    proxy_result = benchmark_model(
        name="FullSequenceProxy",
        model=proxy_model,
        spectrogram=spectrogram,
        cfg=cfg,
        device=device,
        warmup_steps=args.warmup,
        benchmark_steps=args.steps,
    )

    del proxy_model
    cleanup()

    # -------------------------------------------------------------
    # Confronto
    # -------------------------------------------------------------

    speedup = (
        proxy_result["step_time_sec_avg"]
        / mae_result["step_time_sec_avg"]
    )

    memory_reduction_allocated = safe_percent_reduction(
        mae_result["peak_gpu_mem_allocated_mb"],
        proxy_result["peak_gpu_mem_allocated_mb"],
    )

    memory_reduction_reserved = safe_percent_reduction(
        mae_result["peak_gpu_mem_reserved_mb"],
        proxy_result["peak_gpu_mem_reserved_mb"],
    )

    print(
        "\n=============================="
    )
    print(
        "CONFRONTO FINALE"
    )
    print(
        "=============================="
    )

    print(
        f"Speedup MAE: "
        f"{speedup:.2f}x"
    )

    if memory_reduction_allocated is not None:
        print(
            f"Riduzione peak allocated MAE: "
            f"{memory_reduction_allocated:.2f}%"
        )

    if memory_reduction_reserved is not None:
        print(
            f"Riduzione peak reserved MAE: "
            f"{memory_reduction_reserved:.2f}%"
        )

    # -------------------------------------------------------------
    # Salvataggio strutturato
    # -------------------------------------------------------------

    now = datetime.now().astimezone()
    timestamp_for_file = now.strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    run_id = make_run_id(
        args.run_name,
        timestamp_for_file,
    )

    record = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="milliseconds"),
        "config": {
            "encoder_layers": int(cfg.model.encoder.layers),
            "decoder_layers": int(cfg.model.decoder.layers),
            "mask_strategy": str(cfg.masking.strategy),
            "mask_ratio": float(cfg.masking.ratio),
            "batch_size": int(batch_size),
            "warmup_steps": int(args.warmup),
            "steps": int(args.steps),
            "seed": int(cfg.experiment.seed),
        },
        "hardware": hardware_metadata(device),
        "mae": mae_result,
        "proxy": proxy_result,
        "comparison": {
            "speedup_mae_vs_proxy": speedup,
            "memory_reduction_allocated_percent": memory_reduction_allocated,
            "memory_reduction_reserved_percent": memory_reduction_reserved,
        },
    }

    json_path, csv_path = save_benchmark_outputs(
        output_dir=args.output_dir,
        run_record=record,
    )

    print(
        "\nRisultati salvati:"
    )
    print(
        f"  JSON: {json_path}"
    )
    print(
        f"  CSV:  {csv_path}"
    )


if __name__ == "__main__":
    main()
