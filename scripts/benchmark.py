"""
Benchmark compute/memory:

    MAE-AST
        VS
    FullSequenceProxy

Il benchmark utilizza un batch sintetico già residente
in memoria per isolare il costo del modello da:

- accesso al disco;
- decoding degli audio;
- preprocessing;
- DataLoader.

Vengono misurati:

- tempo medio per step;
- examples/sec;
- memoria GPU allocata;
- picco di memoria GPU.
"""

from __future__ import annotations

import argparse
import gc
import time

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
        "--set",
        nargs="*",
        default=None,
    )

    return parser.parse_args()


def current_gpu_memory_mb(
        device: torch.device,
):
    """Memoria GPU attualmente allocata."""

    if device.type != "cuda":
        return None

    return (
            torch.cuda.memory_allocated(
                device
            )
            / (1024 ** 2)
    )


def peak_gpu_memory_mb(
        device: torch.device,
):
    """Picco di memoria GPU allocata."""

    if device.type != "cuda":
        return None

    return (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 2)
    )


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
            output = (
                model.forward_pretrain(
                    spectrogram
                )
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

    # -------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------

    print(
        f"\nWarmup {name}: "
        f"{warmup_steps} step"
    )

    for _ in range(
            warmup_steps
    ):
        run_step()

    sync_device(device)

    # Gli stati dell'optimizer sono ormai inizializzati.
    baseline_memory = (
        current_gpu_memory_mb(
            device
        )
    )

    reset_peak_gpu_memory(
        device
    )

    # -------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------

    sync_device(device)

    start = time.perf_counter()

    for _ in range(
            benchmark_steps
    ):
        run_step()

    sync_device(device)

    elapsed = (
            time.perf_counter()
            - start
    )

    batch_size = (
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

    peak_memory = (
        peak_gpu_memory_mb(
            device
        )
    )

    result = {
        "model":
            name,

        "steps":
            benchmark_steps,

        "batch_size":
            batch_size,

        "total_time_sec":
            elapsed,

        "step_time_sec_avg":
            step_time,

        "examples_per_sec":
            examples_per_sec,

        "baseline_gpu_mem_mb":
            baseline_memory,

        "peak_gpu_mem_mb":
            peak_memory,
    }

    print(
        f"\n{name}"
    )

    print(
        f"  step time: "
        f"{step_time:.6f} s"
    )

    print(
        f"  examples/s: "
        f"{examples_per_sec:.2f}"
    )

    if baseline_memory is not None:
        print(
            f"  GPU baseline: "
            f"{baseline_memory:.1f} MB"
        )

    if peak_memory is not None:
        print(
            f"  GPU peak: "
            f"{peak_memory:.1f} MB"
        )

    return result


def cleanup():
    """
    Libera gli oggetti CUDA tra i due benchmark.
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

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
        else int(
            cfg.training.batch_size
        )
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
        f"Mask ratio: "
        f"{cfg.masking.ratio}"
    )

    # -------------------------------------------------------------
    # Batch identico per entrambi i modelli
    # -------------------------------------------------------------

    spectrogram = torch.randn(
        batch_size,
        int(
            cfg.audio.pretrain_target_frames
        ),
        int(
            cfg.audio.n_mels
        ),
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

    # -------------------------------------------------------------
    # Confronto
    # -------------------------------------------------------------

    speedup = (
            proxy_result[
                "step_time_sec_avg"
            ]
            /
            mae_result[
                "step_time_sec_avg"
            ]
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

    mae_peak = (
        mae_result[
            "peak_gpu_mem_mb"
        ]
    )

    proxy_peak = (
        proxy_result[
            "peak_gpu_mem_mb"
        ]
    )

    if (
            mae_peak is not None
            and proxy_peak is not None
    ):

        memory_reduction = (
                                   1.0
                                   - mae_peak
                                   / proxy_peak
                           ) * 100.0

        print(
            f"Riduzione memoria MAE: "
            f"{memory_reduction:.2f}%"
        )


if __name__ == "__main__":
    main()