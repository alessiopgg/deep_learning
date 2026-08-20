import csv
import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark.py"
)

spec = importlib.util.spec_from_file_location(
    "benchmark_script",
    SCRIPT_PATH,
)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)

CSV_FIELDS = benchmark.CSV_FIELDS
build_csv_rows = benchmark.build_csv_rows
save_benchmark_outputs = benchmark.save_benchmark_outputs


def _fake_run_record(run_id="benchmark_test"):
    return {
        "run_id": run_id,
        "timestamp": "2026-08-20T16:30:00+02:00",
        "config": {
            "encoder_layers": 12,
            "decoder_layers": 2,
            "mask_strategy": "chunk",
            "mask_ratio": 0.75,
            "batch_size": 32,
            "warmup_steps": 20,
            "steps": 100,
            "seed": 42,
        },
        "hardware": {
            "device": "cuda",
            "gpu_name": "Test GPU",
            "gpu_total_memory_mb": 32768.0,
            "torch_version": "test",
            "cuda_version": "test",
            "python_version": "test",
        },
        "mae": {
            "model": "MAE-AST",
            "steps": 100,
            "batch_size": 32,
            "total_parameters": 100,
            "trainable_parameters": 100,
            "encoder_tokens": 128,
            "decoder_tokens": 512,
            "amp_enabled": True,
            "total_time_sec": 10.0,
            "step_time_sec_avg": 0.1,
            "examples_per_sec": 320.0,
            "baseline_gpu_mem_allocated_mb": 1000.0,
            "baseline_gpu_mem_reserved_mb": 1200.0,
            "peak_gpu_mem_allocated_mb": 8000.0,
            "peak_gpu_mem_reserved_mb": 9000.0,
        },
        "proxy": {
            "model": "FullSequenceProxy",
            "steps": 100,
            "batch_size": 32,
            "total_parameters": 101,
            "trainable_parameters": 101,
            "encoder_tokens": 512,
            "decoder_tokens": 512,
            "amp_enabled": True,
            "total_time_sec": 20.0,
            "step_time_sec_avg": 0.2,
            "examples_per_sec": 160.0,
            "baseline_gpu_mem_allocated_mb": 1500.0,
            "baseline_gpu_mem_reserved_mb": 1700.0,
            "peak_gpu_mem_allocated_mb": 12000.0,
            "peak_gpu_mem_reserved_mb": 13000.0,
        },
        "comparison": {
            "speedup_mae_vs_proxy": 2.0,
            "memory_reduction_allocated_percent": 33.333333,
            "memory_reduction_reserved_percent": 30.769231,
        },
    }


def test_build_csv_rows_contains_one_row_per_model():
    rows = build_csv_rows(
        _fake_run_record()
    )

    assert len(rows) == 2
    assert list(rows[0].keys()) == CSV_FIELDS
    assert rows[0]["model"] == "MAE-AST"
    assert rows[0]["encoder_tokens"] == 128
    assert rows[1]["model"] == "FullSequenceProxy"
    assert rows[1]["encoder_tokens"] == 512
    assert rows[0]["speedup_mae_vs_proxy"] == 2.0


def test_save_benchmark_outputs_writes_json_and_appends_csv(tmp_path):
    first_record = _fake_run_record(
        "benchmark_first"
    )

    json_path, csv_path = save_benchmark_outputs(
        tmp_path,
        first_record,
    )

    assert json_path.exists()
    assert csv_path.exists()

    with json_path.open(
            "r",
            encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert loaded["run_id"] == "benchmark_first"
    assert loaded["comparison"]["speedup_mae_vs_proxy"] == 2.0

    second_record = _fake_run_record(
        "benchmark_second"
    )

    save_benchmark_outputs(
        tmp_path,
        second_record,
    )

    with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    # Due modelli per run, due run totali.
    assert len(rows) == 4
    assert rows[0]["run_id"] == "benchmark_first"
    assert rows[1]["run_id"] == "benchmark_first"
    assert rows[2]["run_id"] == "benchmark_second"
    assert rows[3]["run_id"] == "benchmark_second"
