import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_esc50_5fold.py"
)

spec = importlib.util.spec_from_file_location(
    "run_esc50_5fold",
    SCRIPT_PATH,
)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def test_validate_folds_deduplicates_and_preserves_order():
    assert runner.validate_folds([3, 1, 3, 5]) == [3, 1, 5]

    with pytest.raises(ValueError):
        runner.validate_folds([0])

    with pytest.raises(ValueError):
        runner.validate_folds([6])


def test_build_command_uses_same_checkpoint_and_fold_specific_name():
    command = runner.build_finetune_command(
        python_executable="python",
        config="configs/experiments/esc50_finetune_6l.yaml",
        local_config="configs/local/windows_esc50_fold2.yaml",
        checkpoint="outputs/pretrain/best.pt",
        experiment_name="mae6l_fold2",
        output_root="outputs",
        logging_backend="wandb",
        wandb_mode="offline",
        wandb_group="mae6l",
        extra_overrides=["data.num_workers=0"],
    )

    assert "outputs/pretrain/best.pt" in command
    assert "configs/local/windows_esc50_fold2.yaml" in command
    assert "experiment.name=mae6l_fold2" in command
    assert "logging.wandb_group=mae6l" in command
    assert "data.num_workers=0" in command


def test_read_best_accuracy(tmp_path):
    log_path = tmp_path / "finetune_log.jsonl"
    records = [
        {"epoch": 1, "validation": {"accuracy": 0.60}},
        {"epoch": 2, "validation": {"accuracy": 0.75}},
        {"epoch": 3, "validation": {"accuracy": 0.70}},
    ]

    with log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    accuracy, epoch = runner.read_best_accuracy(log_path)
    assert accuracy == pytest.approx(0.75)
    assert epoch == 2


def test_aggregate_five_folds():
    summaries = [
        {"fold": 1, "best_val_accuracy": 0.80, "best_epoch": 20},
        {"fold": 2, "best_val_accuracy": 0.85, "best_epoch": 21},
        {"fold": 3, "best_val_accuracy": 0.90, "best_epoch": 22},
        {"fold": 4, "best_val_accuracy": 0.95, "best_epoch": 23},
        {"fold": 5, "best_val_accuracy": 1.00, "best_epoch": 24},
    ]

    result = runner.aggregate_fold_summaries(summaries)

    assert result["num_folds"] == 5
    assert result["mean_accuracy"] == pytest.approx(0.90)
    assert result["std_population_accuracy"] > 0
    assert result["std_sample_accuracy"] > result["std_population_accuracy"]
