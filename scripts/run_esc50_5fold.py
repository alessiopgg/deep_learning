"""
Runner automatico per il fine-tuning ESC-50 a 5 fold.

Ogni fold parte dallo STESSO checkpoint di pretraining AudioSet e viene
eseguito come processo separato tramite ``scripts/finetune.py``.

Il runner:
1. valida checkpoint e configurazioni locali dei fold;
2. lancia i fold richiesti;
3. mantiene output indipendenti per ogni fold;
4. salva un riepilogo di ogni fold completato;
5. aggrega accuracy media e deviazione standard;
6. permette di saltare fold gia' completati in caso di riavvio.

Non implementa il resume intra-fold: se un fold e' rimasto incompleto,
va rilanciato con ``--overwrite-partial`` oppure con un nuovo prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Iterable, Sequence


DEFAULT_FOLDS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning ESC-50 automatico su 5 fold.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Config esperimento ESC-50, es. esc50_finetune_final_6l.yaml.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root locale del dataset ESC-50.",
    )
    parser.add_argument(
        "--train-manifest-template",
        default="datafiles/esc50_train_fold{fold}.json",
        help="Template manifest train ESC-50; deve contenere {fold}.",
    )
    parser.add_argument(
        "--val-manifest-template",
        default="datafiles/esc50_eval_fold{fold}.json",
        help="Template manifest validation ESC-50; deve contenere {fold}.",
    )
    parser.add_argument(
        "--stats-file",
        default="datafiles/audioset500k_train_stats_padded.json",
        help="Statistiche di normalizzazione usate nel fine-tuning.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint AudioSet pretrained condiviso da tutti i fold.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(DEFAULT_FOLDS),
        help="Fold da eseguire. Default: 1 2 3 4 5.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="esc50_5fold",
        help="Prefisso per i nomi delle run, es. mae_ast_6l_esc50.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root degli output. Viene passata anche ai child process.",
    )
    parser.add_argument(
        "--logging-backend",
        choices=("local", "wandb"),
        default="wandb",
        help="Backend di tracking dei singoli fold.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default="offline",
        help="Modalita' W&B quando logging-backend=wandb.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Gruppo W&B comune. Default: experiment-prefix.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override OmegaConf aggiuntivi passati a ogni fold.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Salta i fold che hanno gia' fold_summary.json valido.",
    )
    parser.add_argument(
        "--overwrite-partial",
        action="store_true",
        help=(
            "Cancella l'output di un fold incompleto prima di rilanciarlo. "
            "Non cancella fold gia' completati se --skip-completed e' attivo."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra i comandi senza avviare il fine-tuning.",
    )

    return parser.parse_args()


def validate_folds(folds: Iterable[int]) -> list[int]:
    result = []
    seen = set()

    for fold in folds:
        fold = int(fold)
        if fold not in DEFAULT_FOLDS:
            raise ValueError(
                f"Fold ESC-50 non valido: {fold}. Valori ammessi: 1..5."
            )
        if fold not in seen:
            result.append(fold)
            seen.add(fold)

    if not result:
        raise ValueError("Specificare almeno un fold.")

    return result


def fold_experiment_name(prefix: str, fold: int) -> str:
    return f"{prefix}_fold{fold}"


def fold_output_dir(output_root: str | Path, prefix: str, fold: int) -> Path:
    return Path(output_root) / fold_experiment_name(prefix, fold)


def build_finetune_command(
    *,
    python_executable: str,
    config: str,
    checkpoint: str,
    experiment_name: str,
    output_root: str,
    logging_backend: str,
    wandb_mode: str,
    wandb_group: str,
    extra_overrides: Sequence[str] | None = None,
) -> list[str]:
    overrides = [
        f"experiment.name={experiment_name}",
        f"checkpoint.output_dir={output_root}",
        f"logging.backend={logging_backend}",
    ]

    if logging_backend == "wandb":
        overrides.extend(
            [
                f"logging.wandb_mode={wandb_mode}",
                f"logging.wandb_group={wandb_group}",
            ]
        )

    if extra_overrides:
        overrides.extend(str(item) for item in extra_overrides)

    return [
        python_executable,
        "scripts/finetune.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--set",
        *overrides,
    ]


def read_best_accuracy(log_path: str | Path) -> tuple[float, int]:
    """Restituisce (best_accuracy, best_epoch) da finetune_log.jsonl."""
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log fine-tuning non trovato: {log_path}")

    best_accuracy = None
    best_epoch = None

    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                accuracy = float(record["validation"]["accuracy"])
                epoch = int(record["epoch"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Record non valido in {log_path}, riga {line_number}."
                ) from exc

            if best_accuracy is None or accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

    if best_accuracy is None or best_epoch is None:
        raise ValueError(f"Nessuna metrica di validation in {log_path}.")

    return best_accuracy, best_epoch


def load_fold_summary(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    required = {"fold", "best_val_accuracy", "best_epoch", "checkpoint"}
    missing = required.difference(summary)
    if missing:
        raise ValueError(
            f"Summary fold incompleto {path}: mancano {sorted(missing)}"
        )

    return summary


def aggregate_fold_summaries(summaries: Sequence[dict]) -> dict:
    if not summaries:
        raise ValueError("Nessun fold disponibile per l'aggregazione.")

    ordered = sorted(summaries, key=lambda item: int(item["fold"]))
    accuracies = [float(item["best_val_accuracy"]) for item in ordered]

    result = {
        "num_folds": len(ordered),
        "folds": ordered,
        "mean_accuracy": fmean(accuracies),
        "std_population_accuracy": pstdev(accuracies),
        "std_sample_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
    }

    return result


def save_global_summary(
    *,
    summary: dict,
    output_root: str | Path,
    experiment_prefix: str,
    checkpoint: str,
    config: str,
) -> tuple[Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / f"{experiment_prefix}_5fold_results.json"
    csv_path = output_root / f"{experiment_prefix}_5fold_results.csv"

    payload = {
        "experiment_prefix": experiment_prefix,
        "config": config,
        "checkpoint": checkpoint,
        **summary,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "best_epoch", "best_val_accuracy"])
        for item in summary["folds"]:
            writer.writerow(
                [
                    item["fold"],
                    item["best_epoch"],
                    item["best_val_accuracy"],
                ]
            )
        writer.writerow([])
        writer.writerow(["mean_accuracy", summary["mean_accuracy"]])
        writer.writerow(
            ["std_population_accuracy", summary["std_population_accuracy"]]
        )
        writer.writerow(["std_sample_accuracy", summary["std_sample_accuracy"]])

    return json_path, csv_path


def main() -> None:
    args = parse_args()
    folds = validate_folds(args.folds)

    checkpoint = Path(args.checkpoint)
    config = Path(args.config)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint}")
    if not config.exists():
        raise FileNotFoundError(f"Config non trovato: {config}")
    dataset_root = Path(args.dataset_root)
    stats_file = Path(args.stats_file)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset ESC-50 non trovato: {dataset_root}"
        )
    if not stats_file.exists():
        raise FileNotFoundError(
            f"File statistiche non trovato: {stats_file}"
        )

    for template_name, template in (
        ("--train-manifest-template", args.train_manifest_template),
        ("--val-manifest-template", args.val_manifest_template),
    ):
        if "{fold}" not in template:
            raise ValueError(
                f"{template_name} deve contenere il placeholder {{fold}}."
            )

    train_manifests = {
        fold: Path(args.train_manifest_template.format(fold=fold))
        for fold in folds
    }
    val_manifests = {
        fold: Path(args.val_manifest_template.format(fold=fold))
        for fold in folds
    }

    missing_manifests = [
        str(path)
        for path in [*train_manifests.values(), *val_manifests.values()]
        if not path.exists()
    ]
    if missing_manifests:
        raise FileNotFoundError(
            "Manifest ESC-50 mancanti:\n  - "
            + "\n  - ".join(missing_manifests)
        )

    wandb_group = args.wandb_group or args.experiment_prefix
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MAE-AST — ESC-50 5-FOLD RUNNER")
    print("=" * 72)
    print(f"Checkpoint condiviso: {checkpoint}")
    print(f"Config:                {config}")
    print(f"Dataset root:          {dataset_root}")
    print(f"Stats:                 {stats_file}")
    print(f"Fold:                  {folds}")
    print(f"Prefix:                {args.experiment_prefix}")
    print(f"Output root:           {output_root}")
    print(f"Logging:               {args.logging_backend}")
    if args.logging_backend == "wandb":
        print(f"W&B mode/group:        {args.wandb_mode} / {wandb_group}")

    summaries: list[dict] = []

    for fold in folds:
        experiment_name = fold_experiment_name(args.experiment_prefix, fold)
        output_dir = fold_output_dir(output_root, args.experiment_prefix, fold)
        summary_path = output_dir / "fold_summary.json"

        print("\n" + "-" * 72)
        print(f"FOLD {fold} — {experiment_name}")
        print("-" * 72)

        if summary_path.exists() and args.skip_completed:
            summary = load_fold_summary(summary_path)
            summaries.append(summary)
            print(
                "Fold gia' completato: "
                f"accuracy={float(summary['best_val_accuracy']):.4f} "
                f"epoch={int(summary['best_epoch'])}. Salto."
            )
            continue

        if output_dir.exists() and any(output_dir.iterdir()):
            if args.overwrite_partial:
                print(f"Rimuovo output parziale: {output_dir}")
                shutil.rmtree(output_dir)
            else:
                raise RuntimeError(
                    f"Output gia' esistente per fold {fold}: {output_dir}\n"
                    "Usa --skip-completed per fold conclusi oppure "
                    "--overwrite-partial per rilanciare un fold incompleto."
                )

        fold_overrides = [
            f"data.train_manifest={train_manifests[fold]}",
            f"data.val_manifest={val_manifests[fold]}",
            f"data.dataset_root={dataset_root}",
            f"data.stats_file={stats_file}",
        ]

        if args.set:
            fold_overrides.extend(args.set)

        command = build_finetune_command(
            python_executable=sys.executable,
            config=str(config),
            checkpoint=str(checkpoint),
            experiment_name=experiment_name,
            output_root=str(output_root),
            logging_backend=args.logging_backend,
            wandb_mode=args.wandb_mode,
            wandb_group=wandb_group,
            extra_overrides=fold_overrides,
        )

        print("Comando:")
        print("  " + subprocess.list2cmdline(command))

        if args.dry_run:
            continue

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Fine-tuning fold {fold} fallito con exit code "
                f"{completed.returncode}."
            )

        log_path = output_dir / "finetune_log.jsonl"
        best_accuracy, best_epoch = read_best_accuracy(log_path)

        summary = {
            "fold": fold,
            "experiment_name": experiment_name,
            "best_val_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "checkpoint": str(checkpoint),
            "config": str(config),
            "train_manifest": str(train_manifests[fold]),
            "val_manifest": str(val_manifests[fold]),
            "dataset_root": str(dataset_root),
            "stats_file": str(stats_file),
            "output_dir": str(output_dir),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        summaries.append(summary)
        print(
            f"Fold {fold} completato: "
            f"best accuracy={best_accuracy:.4f} (epoch {best_epoch})"
        )

    if args.dry_run:
        print("\nDRY RUN completato: nessun training avviato.")
        return

    if len(summaries) != len(folds):
        raise RuntimeError(
            "Non tutti i fold richiesti hanno prodotto un riepilogo valido."
        )

    aggregate = aggregate_fold_summaries(summaries)
    json_path, csv_path = save_global_summary(
        summary=aggregate,
        output_root=output_root,
        experiment_prefix=args.experiment_prefix,
        checkpoint=str(checkpoint),
        config=str(config),
    )

    print("\n" + "=" * 72)
    print("ESC-50 — RISULTATO AGGREGATO")
    print("=" * 72)
    for item in aggregate["folds"]:
        print(
            f"Fold {int(item['fold'])}: "
            f"{100.0 * float(item['best_val_accuracy']):.2f}% "
            f"(best epoch {int(item['best_epoch'])})"
        )

    print(
        "\nMedia ± std (population): "
        f"{100.0 * aggregate['mean_accuracy']:.2f}% ± "
        f"{100.0 * aggregate['std_population_accuracy']:.2f}%"
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
