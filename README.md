# MAE-AST — Masked Autoencoding for Audio Spectrogram Transformers

PyTorch implementation and experimental study of **MAE-AST**, a masked autoencoding approach for self-supervised audio representation learning.

The project studies three aspects of the method:

1. the effect of **encoder depth, masking strategy and masking ratio**;
2. the contribution of the **reconstruction and classification objectives**;
3. the **computational efficiency** obtained by processing only visible tokens in the encoder.

Pretraining is performed on an AudioSet-derived subset, while learned representations are evaluated on **ESC-50** using the official 5-fold protocol.

---

## Experimental study

### Main ablation

Eight models are obtained from the factorial combination of:

- encoder depth: **6 / 12 layers**;
- masking: **random / chunk**;
- mask ratio: **50% / 75%**.

| Run | Encoder | Masking | Ratio | ESC-50 accuracy |
|---|---:|---|---:|---:|
| M1 | 6L | random | 50% | 82.50 ± 3.05% |
| M2 | 6L | chunk | 50% | 84.45 ± 2.77% |
| M3 | 6L | random | 75% | 83.55 ± 2.25% |
| M4 | 6L | chunk | 75% | 84.35 ± 1.46% |
| M5 | 12L | random | 50% | 84.00 ± 2.88% |
| M6 | 12L | chunk | 50% | **85.90 ± 1.49%** |
| M7 | 12L | random | 75% | 85.15 ± 3.00% |
| M8 | 12L | chunk | 75% | 84.50 ± 1.94% |

Reported values are mean ± standard deviation across the five ESC-50 folds.

### Loss ablation

Using the M4 architecture (6L, chunk masking, 75% ratio):

| Objective | ESC-50 accuracy |
|---|---:|
| Reconstruction only | 80.60 ± 1.80% |
| Classification only | **84.60 ± 0.87%** |
| Joint objective | 84.35 ± 1.46% |

The joint pretraining objective is:

\[
\mathcal{L} = 10\,\mathcal{L}_{rec} + \mathcal{L}_{cls}.
\]

### Efficiency benchmark

MAE-AST is compared with a **FullSequenceProxy**, a controlled baseline in which the encoder processes the complete token sequence.

This is a **per-step compute/memory micro-benchmark**, not an end-to-end training-time comparison.

Each configuration uses:

- batch size: 32;
- 20 warm-up steps;
- 100 measured training steps;
- 5 independent repetitions;
- synthetic batches already resident in memory.

| Encoder | Mask ratio | Speed-up | Peak memory reduction |
|---|---:|---:|---:|
| 6L | 50% | 1.50× | 26.63% |
| 6L | 75% | 1.74× | 40.65% |
| 12L | 50% | 1.67× | 32.51% |
| 12L | 75% | **2.13×** | **49.10%** |

The benchmark isolates model compute from disk access, audio decoding, preprocessing and DataLoader overhead.

---

## Repository structure

```text
.
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── pretrain_final_6l.yaml
│       ├── pretrain_final_12l.yaml
│       ├── esc50_finetune_final_6l.yaml
│       └── esc50_finetune_final_12l.yaml
├── datafiles/
│   ├── audioset500k_train_stats_padded.json
│   └── esc50_*_fold*.json
├── results/
│   ├── main_ablation/
│   ├── loss_ablation/
│   └── efficiency/
├── scripts/
│   ├── pretrain.py
│   ├── finetune.py
│   ├── benchmark.py
│   ├── run_esc50_5fold.py
│   ├── run_final_pretrain_campaign_16ep.sh
│   ├── run_final_esc50_campaign_16ep.sh
│   ├── run_final_loss_ablation_16ep.sh
│   ├── run_final_loss_ablation_esc50.sh
│   └── run_benchmark_final.sh
├── src/mae_ast/
│   ├── data/
│   ├── models/
│   └── training/
├── tools/
│   ├── prepare_audioset.py
│   ├── prepare_esc50.py
│   ├── split_audioset.py
│   └── compute_stats.py
└── pyproject.toml
```

---

## Model

Audio is converted to a log-Mel representation and divided into non-overlapping **16 × 16 spectrogram patches**.

For the pretraining input size of **1024 × 128**, this produces 512 tokens.

MAE-AST masks a fraction of these tokens before the Transformer encoder:

- 50% masking → 256 visible encoder tokens;
- 75% masking → 128 visible encoder tokens.

Only visible tokens are processed by the encoder. The decoder reconstructs the complete sequence.

The final models use:

- embedding dimension: 768;
- 12 attention heads;
- encoder depth: 6 or 12 layers;
- decoder depth: 2 layers;
- AdamW optimization;
- mixed-precision training.

---

## Setup

A recent Python environment with PyTorch and CUDA support is recommended.

Install the package from the repository root:

```bash
pip install -e .
```

Dependencies and package metadata are defined in `pyproject.toml`.

---

## Data

### AudioSet pretraining

The final experiments use an AudioSet-derived training set with:

- 478,284 training clips;
- 12,264 validation clips;
- 527 classes.

Audio is processed at 16 kHz with 128 Mel bins.

The normalization statistics used by the final experiments are included in:

```text
datafiles/audioset500k_train_stats_padded.json
```

By default, the campaign runner expects:

```text
$AUDIOSET_ROOT/
└── manifests/
    ├── audioset_500k_pretrain_train.json
    └── audioset_500k_pretrain_val.json
```

Dataset preparation utilities are available under `tools/`.

### ESC-50

The repository includes the train/evaluation manifests for all five official ESC-50 folds:

```text
datafiles/esc50_train_fold*.json
datafiles/esc50_eval_fold*.json
```

The ESC-50 audio files themselves are not distributed in this repository.

---

## Reproducing the experiments

GPU selection is intentionally left to the execution environment. For example:

```bash
CUDA_VISIBLE_DEVICES=0 ...
```

### Main pretraining campaign

```bash
AUDIOSET_ROOT=/path/to/audioset500k \
bash scripts/run_final_pretrain_campaign_16ep.sh
```

This executes the M1-M8 pretraining matrix.

The final training horizon is **239,136 optimization steps**, corresponding to the experimental 16-epoch budget.

### ESC-50 evaluation

```bash
ESC50_ROOT=/path/to/ESC-50-master \
bash scripts/run_final_esc50_campaign_16ep.sh
```

Each pretrained model is evaluated over the five official ESC-50 folds.

Alternative pretraining/output locations can be supplied through the environment variables documented in the runner.

### Loss ablation

```bash
AUDIOSET_ROOT=/path/to/audioset500k \
bash scripts/run_final_loss_ablation_16ep.sh
```

followed by:

```bash
ESC50_ROOT=/path/to/ESC-50-master \
bash scripts/run_final_loss_ablation_esc50.sh
```

### Efficiency benchmark

```bash
bash scripts/run_benchmark_final.sh
```

Optional benchmark parameters can be changed through:

```text
BATCH_SIZE
WARMUP
STEPS
REPS
OUTPUT_DIR
```

The published results use the defaults encoded in the runner.

---

## Results and provenance

`results/` contains the artifacts used to derive the reported experimental results:

```text
results/
├── main_ablation/
│   ├── pretrain/
│   └── esc50/
├── loss_ablation/
│   ├── pretrain/
│   └── esc50/
└── efficiency/
```

For the training experiments, the repository retains:

- effective configuration snapshots;
- epoch-level training logs;
- ESC-50 fold summaries;
- aggregate CSV/JSON results.

For the efficiency study, all individual benchmark repetitions and aggregate tables are retained.

Model checkpoints are intentionally excluded because of their size.

### Configuration provenance

The YAML files under `configs/experiments/` are the **portable configurations intended for reproducing the experiments**.

The `config.yaml` files stored under `results/` are instead immutable snapshots of the original experimental runs. They are preserved for provenance and may therefore contain paths from the machine on which the original experiments were executed.

Those historical paths are not required by the public runners.

---

## Notes on reproducibility

The project separates:

- portable experiment definitions in `configs/`;
- dataset metadata in `datafiles/`;
- executable pipelines in `scripts/`;
- implementation code in `src/`;
- immutable experimental evidence in `results/`.

Training checkpoints, datasets, local machine configuration and generated caches are deliberately excluded from version control.
