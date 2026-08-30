#!/usr/bin/env bash

set -Eeuo pipefail

# MAE-AST — Final loss ablation
#
# L1: reconstruction-only   10 * L_rec + 0 * L_cls
# L2: classification-only   0 * L_rec + 1 * L_cls
#
# Entrambi:
#   6L / chunk 75% / 239136 step
#
# Uso:
#   AUDIOSET_ROOT=/path/to/audioset500k \
#   bash scripts/run_final_loss_ablation_16ep.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_CFG="$PROJECT_ROOT/configs/experiments/pretrain_final_6l.yaml"

AUDIOSET_ROOT="${AUDIOSET_ROOT:-}"

if [ -z "$AUDIOSET_ROOT" ]; then
    echo "ERRORE: AUDIOSET_ROOT non definita."
    exit 1
fi

TRAIN_MANIFEST="${TRAIN_MANIFEST:-$AUDIOSET_ROOT/manifests/audioset_500k_pretrain_train.json}"
VAL_MANIFEST="${VAL_MANIFEST:-$AUDIOSET_ROOT/manifests/audioset_500k_pretrain_val.json}"
STATS_FILE="${STATS_FILE:-$PROJECT_ROOT/datafiles/audioset500k_train_stats_padded.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/final_loss_ablation_16ep}"

TARGET_STEPS=239136

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

for FILE in \
    "$BASE_CFG" \
    "$TRAIN_MANIFEST" \
    "$VAL_MANIFEST" \
    "$STATS_FILE"
do
    if [ ! -f "$FILE" ]; then
        echo "ERRORE: file richiesto non trovato:"
        echo "  $FILE"
        exit 1
    fi
done

if [ ! -d "$AUDIOSET_ROOT" ]; then
    echo "ERRORE: dataset root non trovata:"
    echo "  $AUDIOSET_ROOT"
    exit 1
fi


run_complete () {
    local DIR="$1"

    [ -f "$DIR/train_log.jsonl" ] || return 1
    [ -f "$DIR/best.pt" ] || return 1

    python - "$DIR/train_log.jsonl" "$TARGET_STEPS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])

max_step = 0

for line in path.read_text().splitlines():
    if not line.strip():
        continue

    try:
        row = json.loads(line)
    except Exception:
        continue

    max_step = max(max_step, int(row.get("global_step", 0)))

sys.exit(0 if max_step >= target else 1)
PY
}


run_ablation () {
    local CODE="$1"
    local NAME="$2"
    local REC_WEIGHT="$3"
    local CLS_WEIGHT="$4"

    local DIR="$OUTPUT_ROOT/$NAME"
    local LAST="$DIR/last.pt"

    echo
    echo "======================================================================"
    echo "$CODE — $NAME"
    echo "======================================================================"
    echo "Reconstruction weight: $REC_WEIGHT"
    echo "Classification weight: $CLS_WEIGHT"
    echo "Output:                $DIR"

    if run_complete "$DIR"; then
        echo "$CODE già completata — SKIP."
        return
    fi

    OVERRIDES=(
        "experiment.name=$NAME"

        "data.train_manifest=$TRAIN_MANIFEST"
        "data.val_manifest=$VAL_MANIFEST"
        "data.dataset_root=$AUDIOSET_ROOT"
        "data.stats_file=$STATS_FILE"

        "loss.reconstruction_weight=$REC_WEIGHT"
        "loss.classification_weight=$CLS_WEIGHT"

        "training.max_steps=$TARGET_STEPS"

        "logging.wandb_dir=$OUTPUT_ROOT/wandb"

        "checkpoint.output_dir=$OUTPUT_ROOT"
    )

    if [ -f "$LAST" ]; then
        echo "Resume da:"
        echo "  $LAST"

        python -u scripts/pretrain.py \
            --config "$BASE_CFG" \
            --resume "$LAST" \
            --set "${OVERRIDES[@]}"

    else
        if [ -d "$DIR" ] && \
           [ -n "$(find "$DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then

            echo "ERRORE: directory esistente ma last.pt assente:"
            echo "  $DIR"
            exit 1
        fi

        python -u scripts/pretrain.py \
            --config "$BASE_CFG" \
            --set "${OVERRIDES[@]}"
    fi

    if ! run_complete "$DIR"; then
        echo "ERRORE: $CODE non risulta completata."
        exit 1
    fi
}


run_ablation \
    "L1" \
    "L1_final_6l_chunk75_recon_only_16ep" \
    "10.0" \
    "0.0"

run_ablation \
    "L2" \
    "L2_final_6l_chunk75_cls_only_16ep" \
    "0.0" \
    "1.0"

echo
echo "LOSS ABLATION L1-L2 COMPLETATA"
echo "Output: $OUTPUT_ROOT"
