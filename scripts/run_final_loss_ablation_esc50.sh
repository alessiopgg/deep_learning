#!/usr/bin/env bash

set -Eeuo pipefail

# MAE-AST — Final loss ablation on ESC-50
#
# L1 e L2:
#   encoder 6L
#   ESC-50 5-fold
#   50 epoche per fold
#
# Uso:
#   ESC50_ROOT=/path/to/ESC-50-master \
#   bash scripts/run_final_loss_ablation_esc50.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="$PROJECT_ROOT/configs/experiments/esc50_finetune_final_6l.yaml"

PRE_ROOT="${PRE_ROOT:-$PROJECT_ROOT/outputs/final_loss_ablation_16ep}"
FT_ROOT="${FT_ROOT:-$PROJECT_ROOT/outputs/final_loss_ablation_esc50}"

ESC50_ROOT="${ESC50_ROOT:-}"

if [ -z "$ESC50_ROOT" ]; then
    echo "ERRORE: ESC50_ROOT non definita."
    exit 1
fi

STATS_FILE="${STATS_FILE:-$PROJECT_ROOT/datafiles/audioset500k_train_stats_padded.json}"

TRAIN_MANIFEST_TEMPLATE="$PROJECT_ROOT/datafiles/esc50_train_fold{fold}.json"
VAL_MANIFEST_TEMPLATE="$PROJECT_ROOT/datafiles/esc50_eval_fold{fold}.json"

L1_CKPT="$PRE_ROOT/L1_final_6l_chunk75_recon_only_16ep/best.pt"
L2_CKPT="$PRE_ROOT/L2_final_6l_chunk75_cls_only_16ep/best.pt"

mkdir -p "$FT_ROOT"
cd "$PROJECT_ROOT"

for FILE in \
    "$CONFIG" \
    "$L1_CKPT" \
    "$L2_CKPT" \
    "$STATS_FILE"
do
    if [ ! -f "$FILE" ]; then
        echo "ERRORE: file richiesto non trovato:"
        echo "  $FILE"
        exit 1
    fi
done

if [ ! -d "$ESC50_ROOT" ]; then
    echo "ERRORE: dataset ESC-50 non trovato:"
    echo "  $ESC50_ROOT"
    exit 1
fi

for FOLD in 1 2 3 4 5
do
    for FILE in \
        "$PROJECT_ROOT/datafiles/esc50_train_fold${FOLD}.json" \
        "$PROJECT_ROOT/datafiles/esc50_eval_fold${FOLD}.json"
    do
        if [ ! -f "$FILE" ]; then
            echo "ERRORE: manifest ESC-50 non trovato:"
            echo "  $FILE"
            exit 1
        fi
    done
done


run_esc50 () {
    local CODE="$1"
    local CKPT="$2"

    local PREFIX="${CODE}_final_16ep_5fold"

    echo
    echo "======================================================================"
    echo "$CODE — ESC-50 5-FOLD"
    echo "======================================================================"
    echo "Checkpoint: $CKPT"
    echo "Dataset:    $ESC50_ROOT"
    echo "Output:     $FT_ROOT"

    python -u scripts/run_esc50_5fold.py \
        --config "$CONFIG" \
        --dataset-root "$ESC50_ROOT" \
        --train-manifest-template "$TRAIN_MANIFEST_TEMPLATE" \
        --val-manifest-template "$VAL_MANIFEST_TEMPLATE" \
        --stats-file "$STATS_FILE" \
        --checkpoint "$CKPT" \
        --folds 1 2 3 4 5 \
        --experiment-prefix "$PREFIX" \
        --output-root "$FT_ROOT" \
        --logging-backend local \
        --skip-completed
}


run_esc50 "L1" "$L1_CKPT"
run_esc50 "L2" "$L2_CKPT"

echo
echo "LOSS ABLATION ESC-50 COMPLETATA"
echo "Output: $FT_ROOT"
