#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# MAE-AST — Final ESC-50 campaign M1-M8
#
# Protocollo:
#   - checkpoint M1-M8 dopo il pretraining finale
#   - ESC-50
#   - 5 fold ufficiali
#   - 50 epoche per fold
#
# Uso:
#   ESC50_ROOT=/path/to/ESC-50-master \
#   bash scripts/run_final_esc50_campaign_16ep.sh
#
# Opzionali:
#   PRE_ROOT=/path/to/pretraining_outputs
#   FT_ROOT=/path/to/finetuning_outputs
#   STATS_FILE=/path/to/stats.json
#
# La GPU viene scelta esternamente, per esempio:
#   CUDA_VISIBLE_DEVICES=1 ESC50_ROOT=... bash ...
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG6="$PROJECT_ROOT/configs/experiments/esc50_finetune_final_6l.yaml"
CONFIG12="$PROJECT_ROOT/configs/experiments/esc50_finetune_final_12l.yaml"

PRE_ROOT="${PRE_ROOT:-$PROJECT_ROOT/outputs/final_pretrain_16ep}"
FT_ROOT="${FT_ROOT:-$PROJECT_ROOT/outputs/final_esc50_16ep}"

ESC50_ROOT="${ESC50_ROOT:-}"

STATS_FILE="${STATS_FILE:-$PROJECT_ROOT/datafiles/audioset500k_train_stats_padded.json}"

TRAIN_MANIFEST_TEMPLATE="$PROJECT_ROOT/datafiles/esc50_train_fold{fold}.json"
VAL_MANIFEST_TEMPLATE="$PROJECT_ROOT/datafiles/esc50_eval_fold{fold}.json"

if [ -z "$ESC50_ROOT" ]; then
    echo "ERRORE: ESC50_ROOT non definita."
    echo
    echo "Esempio:"
    echo "  ESC50_ROOT=/path/to/ESC-50-master \\"
    echo "  bash scripts/run_final_esc50_campaign_16ep.sh"
    exit 1
fi

mkdir -p "$FT_ROOT"
cd "$PROJECT_ROOT"


# ============================================================
# Controlli preliminari
# ============================================================

for FILE in \
    "$CONFIG6" \
    "$CONFIG12" \
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


# ============================================================
# Singolo modello M1-M8
# ============================================================

run_esc50 () {
    local CODE="$1"
    local DEPTH="$2"
    local CKPT_NAME="$3"
    local PREFIX="$4"

    local CONFIG
    local CKPT="$PRE_ROOT/$CKPT_NAME/best.pt"

    case "$DEPTH" in
        6)
            CONFIG="$CONFIG6"
            ;;
        12)
            CONFIG="$CONFIG12"
            ;;
        *)
            echo "ERRORE: profondità encoder non valida: $DEPTH"
            exit 1
            ;;
    esac

    if [ ! -f "$CKPT" ]; then
        echo "ERRORE: checkpoint $CODE non trovato:"
        echo "  $CKPT"
        exit 1
    fi

    echo
    echo "======================================================================"
    echo "$CODE — ESC-50 5-FOLD"
    echo "======================================================================"
    echo "Encoder:    ${DEPTH}L"
    echo "Checkpoint: $CKPT"
    echo "Config:     $CONFIG"
    echo "Dataset:    $ESC50_ROOT"
    echo "Output:     $FT_ROOT"
    echo "======================================================================"

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

    echo
    echo "$CODE — 5 fold completati."
}


# ============================================================
# Campagna M1-M8
# ============================================================

echo
echo "######################################################################"
echo "# MAE-AST — FINAL ESC-50 CAMPAIGN"
echo "######################################################################"
echo "Pretraining: $PRE_ROOT"
echo "Dataset:     $ESC50_ROOT"
echo "Output:      $FT_ROOT"
echo

run_esc50 \
    "M1" \
    "6" \
    "M1_final_6l_random50_16ep" \
    "M1_final_16ep_5fold"

run_esc50 \
    "M2" \
    "6" \
    "M2_final_6l_chunk50_16ep" \
    "M2_final_16ep_5fold"

# I nomi M3 mantengono "corrected" per corrispondere
# alla provenance degli esperimenti finali pubblicati.
run_esc50 \
    "M3" \
    "6" \
    "M3_corrected_6l_random75_adamw_16ep" \
    "M3_corrected_16ep_5fold"

run_esc50 \
    "M4" \
    "6" \
    "M4_final_6l_chunk75_16ep" \
    "M4_final_16ep_5fold"

run_esc50 \
    "M5" \
    "12" \
    "M5_final_12l_random50_16ep" \
    "M5_final_16ep_5fold"

run_esc50 \
    "M6" \
    "12" \
    "M6_final_12l_chunk50_16ep" \
    "M6_final_16ep_5fold"

run_esc50 \
    "M7" \
    "12" \
    "M7_final_12l_random75_16ep" \
    "M7_final_16ep_5fold"

run_esc50 \
    "M8" \
    "12" \
    "M8_final_12l_chunk75_16ep" \
    "M8_final_16ep_5fold"

echo
echo "######################################################################"
echo "# ESC-50 M1-M8 COMPLETATO"
echo "######################################################################"
echo "Output:"
echo "  $FT_ROOT"
