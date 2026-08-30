#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# MAE-AST — Final pretraining campaign M1-M8
#
# Esperimenti:
#   depth     = {6L, 12L}
#   masking   = {random, chunk}
#   mask ratio= {0.50, 0.75}
#
# Budget finale:
#   239136 optimizer step ≈ 16 epoche su AudioSet-500k
#
# Uso:
#   AUDIOSET_ROOT=/path/to/audioset500k \
#   bash scripts/run_final_pretrain_campaign_16ep.sh
#
# Opzionali:
#   OUTPUT_ROOT=/path/to/outputs
#   TRAIN_MANIFEST=/path/to/train.json
#   VAL_MANIFEST=/path/to/val.json
#   STATS_FILE=/path/to/stats.json
#
# La selezione della GPU è lasciata all'ambiente:
#   CUDA_VISIBLE_DEVICES=1 AUDIOSET_ROOT=... bash ...
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE6="$PROJECT_ROOT/configs/experiments/pretrain_final_6l.yaml"
BASE12="$PROJECT_ROOT/configs/experiments/pretrain_final_12l.yaml"

AUDIOSET_ROOT="${AUDIOSET_ROOT:-}"

if [ -z "$AUDIOSET_ROOT" ]; then
    echo "ERRORE: AUDIOSET_ROOT non definita."
    echo
    echo "Esempio:"
    echo "  AUDIOSET_ROOT=/path/to/audioset500k \\"
    echo "  bash scripts/run_final_pretrain_campaign_16ep.sh"
    exit 1
fi

TRAIN_MANIFEST="${TRAIN_MANIFEST:-$AUDIOSET_ROOT/manifests/audioset_500k_pretrain_train.json}"
VAL_MANIFEST="${VAL_MANIFEST:-$AUDIOSET_ROOT/manifests/audioset_500k_pretrain_val.json}"

STATS_FILE="${STATS_FILE:-$PROJECT_ROOT/datafiles/audioset500k_train_stats_padded.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/final_pretrain_16ep}"

TARGET_STEPS=239136

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"


# ============================================================
# Controlli preliminari
# ============================================================

for FILE in \
    "$BASE6" \
    "$BASE12" \
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


# ============================================================
# Verifica completamento run
# ============================================================

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


# ============================================================
# Singolo esperimento
# ============================================================

run_pretrain () {
    local CODE="$1"
    local NAME="$2"
    local BASE="$3"
    local STRATEGY="$4"
    local RATIO="$5"

    local DIR="$OUTPUT_ROOT/$NAME"
    local LAST="$DIR/last.pt"

    echo
    echo "======================================================================"
    echo "$CODE — $NAME"
    echo "======================================================================"
    echo "Masking: $STRATEGY"
    echo "Ratio:   $RATIO"
    echo "Steps:   $TARGET_STEPS"
    echo "Output:  $DIR"
    echo "======================================================================"

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

        "masking.strategy=$STRATEGY"
        "masking.ratio=$RATIO"

        "training.max_steps=$TARGET_STEPS"

        "logging.wandb_dir=$OUTPUT_ROOT/wandb"

        "checkpoint.output_dir=$OUTPUT_ROOT"
    )

    if [ -f "$LAST" ]; then
        echo "Resume da:"
        echo "  $LAST"

        python -u scripts/pretrain.py \
            --config "$BASE" \
            --resume "$LAST" \
            --set "${OVERRIDES[@]}"

    else
        if [ -d "$DIR" ] && \
           [ -n "$(find "$DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then

            echo "ERRORE: directory esistente ma last.pt assente:"
            echo "  $DIR"
            echo "Nessun file viene cancellato automaticamente."
            exit 1
        fi

        python -u scripts/pretrain.py \
            --config "$BASE" \
            --set "${OVERRIDES[@]}"
    fi

    if ! run_complete "$DIR"; then
        echo "ERRORE: $CODE terminata ma il run non risulta completo."
        exit 1
    fi

    echo "$CODE completata."
}


# ============================================================
# Campagna M1-M8
# ============================================================

echo
echo "######################################################################"
echo "# MAE-AST — FINAL PRETRAIN CAMPAIGN"
echo "######################################################################"
echo "Dataset: $AUDIOSET_ROOT"
echo "Output:  $OUTPUT_ROOT"
echo

run_pretrain \
    "M1" \
    "M1_final_6l_random50_16ep" \
    "$BASE6" \
    "random" \
    "0.50"

run_pretrain \
    "M2" \
    "M2_final_6l_chunk50_16ep" \
    "$BASE6" \
    "chunk" \
    "0.50"

# Nome mantenuto per corrispondenza con i risultati pubblicati.
run_pretrain \
    "M3" \
    "M3_corrected_6l_random75_adamw_16ep" \
    "$BASE6" \
    "random" \
    "0.75"

run_pretrain \
    "M4" \
    "M4_final_6l_chunk75_16ep" \
    "$BASE6" \
    "chunk" \
    "0.75"

run_pretrain \
    "M5" \
    "M5_final_12l_random50_16ep" \
    "$BASE12" \
    "random" \
    "0.50"

run_pretrain \
    "M6" \
    "M6_final_12l_chunk50_16ep" \
    "$BASE12" \
    "chunk" \
    "0.50"

run_pretrain \
    "M7" \
    "M7_final_12l_random75_16ep" \
    "$BASE12" \
    "random" \
    "0.75"

run_pretrain \
    "M8" \
    "M8_final_12l_chunk75_16ep" \
    "$BASE12" \
    "chunk" \
    "0.75"

echo
echo "######################################################################"
echo "# PRETRAINING M1-M8 COMPLETATO"
echo "######################################################################"
echo "Output:"
echo "  $OUTPUT_ROOT"
