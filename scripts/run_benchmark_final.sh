#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# MAE-AST — Final compute/memory benchmark
#
# Confronto:
#   MAE-AST vs FullSequenceProxy
#
# Fattori:
#   encoder layers = {6, 12}
#   mask ratio     = {0.50, 0.75}
#
# Protocollo finale:
#   batch size = 32
#   warm-up    = 20 step
#   misura     = 100 step
#   repliche   = 5
#
# Il benchmark usa batch sintetici residenti in memoria:
# misura quindi il costo per-step del modello, non il wall-clock
# end-to-end del pretraining.
#
# Opzionali:
#   OUTPUT_DIR=/path/to/benchmarks
#   BATCH_SIZE=32
#   WARMUP=20
#   STEPS=100
#   REPS=5
#
# La GPU viene scelta esternamente, ad esempio:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_benchmark_final.sh
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG6="$PROJECT_ROOT/configs/experiments/pretrain_final_6l.yaml"
CONFIG12="$PROJECT_ROOT/configs/experiments/pretrain_final_12l.yaml"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/benchmarks_final}"

BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-20}"
STEPS="${STEPS:-100}"
REPS="${REPS:-5}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

for FILE in "$CONFIG6" "$CONFIG12"; do
    if [ ! -f "$FILE" ]; then
        echo "ERRORE: config richiesta non trovata:"
        echo "  $FILE"
        exit 1
    fi
done


run_one () {
    local LAYERS="$1"
    local MASK="$2"
    local REP="$3"

    local CONFIG
    local MASK_TAG

    case "$LAYERS" in
        6)
            CONFIG="$CONFIG6"
            ;;
        12)
            CONFIG="$CONFIG12"
            ;;
        *)
            echo "ERRORE: profondità encoder non valida: $LAYERS"
            exit 1
            ;;
    esac

    case "$MASK" in
        0.50)
            MASK_TAG="50"
            ;;
        0.75)
            MASK_TAG="75"
            ;;
        *)
            echo "ERRORE: mask ratio non valido: $MASK"
            exit 1
            ;;
    esac

    local RUN_NAME="tableC_${LAYERS}l_mask${MASK_TAG}_rep${REP}"

    echo
    echo "======================================================================"
    echo "$RUN_NAME"
    echo "======================================================================"
    echo "Encoder:   ${LAYERS}L"
    echo "Mask:      $MASK"
    echo "Batch:     $BATCH_SIZE"
    echo "Warm-up:   $WARMUP"
    echo "Steps:     $STEPS"
    echo "Replica:   $REP/$REPS"
    echo "Output:    $OUTPUT_DIR"

    # Permette di rilanciare la campagna senza duplicare le run complete.
    if compgen -G "$OUTPUT_DIR/${RUN_NAME}_*.json" > /dev/null; then
        echo "[SKIP] risultato già presente."
        return
    fi

    PYTHONUNBUFFERED=1 \
    python -u scripts/benchmark.py \
        --config "$CONFIG" \
        --batch-size "$BATCH_SIZE" \
        --warmup "$WARMUP" \
        --steps "$STEPS" \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$RUN_NAME" \
        --set \
            masking.ratio="$MASK"

    if ! compgen -G "$OUTPUT_DIR/${RUN_NAME}_*.json" > /dev/null; then
        echo "ERRORE: benchmark completato ma JSON non trovato:"
        echo "  $RUN_NAME"
        exit 1
    fi
}


echo
echo "######################################################################"
echo "# MAE-AST — FINAL COMPUTE/MEMORY BENCHMARK"
echo "######################################################################"
echo "Output:   $OUTPUT_DIR"
echo "Batch:    $BATCH_SIZE"
echo "Warm-up:  $WARMUP"
echo "Steps:    $STEPS"
echo "Repliche: $REPS"

for ((REP=1; REP<=REPS; REP++)); do
    run_one 6  0.50 "$REP"
    run_one 6  0.75 "$REP"
    run_one 12 0.50 "$REP"
    run_one 12 0.75 "$REP"
done

echo
echo "######################################################################"
echo "# BENCHMARK COMPLETATO"
echo "######################################################################"
echo "Output:"
echo "  $OUTPUT_DIR"
