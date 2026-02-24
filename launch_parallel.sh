#!/bin/bash
# launch_parallel.sh
# ══════════════════════════════════════════════════════════════════════════════
# 8 GPUs (0-7), ~88 images chacun = 700 images PIE-Bench
# CUDA_VISIBLE_DEVICES setté dans le shell avant Python (seule méthode fiable)
#
# Usage:
#   chmod +x launch_parallel.sh && ./launch_parallel.sh
#   ./launch_parallel.sh --test    # 2 images par GPU
# ══════════════════════════════════════════════════════════════════════════════

GPUS=(1 2 3 4 7)  # GPUs 5 et 6 occupés par zhou
NUM_CHUNKS=5
EXTRA_ARGS=""

if [[ "$1" == "--test" ]]; then
    EXTRA_ARGS="--test_run"
    echo "=== TEST MODE (2 images par GPU) ==="
fi

LOG_DIR="logs/parallel_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Lancement sur GPUs : ${GPUS[*]}"
echo "Logs : $LOG_DIR"
echo ""

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    LOG_FILE="$LOG_DIR/gpu${GPU_ID}.log"
    echo "  CUDA_VISIBLE_DEVICES=$GPU_ID  chunk $i/$NUM_CHUNKS → $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python benchmark_parallel.py \
        --gpu_id     $GPU_ID \
        --num_chunks $NUM_CHUNKS \
        --chunk_idx  $i \
        $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    sleep 2   # petit délai pour éviter conflits de chargement simultané
done

echo ""
echo "PIDs : ${PIDS[*]}"
echo ""
echo "Suivre en direct :"
echo "  tail -f $LOG_DIR/gpu1.log"
echo "  tail -f $LOG_DIR/gpu*.log"
echo ""

# ── attendre tous ─────────────────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    CODE=$?
    GPU_ID=${GPUS[$i]}
    if [ $CODE -ne 0 ]; then
        echo "ERREUR GPU $GPU_ID (code $CODE) — $LOG_DIR/gpu${GPU_ID}.log"
        FAILED=1
    else
        echo "GPU $GPU_ID : OK"
    fi
done

echo ""
echo "=== Fusion des résultats ==="
python merge_results.py --output_dir outputs/PIE_BENCH_PARALLEL