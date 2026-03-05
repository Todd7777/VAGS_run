#!/bin/bash
# run_local.sh
# ══════════════════════════════════════════════════════════════════════════════
# Local runner for VAGS on a machine with GPUs already set up.
# Assumes the 'image_editing' conda environment already exists with all
# dependencies installed (colleague's pre-checked setup).
#
# Usage:
#   chmod +x run_local.sh
#   ./run_local.sh              # full PIE-Bench, all 4 proposed methods
#   ./run_local.sh --test       # smoke test, 4 pairs per method
#   ./run_local.sh --flowedit   # run only FlowEdit methods (GPUs 0-1)
#   ./run_local.sh --splitflow  # run only SplitFlow methods (GPUs 2-3)
#   ./run_local.sh --gpu_map 0 1 2 3  # override GPU assignment
# ══════════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse args ────────────────────────────────────────────────────────────────
MODE="full"
MAX_PAIRS=""
GPU_MAP=""
METHODS=""

for arg in "$@"; do
    case $arg in
        --test)       MODE="test"; MAX_PAIRS="--max_pairs 4" ;;
        --flowedit)   METHODS="--methods 0 1"; GPU_MAP="--gpu_map 0 1" ;;
        --splitflow)  METHODS="--methods 2 3"; GPU_MAP="--gpu_map 0 1" ;;
        --gpu_map)    shift ;;  # handled below
        *) ;;
    esac
done

# ── Activate conda env ────────────────────────────────────────────────────────
# Source conda from common install paths
for CONDA_SH in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/shared/ssd_14T/home/ahmadou/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        break
    fi
done

conda activate image_editing

# ── Verify GPU visibility ─────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo " VAGS Local Run"
echo "══════════════════════════════════════════════"
echo ""

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" == "0" ]; then
    echo "ERROR: No CUDA GPUs detected. Check drivers and environment."
    exit 1
fi

echo "GPUs available: $GPU_COUNT"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}  ({props.total_memory // 1024**3} GB)')
"
echo ""

# ── Set memory allocator for stability ───────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

if [ "$MODE" == "test" ]; then
    echo "MODE: TEST (4 pairs per method)"
    echo ""
    python benchmark_all_methods.py \
        --pie_bench \
        --max_pairs 4 \
        $METHODS \
        $GPU_MAP
else
    echo "MODE: FULL PIE-Bench (700 pairs)"
    echo ""
    python benchmark_all_methods.py \
        --pie_bench \
        $METHODS \
        $GPU_MAP
fi

echo ""
echo "══════════════════════════════════════════════"
echo " Done. Results saved to outputs/"
echo "══════════════════════════════════════════════"
