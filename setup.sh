#!/bin/bash
# setup.sh
# ══════════════════════════════════════════════════════════════════════════════
# One-command setup for VAGS on Jarvis Labs (8 x H200)
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# After this completes:
#   Test run:  ./launch_parallel.sh --test
#   Full run:  ./launch_parallel.sh
#   OR
#   Benchmark (4 proposed methods, GPUs 0-3):
#     conda activate image_editing
#     python benchmark_all_methods.py --pie_bench
# ══════════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "══════════════════════════════════════════════"
echo " VAGS Setup — Jarvis Labs 8×H200"
echo "══════════════════════════════════════════════"

# ── 1. Conda environment ──────────────────────────────────────────────────────
echo ""
echo "[1/5] Setting up conda environment..."

# Source conda (handle different install locations)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found. Install miniconda first:"
    echo "  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
    echo "  source ~/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

if conda env list | grep -q "^image_editing"; then
    echo "  Environment 'image_editing' already exists — skipping creation."
else
    echo "  Creating environment 'image_editing' (Python 3.11)..."
    conda create -n image_editing python=3.11 -y
fi

conda activate image_editing

# ── 2. PyTorch + CUDA ─────────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing PyTorch (CUDA 12.4)..."

pip install --quiet \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify GPU is visible
python -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs detected: {n}')
for i in range(n):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
if n == 0:
    raise RuntimeError('No GPUs detected — check CUDA drivers.')
"

# ── 3. Project dependencies ───────────────────────────────────────────────────
echo ""
echo "[3/5] Installing project dependencies..."

pip install --quiet \
    accelerate==1.0.1 \
    diffusers==0.35.2 \
    transformers==4.57.1 \
    huggingface-hub==0.36.0 \
    safetensors==0.6.2 \
    tokenizers==0.22.1 \
    sentencepiece==0.2.1 \
    peft==0.12.0 \
    lpips==0.1.4 \
    clean-fid==0.1.35 \
    dreamsim==0.2.1 \
    open-clip-torch==3.3.0 \
    clip \
    einops==0.8.2 \
    jaxtyping==0.3.9 \
    omegaconf==2.3.0 \
    pytorch-lightning==2.4.0 \
    wandb==0.19.0 \
    tqdm \
    pyyaml \
    scikit-image==0.24.0 \
    scipy==1.13.1 \
    opencv-python==4.10.0.84 \
    pillow \
    numpy \
    invisible-watermark \
    timm==1.0.3 \
    monai==1.3.2 \
    psutil \
    packaging \
    matplotlib

echo "  Dependencies installed."

# ── 4. HuggingFace login ──────────────────────────────────────────────────────
echo ""
echo "[4/5] HuggingFace authentication..."

if python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "  Already logged in to HuggingFace."
else
    echo "  You need a HuggingFace token to download FLUX.1-dev and SD3.5-large."
    echo "  Get yours at: https://huggingface.co/settings/tokens"
    echo ""
    huggingface-cli login
fi

# ── 5. Pre-download model weights ─────────────────────────────────────────────
echo ""
echo "[5/5] Pre-downloading model weights (this takes ~10-20 min on first run)..."

# Store cache on persistent disk if available, otherwise home dir
if [ -d "/workspace" ]; then
    export HF_HOME="/workspace/.cache/huggingface"
    echo "  Using /workspace for HF cache (persistent disk)."
else
    export HF_HOME="$HOME/.cache/huggingface"
    echo "  Using $HOME/.cache/huggingface for HF cache."
fi

python - <<'EOF'
import torch
from diffusers import StableDiffusion3Pipeline

print("  Downloading stabilityai/stable-diffusion-3.5-large ...")
StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.float16
)
print("  SD3.5-large: OK")

print("  Downloading stabilityai/stable-diffusion-3-medium-diffusers ...")
StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
print("  SD3-medium: OK")
EOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo " Setup complete!"
echo "══════════════════════════════════════════════"
echo ""
echo " Next steps:"
echo ""
echo "  conda activate image_editing"
echo ""
echo "  # Quick test (2 images per GPU, 8 chunks):"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./launch_parallel.sh --test"
echo ""
echo "  # Full PIE-Bench parallel run (all 8 H200s):"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./launch_parallel.sh"
echo ""
echo "  # Benchmark proposed methods (GPUs 0-3):"
echo "  python benchmark_all_methods.py --pie_bench"
echo ""
echo "  # Monitor:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f logs/parallel_run_*/gpu*.log"
echo ""
