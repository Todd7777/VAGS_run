"""
modal_app.py
════════════════════════════════════════════════════════════════════════════════
VAGS benchmark runner for Modal Labs.

Architecture
────────────
  • One Modal Function per method → each gets its own H200 GPU container
  • modal.Volume  "vags-models"   → persistent HuggingFace model cache
  • modal.Volume  "vags-outputs"  → persistent output CSVs + images
  • The repo code is mounted read-only via modal.Mount

Usage (from your Mac terminal, inside the VAGS directory)
────────────────────────────────────────────────────────
  pip install modal
  modal setup               # authenticate once
  modal run modal_app.py    # run all 4 methods in parallel (PIE-Bench)
  modal run modal_app.py::run_test   # quick 4-image smoke test
  modal run modal_app.py::download_models  # pre-warm model cache

Environment variables required (set as Modal secrets)
──────────────────────────────────────────────────────
  HF_TOKEN  → your HuggingFace token (needed for SD3.5-large + SD3-medium)

  modal secret create huggingface HF_TOKEN=hf_your_token_here

GPU / cost notes
────────────────
  Each method spawns 1 × H200.  4 methods = 4 GPUs in parallel.
  Estimated wall-clock: ~1-2 h for full PIE-Bench (700 pairs) — H200 is ~2x faster than A100.
  Modal bills per second of GPU usage, not per hour like cloud VMs.
════════════════════════════════════════════════════════════════════════════════
"""

import modal
from pathlib import Path

# ── Volumes (persistent across runs) ─────────────────────────────────────────
model_vol   = modal.Volume.from_name("vags-models",  create_if_missing=True)
output_vol  = modal.Volume.from_name("vags-outputs", create_if_missing=True)

MODELS_PATH  = Path("/models")
OUTPUTS_PATH = Path("/outputs")
CODE_PATH    = Path("/vags")

# ── Container image ───────────────────────────────────────────────────────────
# Built once and cached by Modal — subsequent runs start instantly.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "libglib2.0-0", "libsm6",
        "libxrender1", "libxext6", "libgl1",
    )
    .pip_install(
        # Core ML
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # Diffusion / HF stack
        "diffusers==0.35.2",
        "transformers==4.57.1",
        "accelerate==1.0.1",
        "huggingface-hub==0.36.0",
        "safetensors==0.6.2",
        "tokenizers==0.22.1",
        "sentencepiece==0.2.1",
        "peft==0.12.0",
        # Metrics
        "lpips==0.1.4",
        "clean-fid==0.1.35",
        "dreamsim==0.2.1",
        "open-clip-torch==3.3.0",
        "clip @ git+https://github.com/openai/CLIP.git",
        # Utils
        "einops==0.8.2",
        "jaxtyping==0.3.9",
        "omegaconf==2.3.0",
        "pytorch-lightning==2.4.0",
        "wandb==0.19.0",
        "tqdm",
        "pyyaml",
        "scikit-image==0.24.0",
        "scipy==1.13.1",
        "opencv-python-headless==4.10.0.84",
        "pillow",
        "numpy",
        "invisible-watermark",
        "timm==1.0.3",
        "monai==1.3.2",
        "psutil",
        "packaging",
        "matplotlib",
        "ftfy",
        "regex",
    )
)

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("vags-benchmark", image=image)

# ── Code mount (read-only, ships your local VAGS directory to the container) ──
vags_mount = modal.Mount.from_local_dir(
    local_path=".",
    remote_path=str(CODE_PATH),
    condition=lambda path: not any(
        skip in path for skip in [
            "__pycache__", ".git", "venv", ".venv", "env",
            "outputs/", "results/", "logs/", ".DS_Store",
            "Data/Images/", "Data/flowedit_data/",
            "models/",
        ]
    ),
)

# ── Shared bootstrap helper (runs inside every container) ────────────────────

def _bootstrap():
    """Set paths and HF cache dir inside the container."""
    import sys, os
    sys.path.insert(0, str(CODE_PATH))
    sys.path.insert(0, str(CODE_PATH / "methods" / "SplitFlow"))
    sys.path.insert(0, str(CODE_PATH / "methods" / "FlowEdit"))
    sys.path.insert(0, str(CODE_PATH / "methods" / "PnPInversion"))
    os.environ["HF_HOME"]             = str(MODELS_PATH)
    os.environ["TRANSFORMERS_CACHE"]  = str(MODELS_PATH)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_PATH)
    # Single GPU per container — always cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ═════════════════════════════════════════════════════════════════════════════
# Utility: pre-download model weights into the persistent volume
# Run once:  modal run modal_app.py::download_models
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="H200",
    timeout=60 * 60 * 2,           # 2 h — first download can be slow
    volumes={
        str(MODELS_PATH): model_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    mounts=[vags_mount],
)
def download_models():
    import os, torch
    from diffusers import StableDiffusion3Pipeline

    _bootstrap()
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    models = [
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3-medium-diffusers",
    ]
    for m in models:
        print(f"Downloading {m} ...")
        StableDiffusion3Pipeline.from_pretrained(m, torch_dtype=torch.float16)
        print(f"  {m} — OK")

    model_vol.commit()
    print("All models downloaded and committed to volume.")


# ═════════════════════════════════════════════════════════════════════════════
# Core runner — one GPU container per method
# ═════════════════════════════════════════════════════════════════════════════

def _run_method(method_name: str, max_pairs: int | None = None):
    """
    Load pairs, run one method, save CSVs + images to output volume.
    Runs inside a container — all imports happen here.
    """
    import sys, os, json
    from pathlib import Path as P
    from datetime import datetime

    _bootstrap()

    # Late import so the module resolves CODE_PATH correctly inside container
    sys.argv = ["benchmark_all_methods.py"]   # avoid argparse confusion
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark_all_methods",
        str(CODE_PATH / "benchmark_all_methods.py"),
    )
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)

    # Load PIE-Bench pairs
    mapping_file = str(CODE_PATH / "Data" / "PIE-Bench_v1" / "mapping_file.json")
    images_root  = str(CODE_PATH / "Data" / "PIE-Bench_v1" / "annotation_images")
    pairs = bm.load_pairs_pie(mapping_file, images_root, max_pairs)
    print(f"[{method_name}] Loaded {len(pairs)} pairs")

    # Output directory on the persistent volume
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUTS_PATH / f"{method_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map method name → runner function
    runner_map = {
        "flowedit_sd35_conflictaware_cosine":    bm.run_flowedit_sd35_conflictaware_cosine,
        "flowedit_sd35_conflictaware_relative":  bm.run_flowedit_sd35_conflictaware_relative,
        "splitflow_sd35_conflictaware_cosine":   bm.run_splitflow_sd35_conflictaware_cosine,
        "splitflow_sd35_conflictaware_relative": bm.run_splitflow_sd35_conflictaware_relative,
    }

    fn = runner_map[method_name]
    print(f"[{method_name}] Starting on cuda:0 ...")
    fn(pairs, gpu_id=0, out_dir=out_dir)
    print(f"[{method_name}] Done. Results in {out_dir}")

    # Commit results to the persistent volume so they survive container shutdown
    output_vol.commit()


# ── Individual GPU functions (one per method) ─────────────────────────────────

@app.function(
    gpu="H200",
    timeout=60 * 60 * 6,
    volumes={
        str(MODELS_PATH): model_vol,
        str(OUTPUTS_PATH): output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    mounts=[vags_mount],
    memory=98304,
)
def run_flowedit_cosine(max_pairs: int | None = None):
    _run_method("flowedit_sd35_conflictaware_cosine", max_pairs)


@app.function(
    gpu="H200",
    timeout=60 * 60 * 6,
    volumes={
        str(MODELS_PATH): model_vol,
        str(OUTPUTS_PATH): output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    mounts=[vags_mount],
    memory=98304,
)
def run_flowedit_relative(max_pairs: int | None = None):
    _run_method("flowedit_sd35_conflictaware_relative", max_pairs)


@app.function(
    gpu="H200",
    timeout=60 * 60 * 6,
    volumes={
        str(MODELS_PATH): model_vol,
        str(OUTPUTS_PATH): output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    mounts=[vags_mount],
    memory=98304,
)
def run_splitflow_cosine(max_pairs: int | None = None):
    _run_method("splitflow_sd35_conflictaware_cosine", max_pairs)


@app.function(
    gpu="H200",
    timeout=60 * 60 * 6,
    volumes={
        str(MODELS_PATH): model_vol,
        str(OUTPUTS_PATH): output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    mounts=[vags_mount],
    memory=98304,
)
def run_splitflow_relative(max_pairs: int | None = None):
    _run_method("splitflow_sd35_conflictaware_relative", max_pairs)


# ═════════════════════════════════════════════════════════════════════════════
# Entry points — called with `modal run modal_app.py`
# ═════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    """
    Run all 4 proposed methods in parallel, each on its own H200.
    modal run modal_app.py
    """
    print("Launching all 4 methods in parallel on H200 GPUs...")
    fns = [
        run_flowedit_cosine,
        run_flowedit_relative,
        run_splitflow_cosine,
        run_splitflow_relative,
    ]
    # .spawn() submits all 4 to Modal simultaneously (true parallelism)
    handles = [fn.spawn() for fn in fns]
    print(f"All 4 jobs submitted. Waiting for results...")
    for h in handles:
        h.get()
    print("All methods complete. Download results with:")
    print("  modal volume get vags-outputs /")


@app.local_entrypoint()
def run_test():
    """
    Smoke test: 4 pairs per method, all 4 methods in parallel.
    modal run modal_app.py::run_test
    """
    print("TEST MODE — 4 pairs per method...")
    handles = [
        run_flowedit_cosine.spawn(max_pairs=4),
        run_flowedit_relative.spawn(max_pairs=4),
        run_splitflow_cosine.spawn(max_pairs=4),
        run_splitflow_relative.spawn(max_pairs=4),
    ]
    for h in handles:
        h.get()
    print("Test complete.")
