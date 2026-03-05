# VAGS — Modal Labs (H200) + Local Run Guide

Run all 4 proposed benchmark methods in parallel on H200 GPUs via Modal, or locally on any multi-GPU machine.

## One-Time Setup (on your Mac)

### 1. Install Modal
```bash
pip install modal
```

### 2. Authenticate
```bash
modal setup
```
Opens a browser — log in with your Modal account.

### 3. Create the HuggingFace secret
You need a HF token to download SD3.5-large (gated model).
Get it at: https://huggingface.co/settings/tokens

```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

### 4. Pre-download model weights (run once, ~20 min)
This fills the persistent `vags-models` volume so every subsequent run starts instantly.
```bash
cd /Users/ttt/Downloads/VAGS
modal run modal_app.py::download_models
```

---

## Running the Benchmark

### Smoke test (4 pairs per method, ~10 min)
```bash
modal run modal_app.py::run_test
```

### Full PIE-Bench run (700 pairs, all 4 methods in parallel, ~2-4 h)
```bash
modal run modal_app.py
```

This spawns **4 H200 containers simultaneously**:
| Container | Method | GPU |
|-----------|--------|-----|
| 0 | `flowedit_sd35_conflictaware_cosine` | H200 |
| 1 | `flowedit_sd35_conflictaware_relative` | H200 |
| 2 | `splitflow_sd35_conflictaware_cosine` | H200 |
| 3 | `splitflow_sd35_conflictaware_relative` | H200 |

---

## Downloading Results

Results are saved to the persistent `vags-outputs` volume.

### Option A — Modal CLI
```bash
modal volume get vags-outputs / ./modal_results
```

### Option B — Python script
```bash
python modal_download_results.py
python modal_download_results.py --dest ./my_results   # custom path
```

---

## Monitoring

```bash
# Watch live logs in your terminal (during a run)
modal run modal_app.py   # logs stream directly to terminal

# Or view in the Modal dashboard
# https://modal.com/apps  →  vags-benchmark
```

---

## Cost Estimate

| Item | Rate | Est. Full Run |
|------|------|--------------|
| H200 | ~$5.25/hr | 4 GPUs × 1.5 h = **~$32** |
| Container overhead | minimal | ~$1 |
| **Total** | | **~$33 per full PIE-Bench run** |

Modal bills per second — H200 is ~2× faster than A100-80GB so despite the higher per-hour rate, total cost is lower.

---

## Running Locally (Colleague's Pre-Checked Setup)

If the `image_editing` conda environment is already set up with all dependencies:

```bash
chmod +x run_local.sh

# Smoke test — 4 pairs per method
./run_local.sh --test

# Full PIE-Bench run (all 4 methods, GPUs 0-3)
./run_local.sh

# Run only FlowEdit methods (GPUs 0-1)
./run_local.sh --flowedit

# Run only SplitFlow methods (GPUs 0-1)
./run_local.sh --splitflow
```

Or call `benchmark_all_methods.py` directly with full control:
```bash
conda activate image_editing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# All 4 proposed methods, PIE-Bench, GPUs 0-3
python benchmark_all_methods.py --pie_bench

# Test with 4 pairs only
python benchmark_all_methods.py --pie_bench --max_pairs 4

# Specific methods on specific GPUs
python benchmark_all_methods.py --pie_bench --methods 0 1 --gpu_map 0 1

# Monitor GPU usage in another terminal
watch -n 3 nvidia-smi
```

---

## Persistent Volumes

| Volume | Contents | Lives |
|--------|----------|-------|
| `vags-models` | HF model weights (~50 GB) | Until manually deleted |
| `vags-outputs` | CSVs, output images | Until manually deleted |

To list volume contents:
```bash
modal volume ls vags-models
modal volume ls vags-outputs
```

To delete volumes (careful — permanent):
```bash
modal volume delete vags-models
modal volume delete vags-outputs
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `modal: command not found` | `pip install modal` |
| `Secret 'huggingface' not found` | `modal secret create huggingface HF_TOKEN=hf_...` |
| `401 Unauthorized` on model download | Re-run `modal run modal_app.py::download_models` after fixing token |
| `OutOfMemoryError` | Already using 80GB A100 — shouldn't happen with SD3.5 at float16 |
| Job times out (8 h limit) | Reduce `max_pairs` or split PIE-Bench into chunks |
| Results not showing | Run `modal volume ls vags-outputs` to confirm commit succeeded |
