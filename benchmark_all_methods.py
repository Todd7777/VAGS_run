
import os, sys, csv, json, re, yaml, traceback, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve

ROOT         = Path(__file__).parent
METHODS_ROOT = ROOT / "methods"
LOG_FMT      = "[%(asctime)s %(levelname)s %(name)s] %(message)s"

# METRICS = ["CLIP_whole", "CLIP_edited", "clip_i",
#            "LPIPS_alex", "LPIPS_squeeze",
#            "MSE", "PSNR", "SSIM",
#            "StructDist_Sobel", "StructDist_DINO"]

# METRICS_PIE = METRICS

CSV_COLS = ["method", "image", "code",
            "CLIP_whole", "CLIP_edited", "clip_i",
            "LPIPS_alex", "LPIPS_squeeze",
            "MSE", "PSNR", "SSIM",
            "StructDist_Sobel", "StructDist_DINO", "file"]

CSV_COLS_PIE = ["method", "image", "code",
                "CLIP_whole", "CLIP_edited", "clip_i",
                "LPIPS_alex", "LPIPS_squeeze",
                "MSE", "PSNR", "SSIM",
                "StructDist_Sobel", "StructDist_DINO",
                "LPIPS_squeeze_unedit", "MSE_unedit", "PSNR_unedit", "SSIM_unedit",
                "file"]

METRICS = ["CLIP_whole", "CLIP_edited", "clip_i",
           "LPIPS_alex", "LPIPS_squeeze",
           "MSE", "PSNR", "SSIM",
           "StructDist_Sobel", "StructDist_DINO"]

METRICS_PIE = METRICS + ["LPIPS_squeeze_unedit", "MSE_unedit", "PSNR_unedit", "SSIM_unedit"]

def check_and_install():
    """Install missing optional dependencies quietly before importing methods."""
    import importlib, subprocess
    _REQUIRED = {
        "jaxtyping":        "jaxtyping",
        "einops":           "einops",
        "imwatermark":      "invisible-watermark",
        "omegaconf":        "omegaconf",
        "pytorch_lightning": "pytorch-lightning",
        "cleanfid":         "clean-fid",
    }
    for mod, pkg in _REQUIRED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[setup] Installing missing package: {pkg} …")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg,
                 "--break-system-packages", "--quiet"],
                check=False,
            )

check_and_install()

class MetricsEvaluator:
    """
    Unified evaluator for both Div2K (no mask) and PIE-Bench (with mask) modes.

    Both modes compute (all values pre-scaled for direct comparison):
      CLIP_whole, CLIP_edited, clip_i           – CLIP text-image and image-image (×1)
      LPIPS_alex, LPIPS_squeeze                 – two LPIPS backbones (×1e3)
      StructDist_Sobel, StructDist_DINO         – two structure distance variants (×1e3)
      MSE                                       – mean squared error (×1e4)
      PSNR                                      – peak signal-to-noise ratio (dB, ×1)
      SSIM                                      – structural similarity (×1e2)

    """

    def __init__(self, device: str, pie_bench: bool = False):
        self.device    = device
        self.pie_bench = pie_bench
        self._ready    = False

    def _load(self):
        if self._ready:
            return
        import lpips as _lpips
        from transformers import CLIPModel, CLIPProcessor
        self._lpips_alex = _lpips.LPIPS(net="alex").to(self.device).eval()
        self._clip  = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device).eval()
        self._cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        _pnp = str(Path(__file__).parent / "methods" / "PnPInversion")
        if _pnp not in sys.path:
            sys.path.insert(0, _pnp)
        from evaluation.matrics_calculator import MetricsCalculator
        self._mc = MetricsCalculator(self.device)
        self._ready = True

    # ── helpers ───────────────────────────────────────────────────────────────

    def _np512(self, img: Image.Image) -> np.ndarray:
        return np.array(img.convert("RGB").resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255.0

    def _tensor11(self, img: Image.Image) -> torch.Tensor:
        a = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1

    def clip_whole(self, img: Image.Image, prompt: str) -> float:
        self._load()
        with torch.no_grad():
            inp = self._cproc(text=[prompt], images=img,
                              return_tensors="pt", padding=True).to(self.device)
            return self._clip(**inp).logits_per_image.item() / 100.0

    def clip_edited(self, src: Image.Image, edited: Image.Image, prompt: str) -> float:
        self._load()
        s = np.array(src.convert("RGB").resize((512, 512))).astype(np.float32)
        e = np.array(edited.convert("RGB").resize((512, 512))).astype(np.float32)
        diff = np.mean(np.abs(s - e), axis=2)
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(diff)
        except Exception:
            thresh = diff.mean()
        mask = diff > thresh
        if mask.sum() < 100:
            return self.clip_whole(edited, prompt)
        ys, xs = np.where(mask)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        crop = edited.convert("RGB").resize((512, 512)).crop((x1, y1, x2+1, y2+1))
        with torch.no_grad():
            inp = self._cproc(text=[prompt], images=crop,
                              return_tensors="pt", padding=True).to(self.device)
            return self._clip(**inp).logits_per_image.item() / 100.0

    def clip_i(self, src: Image.Image, edited: Image.Image) -> float:
        self._load()
        with torch.no_grad():
            inp   = self._cproc(images=[src, edited],
                                return_tensors="pt").to(self.device)
            feats = self._clip.get_image_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return (feats[0] @ feats[1]).item()

    def mse(self, src: Image.Image, edited: Image.Image) -> float:
        s, e = self._np512(src), self._np512(edited)
        return float(np.mean((s - e) ** 2) * 1e4)

    def psnr(self, src: Image.Image, edited: Image.Image) -> float:
        s, e = self._np512(src), self._np512(edited)
        return float(peak_signal_noise_ratio(s, e, data_range=1.0))

    def ssim(self, src: Image.Image, edited: Image.Image) -> float:
        s, e = self._np512(src), self._np512(edited)
        return float(structural_similarity(s, e, data_range=1.0,
                                           channel_axis=2, win_size=11) * 1e2)

    def struct_dist_sobel(self, src: Image.Image, edited: Image.Image) -> float:
        s, e = self._np512(src), self._np512(edited)
        lum_w = np.array([0.299, 0.587, 0.114])
        ls, le = s @ lum_w, e @ lum_w
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gx_s, gy_s = convolve(ls, kx), convolve(ls, kx.T)
        gx_e, gy_e = convolve(le, kx), convolve(le, kx.T)
        g_s = np.sqrt(gx_s**2 + gy_s**2 + 1e-8)
        g_e = np.sqrt(gx_e**2 + gy_e**2 + 1e-8)
        return float(np.mean(np.abs(g_s - g_e) / (g_s + g_e + 1e-6)) * 1e3)

    # ── Unified entry point ───────────────────────────────────────────────────

    def all_metrics(self, src: Image.Image, edited: Image.Image,
                    prompt: str, mask_encoded=None) -> Dict[str, float]:
        if edited.size != src.size:
            edited = edited.resize(src.size, Image.LANCZOS)
        self._load()
        src512 = src.resize((512, 512))
        tgt512 = edited.resize((512, 512))

        result = {
            "CLIP_whole":       self.clip_whole(edited, prompt),
            "CLIP_edited":      self.clip_edited(src, edited, prompt),
            "clip_i":           self.clip_i(src512, tgt512),
            "LPIPS_alex":       self._lpips_alex(
                                    self._tensor11(src), self._tensor11(edited)
                                ).item() * 1e3,
            "LPIPS_squeeze":    self._mc.calculate_lpips(src512, tgt512, None, None) * 1e3,
            "MSE":              self.mse(src, edited),
            "PSNR":             self.psnr(src, edited),
            "SSIM":             self.ssim(src, edited),
            "StructDist_Sobel": self.struct_dist_sobel(src, edited),
            "StructDist_DINO":  float(self._mc.calculate_structure_distance(
                                    src512, tgt512, None, None)) * 1e3,
        }

        return result

def _strip_pie_brackets(text: str) -> str:
    """Remove PIE-Bench annotation brackets: '[rusty]' → 'rusty'."""
    import re as _re
    return _re.sub(r"\[([^\]]+)\]", r"\1", text).strip()


def load_pairs_pie(mapping_file: str, images_root: str,
                   max_pairs: Optional[int] = None) -> List[Dict]:
    """Return list of editing pairs from a PIE-Bench mapping_file.json."""
    with open(mapping_file) as f:
        data = json.load(f)
    pairs = []
    for key, entry in data.items():
        rel      = entry["image_path"]          # e.g. "0_random_140/000000000000.jpg"
        img_path = str(Path(images_root) / rel)
        resolved = _find_image(img_path)
        if resolved is None:
            continue
        parts     = Path(rel)
        base_name = f"{parts.parent.name}__{parts.stem}"
        pairs.append({
            "image_path":    resolved,
            "base_name":     base_name,
            "source_prompt": _strip_pie_brackets(str(entry["original_prompt"])),
            "target_prompt": _strip_pie_brackets(str(entry["editing_prompt"])),
            "code":          key,
        })
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def load_pairs(yaml_path: str, images_root: str,
               max_pairs: Optional[int] = None) -> List[Dict]:
    """Return list of editing pairs from the FlowEdit YAML."""
    with open(yaml_path) as f:
        entries = list(yaml.safe_load_all(f))[0]
    pairs = []
    for entry in entries:
        img_name  = Path(entry["init_img"]).name
        img_path  = str(Path(images_root) / img_name)
        resolved  = _find_image(img_path)
        if resolved:
            img_path = resolved
        base_name = Path(img_name).stem
        src_prompt = str(entry["source_prompt"]).strip()
        for tp, code in zip(entry["target_prompts"], entry["target_codes"]):
            pairs.append({
                "image_path":    img_path,
                "base_name":     base_name,
                "source_prompt": src_prompt,
                "target_prompt": str(tp).strip(),
                "code":          str(code).strip(),
            })
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs

CSV_COLS = ["method", "image", "code",
            "CLIP_whole", "CLIP_edited", "LPIPS", "MSE", "PSNR", "SSIM",
            "StructDist", "file"]

def write_csv(out_dir: Path, method: str, rows: List[Dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results_{method}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path

def save_image(img: Image.Image, out_dir: Path, method: str,
               base: str, code: str) -> Path:
    folder = out_dir / method
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{base}_{code}.png"
    img.save(p)
    return p

def _make_row(pair: Dict, method: str, metrics: Dict, file_path: str) -> Dict:
    return {
        "method":      method,
        "image":       pair["base_name"],
        "code":        pair["code"],
        **metrics,
        "file":        file_path,
    }

def _load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def _crop16(img: Image.Image) -> Image.Image:
    """Crop so both dimensions are divisible by 16."""
    w, h = img.size
    return img.crop((0, 0, w - w % 16, h - h % 16))

def _setup_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(LOG_FMT))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log

def _find_image(path: str) -> Optional[str]:
    """Return an existing path for the image, trying .png / .jpg / .jpeg in order."""
    p = Path(path)
    if p.exists():
        return str(p)
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = p.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
    return None

def _safe_load_pil(path: str, log=None) -> Optional[Image.Image]:
    """Load an image as PIL RGB, trying multiple extensions. Returns None and warns if not found."""
    resolved = _find_image(path)
    if resolved is None:
        msg = f"Image not found (skipping): {path}"
        if log:
            log.warning(msg)
        else:
            print(f"[WARNING] {msg}")
        return None
    return Image.open(resolved).convert("RGB")

def _patch_flux_rope():
    """
    Monkey-patch apply_rotary_emb to tolerate RoPE dimension mismatches in diffusers > 0.31.0.
    On ≤ 0.31.0 this is a no-op. On newer versions the patch retries with trimmed freqs_cis
    if the original call raises a shape error.
    """
    import functools
    try:
        import diffusers
        from packaging.version import Version
        if Version(diffusers.__version__) <= Version("0.31.0"):
            return
    except Exception:
        pass

    try:
        import diffusers.models.embeddings as _emb
        _orig = getattr(_emb, "apply_rotary_emb", None)
        if _orig is None:
            return

        @functools.wraps(_orig)
        def _safe_apply_rotary_emb(x, freqs_cis, **kwargs):
            try:
                return _orig(x, freqs_cis, **kwargs)
            except (RuntimeError, ValueError):
                seq_len = x.shape[-2] if x.ndim >= 2 else x.shape[0]
                if isinstance(freqs_cis, tuple):
                    freqs_cis = tuple(
                        f[:seq_len] if hasattr(f, "shape") and f.shape[0] > seq_len else f
                        for f in freqs_cis
                    )
                elif hasattr(freqs_cis, "shape") and freqs_cis.shape[0] > seq_len:
                    freqs_cis = freqs_cis[:seq_len]
                return _orig(x, freqs_cis, **kwargs)

        _emb.apply_rotary_emb = _safe_apply_rotary_emb

        try:
            import diffusers.models.attention_processor as _ap
            if hasattr(_ap, "apply_rotary_emb"):
                _ap.apply_rotary_emb = _safe_apply_rotary_emb
        except Exception:
            pass
    except Exception:
        pass

def run_flowedit_sd35(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FlowEdit on stabilityai/stable-diffusion-3.5-large.
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("flowedit_sd35")
    log.info(f"Starting FlowEdit (SD 3.5) on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="FlowEdit (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img  = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                n_min=0, n_max=33,
            )
            x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, "flowedit_sd35",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, "flowedit_sd35", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"FlowEdit SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "flowedit_sd35", rows)
    log.info("FlowEdit (SD 3.5) done.")

# def run_flowedit_sd35_zeroinit(pairs: List[Dict], gpu_id: int, out_dir: Path):
#     """FlowEdit on stabilityai/stable-diffusion-3.5-large.
#     T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
#     """
#     import random
#     torch.manual_seed(42)
#     random.seed(42)

#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     device = "cuda:0"
#     expr_name = "flowedit_sd35_zeroinit"
#     log = _setup_logger(expr_name)
#     log.info(f"Starting FlowEdit (SD 3.5) on GPU {gpu_id}")

#     if str(ROOT) not in sys.path:
#         sys.path.insert(0, str(ROOT))
#     from FlowEdit_utils import FlowEditSD3_CFGZero
#     from diffusers import StableDiffusion3Pipeline

#     SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
#     pipe = StableDiffusion3Pipeline.from_pretrained(
#         SD35_MODEL, torch_dtype=torch.float16).to(device)
#     scheduler = pipe.scheduler
#     evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

#     rows: List[Dict] = []
#     for pair in tqdm(pairs, desc="FlowEdit (SD3.5)"):
#         try:
#             _raw = _safe_load_pil(pair["image_path"], log)
#             if _raw is None:
#                 continue
#             src_img  = _crop16(_raw)
#             img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
#             with torch.autocast("cuda"), torch.inference_mode():
#                 x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
#             x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
#                      * pipe.vae.config.scaling_factor).to(device)

#             x_tar = FlowEditSD3_CFGZero(
#                 pipe, x_src,
#                 pair["source_prompt"], pair["target_prompt"],
#                 negative_prompt="",
#                 T_steps=50, n_avg=1,
#                 src_guidance_scale=3.5, tar_guidance_scale=13.5,
#                 n_min=0, n_max=33,
#             )
#             x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor
#                             + pipe.vae.config.shift_factor)
#             with torch.autocast("cuda"), torch.inference_mode():
#                 img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
#             edited = pipe.image_processor.postprocess(img_out)[0]

#             fp = save_image(edited, out_dir, expr_name,
#                             pair["base_name"], pair["code"])
#             m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
#             rows.append(_make_row(pair, expr_name, m, str(fp)))
#             log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
#         except Exception:
#             log.error(f"FlowEdit SD3.5 failed {pair['base_name']} {pair['code']}:\n"
#                       + traceback.format_exc())

#     write_csv(out_dir, "flowedit_sd35", rows)
#     log.info("FlowEdit (SD 3.5) done.")

def run_flowedit(pairs: List[Dict], gpu_id: int, out_dir: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("flowedit")
    log.info(f"Starting FlowEdit on GPU {gpu_id}")

    _patch_flux_rope()

    sys.path.insert(0, str(METHODS_ROOT / "FlowEdit"))
    from FlowEdit_utils import FlowEditFLUX
    from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    ).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device)

    # Patch FluxPipeline to fix FlowEditFLUX's factor-of-2 error in orig_height/orig_width.
    # FlowEditFLUX computes: orig = latent_dim * vae_scale_factor // 2  (missing factor of 2)
    # This causes wrong image-ID counts in prepare_latents and wrong unpacking in _unpack_latents.
    # Fix: derive image IDs directly from actual latent shape; auto-correct unpack dimensions.
    import types as _t
    from diffusers import FluxPipeline as _FluxPipeline

    _orig_prepare_latents = _FluxPipeline.prepare_latents
    def _fixed_prepare_latents(self, batch_size, num_channels_latents,
                                height, width, dtype, device, generator,
                                latents=None):
        if latents is not None:
            H, W = latents.shape[2], latents.shape[3]
            lat_ids = self._prepare_latent_image_ids(batch_size, H // 2, W // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), lat_ids
        return _orig_prepare_latents(self, batch_size, num_channels_latents,
                                     height, width, dtype, device, generator, latents)

    _orig_unpack_latents = _FluxPipeline._unpack_latents
    @staticmethod
    def _fixed_unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        # Detect factor-of-2 under-estimate and correct it
        h = 2 * (int(height) // (vae_scale_factor * 2))
        w = 2 * (int(width)  // (vae_scale_factor * 2))
        if (h // 2) * (w // 2) != num_patches:
            h = 2 * (int(height * 2) // (vae_scale_factor * 2))
            w = 2 * (int(width * 2)  // (vae_scale_factor * 2))
        latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, h, w)
        return latents

    _FluxPipeline.prepare_latents  = _fixed_prepare_latents
    _FluxPipeline._unpack_latents  = _fixed_unpack_latents

    rows = []
    for pair in tqdm(pairs, desc="FlowEdit"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = (x_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x_src = x_src.to(device)

            x_tar = FlowEditFLUX(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=28, n_avg=1,
                src_guidance_scale=1.5, tar_guidance_scale=5.5,
                n_min=0, n_max=24,
            )

            x_tar_denorm = x_tar / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, "flowedit", pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "flowedit", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"FlowEdit failed {pair['base_name']} {pair['code']}:\n{traceback.format_exc()}")

    write_csv(out_dir, "flowedit", rows)
    log.info("FlowEdit done.")

def _ddim_inversion_sdxl(pipeline, scheduler, image: Image.Image,
                          src_prompt: str, num_steps: int = 50,
                          guidance_scale: float = 1.0,
                          device: str = "cuda:0") -> torch.Tensor:
    """
    DDIM inversion for SDXL: encode image → run forward ODE → return z_T.
    guidance_scale=1.0 means no CFG during inversion (standard practice).
    """
    vae_dtype = next(pipeline.vae.parameters()).dtype
    img_t = pipeline.image_processor.preprocess(image).to(device, vae_dtype)
    with torch.no_grad():
        latents = pipeline.vae.encode(img_t).latent_dist.sample()
    latents = latents * pipeline.vae.config.scaling_factor

    (text_embeds, neg_embeds, pooled_text, pooled_neg) = pipeline.encode_prompt(
        prompt=src_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    add_time_ids = pipeline._get_add_time_ids(
        original_size=(image.height, image.width),
        crops_coords_top_left=(0, 0),
        target_size=(image.height, image.width),
        dtype=vae_dtype,
    ).to(device)

    from diffusers import DDIMScheduler as _StdDDIM
    import inspect as _inspect
    _ddim_keys = set(_inspect.signature(_StdDDIM.__init__).parameters) - {"self"}
    _clean_cfg  = {k: v for k, v in dict(scheduler.config).items() if k in _ddim_keys}
    inv_sched   = _StdDDIM(**_clean_cfg)
    inv_sched.set_timesteps(num_steps, device=device)
    timesteps_inv = inv_sched.timesteps.flip(0)

    latents = latents.to(vae_dtype)
    for t in timesteps_inv:
        with torch.no_grad():
            noise_pred = pipeline.unet(
                latents, t,
                encoder_hidden_states=text_embeds,
                added_cond_kwargs={"text_embeds": pooled_text, "time_ids": add_time_ids},
            ).sample
        alpha_t = inv_sched.alphas_cumprod[t]
        alpha_prev = inv_sched.alphas_cumprod[max(t - inv_sched.config.num_train_timesteps // num_steps, 0)]
        x0_pred   = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        dir_xt    = (1 - alpha_prev).sqrt() * noise_pred
        latents   = alpha_prev.sqrt() * x0_pred + dir_xt

    return latents

def run_cag(pairs: List[Dict], gpu_id: int, out_dir: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("cag")
    log.info(f"Starting CAG (Annealing Guidance) on GPU {gpu_id}")

    sys.path.insert(0, str(METHODS_ROOT / "annealing-guidance"))
    os.chdir(str(METHODS_ROOT / "annealing-guidance"))
    try:
        import src.utils.model_utils as model_utils
        ckpt_path = str(METHODS_ROOT / "annealing-guidance" /
                        "src" / "model" / "checkpoints" / "checkpoint.pt")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"CAG checkpoint not found: {ckpt_path}")
        config, pipeline, guidance_scale_network = model_utils.load_models(
            checkpoint_path=ckpt_path, device=device
        )
        pipeline = pipeline.to(device)
        guidance_scale_network = guidance_scale_network.to(device)
        scheduler = pipeline.scheduler
        has_model = True
    except Exception as e:
        log.warning(f"CAG model load failed ({e}); falling back to SDXL vanilla CFG.")
        from diffusers import StableDiffusionXLPipeline, DDIMScheduler
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to(device)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        scheduler = pipeline.scheduler
        guidance_scale_network = None
        has_model = False
    finally:
        os.chdir(str(ROOT))

    evaluator = MetricsEvaluator(device)
    rows = []

    for pair in tqdm(pairs, desc="CAG"):
        try:
            src_img = _safe_load_pil(pair["image_path"], log)
            if src_img is None:
                continue
            W, H    = src_img.size

            z_T = _ddim_inversion_sdxl(
                pipeline, scheduler, src_img, pair["source_prompt"],
                num_steps=50, device=device
            )
            scheduler.set_timesteps(50, device=device)
            scheduler.config.pred_sample_direction_with_null = has_model
            with torch.no_grad():
                out = pipeline(
                    prompt=pair["target_prompt"],
                    latents=z_T,
                    guidance_scale=7.5,
                    guidance_scale_model=guidance_scale_network if has_model else None,
                    guidance_lambda=0.4 if has_model else None,
                    num_inference_steps=50,
                    height=H, width=W,
                    output_type="pil",
                )
            edited = out.images[0]

            fp = save_image(edited, out_dir, "cag", pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "cag", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"CAG failed {pair['base_name']} {pair['code']}:\n{traceback.format_exc()}")

    write_csv(out_dir, "cag", rows)
    log.info("CAG done.")

def _run_flux_fireflow_method(pairs: List[Dict], gpu_id: int, out_dir: Path,
                               method_name: str, strategy: str,
                               num_steps: int, guidance: float, inject: int,
                               start_layer: int, end_layer: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(method_name)
    log.info(f"Starting {method_name} (strategy={strategy}) on GPU {gpu_id}")

    _patch_flux_rope()

    ff_src = str(METHODS_ROOT / "FireFlow" / "src")
    if ff_src not in sys.path:
        sys.path.insert(0, ff_src)

    from flux.util import load_flow_model, load_t5, load_clip, load_ae
    from flux.sampling import prepare, get_schedule, unpack
    if strategy == "reflow":
        from flux.sampling import denoise as _denoise_fn
        denoise_fn = _denoise_fn
    elif strategy == "rf_solver":
        from flux.sampling import denoise_rf_solver as _denoise_fn
        denoise_fn = _denoise_fn
    elif strategy == "fireflow":
        from flux.sampling import denoise_fireflow as _denoise_fn
        denoise_fn = _denoise_fn
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    t5   = load_t5("cpu", max_length=512)
    clip = load_clip(device)
    model = load_flow_model("flux-dev", device=device)
    ae    = load_ae("flux-dev", device=device)

    evaluator = MetricsEvaluator(device)
    rows = []

    for pair in tqdm(pairs, desc=method_name):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            _np = np.array(src_img).astype(np.float32)
            _t  = torch.from_numpy(_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            _t  = _t.to(device)
            with torch.inference_mode():
                src_latent = ae.encode(_t).to(torch.bfloat16)

            inp     = prepare(t5, clip, src_latent, prompt=pair["source_prompt"])
            inp_tar = prepare(t5, clip, src_latent, prompt=pair["target_prompt"])
            timesteps = get_schedule(num_steps, inp["img"].shape[1],
                                     shift=True)

            info = {
                "feature":        {},
                "inject_step":    inject,
                "start_layer_index": start_layer,
                "end_layer_index":   end_layer,
                "reuse_v":        1 if strategy == "fireflow" else 0,
                "editing_strategy": "replace_v",
                "qkv_ratio":      [1.0, 1.0, 1.0],
            }

            with torch.inference_mode():
                z, info = denoise_fn(model, **inp, timesteps=timesteps,
                                     guidance=1.0, inverse=True, info=info)

            inp_tar["img"] = z
            timesteps = get_schedule(num_steps, inp_tar["img"].shape[1],
                                     shift=True)

            with torch.inference_mode():
                x, _ = denoise_fn(model, **inp_tar, timesteps=timesteps,
                                   guidance=guidance, inverse=False, info=info)

            x_unpacked = unpack(x.float(),
                                 src_img.width, src_img.height)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded = ae.decode(x_unpacked)
            img_np = (decoded.clamp(-1, 1).permute(0, 2, 3, 1)
                      .cpu().float().numpy()[0])
            edited = Image.fromarray((img_np * 127.5 + 127.5).clip(0, 255)
                                     .astype(np.uint8))

            fp = save_image(edited, out_dir, method_name,
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, method_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"{method_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, method_name, rows)
    log.info(f"{method_name} done.")

def run_rf_inversion(pairs, gpu_id, out_dir):
    _run_flux_fireflow_method(
        pairs, gpu_id, out_dir,
        method_name="rf_inversion", strategy="reflow",
        num_steps=25, guidance=2.0, inject=2,
        start_layer=0, end_layer=37,
    )

def run_rf_solver(pairs, gpu_id, out_dir):
    _run_flux_fireflow_method(
        pairs, gpu_id, out_dir,
        method_name="rf_solver", strategy="rf_solver",
        num_steps=25, guidance=2.0, inject=2,
        start_layer=0, end_layer=37,
    )

def run_fireflow(pairs, gpu_id, out_dir):
    _run_flux_fireflow_method(
        pairs, gpu_id, out_dir,
        method_name="fireflow", strategy="fireflow",
        num_steps=8, guidance=2.0, inject=1,
        start_layer=0, end_layer=37,
    )

def run_sd1x_methods(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """
    Runs three SD-1.4 methods sequentially on a single GPU:
      1. DDIM + P2P
      2. Null-Text Inversion + P2P
      3. DirectInversion + P2P  (PnP-Inv)
    All use CompVis/stable-diffusion-v1-4, guidance_scale=7.5, 50 DDIM steps.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("sd1x")
    log.info(f"Starting SD1.x methods on GPU {gpu_id}")

    import PIL.Image as _pil_img
    if not hasattr(_pil_img, "ANTIALIAS"):
        _pil_img.ANTIALIAS = _pil_img.LANCZOS

    pnp_root = METHODS_ROOT / "PnPInversion"
    sys.path = [p for p in sys.path if "PnPInversion" not in p]
    sys.path.insert(0, str(pnp_root / "core"))
    sys.path.insert(0, str(pnp_root))

    (pnp_root / "utils").mkdir(exist_ok=True)
    (pnp_root / "utils" / "__init__.py").touch(exist_ok=True)

    from models.p2p_editor import P2PEditor

    editor = P2PEditor(
        ["ddim+p2p", "null-text-inversion+p2p", "directinversion+p2p"],
        device,
        num_ddim_steps=50,
    )
    evaluator = MetricsEvaluator(str(device))

    method_map = {
        "ddim":      "ddim+p2p",
        "null_text": "null-text-inversion+p2p",
        "pnp_inv":   "directinversion+p2p",
    }

    all_rows: Dict[str, List[Dict]] = {k: [] for k in method_map}

    for pair in tqdm(pairs, desc="SD1.x"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue

        _resolved_path = _find_image(pair["image_path"]) or pair["image_path"]
        for key, edit_method in method_map.items():
            try:
                torch.cuda.empty_cache()
                edited = editor(
                    edit_method,
                    image_path=_resolved_path,
                    prompt_src=pair["source_prompt"],
                    prompt_tar=pair["target_prompt"],
                    guidance_scale=7.5,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                )
                fp = save_image(edited, out_dir, key,
                                pair["base_name"], pair["code"])
                m  = evaluator.all_metrics(src_img, edited,
                                           pair["target_prompt"])
                all_rows[key].append(_make_row(pair, key, m, str(fp)))
                log.info(f"  [{key}] {pair['base_name']} {pair['code']}  "
                         f"CLIP={m['CLIP_whole']:.3f}")
            except Exception:
                log.error(f"{key} failed {pair['base_name']} {pair['code']}:\n"
                          + traceback.format_exc())

    for key, rows in all_rows.items():
        write_csv(out_dir, key, rows)
    log.info("SD1.x methods done.")

def run_irfds(pairs: List[Dict], gpu_id: int, out_dir: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("irfds")
    log.info(f"Starting iRFDS on GPU {gpu_id}")

    irfds_root = str(METHODS_ROOT / "rectified_flow_prior")
    if irfds_root not in sys.path:
        sys.path.insert(0, irfds_root)

    import types as _types, builtins as _bi
    _C_EXT_STUBS = {"tinycudann", "igl", "envlight", "nvdiffrast", "nerfacc",
                    "open3d", "pymeshlab", "xatlas", "cv2"}
    _real_import = _bi.__import__

    def _permissive_import(name, globs=None, locs=None, fromlist=(), level=0):
        try:
            return _real_import(name, globs, locs, fromlist, level)
        except (ImportError, ModuleNotFoundError):
            root = name.split(".")[0]
            if root in _C_EXT_STUBS:
                for i, part in enumerate(name.split(".")):
                    full = ".".join(name.split(".")[:i + 1])
                    if full not in sys.modules:
                        sys.modules[full] = _types.ModuleType(full)
                return sys.modules[name.split(".")[0]]
            raise

    _bi.__import__ = _permissive_import
    for _s in _C_EXT_STUBS:
        if _s not in sys.modules:
            sys.modules[_s] = _types.ModuleType(_s)
    sys.modules["tinycudann"].free_temporary_memory = lambda: None
    sys.modules["igl"].fast_winding_number_for_meshes = lambda *a, **kw: None
    sys.modules["igl"].point_mesh_squared_distance    = lambda *a, **kw: (None,) * 3
    sys.modules["igl"].read_obj                       = lambda *a, **kw: (None,) * 6
    _nvdr_torch = _types.ModuleType("nvdiffrast.torch")
    for _cls in ("RasterizeGLContext", "RasterizeCudaContext", "DepthPeelContext"):
        setattr(_nvdr_torch, _cls, type(_cls, (), {"__init__": lambda *a, **kw: None}))
    for _fn in ("rasterize", "interpolate", "antialias", "texture", "antialias_func"):
        setattr(_nvdr_torch, _fn, lambda *a, **kw: (None, None))
    sys.modules["nvdiffrast.torch"] = _nvdr_torch
    sys.modules["nvdiffrast"].torch = _nvdr_torch
    try:
        import threestudio
    finally:
        _bi.__import__ = _real_import
    from diffusers import StableDiffusion3Pipeline
    import torch.nn.functional as F
    import torchvision.transforms as T

    SD3_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe_sd3  = StableDiffusion3Pipeline.from_pretrained(
        SD3_MODEL, torch_dtype=torch.float16
    ).to(device)
    pipe_sd3.set_progress_bar_config(disable=True)

    guidance_cfg = {
        "half_precision_weights": True,
        "view_dependent_prompting": False,
        "guidance_scale": 1.0,
        "pretrained_model_name_or_path": SD3_MODEL,
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
    }
    pp_cfg = {
        "pretrained_model_name_or_path": SD3_MODEL,
        "spawn": False,
    }
    to_tensor = T.Compose([T.ToTensor()])
    evaluator = MetricsEvaluator(device)
    rows = []

    for pair in tqdm(pairs, desc="iRFDS"):
        try:
            src_img = _safe_load_pil(pair["image_path"], log)
            if src_img is None:
                continue
            img_t     = to_tensor(src_img).unsqueeze(0).to(device)
            img_512   = F.interpolate(img_t, (512, 512), mode="bilinear",
                                      align_corners=False)

            pp_cfg["prompt"] = pair["source_prompt"]
            guidance = threestudio.find("iRFDS-sd3")(guidance_cfg).to(device)
            guidance.camera_embedding = guidance.camera_embedding.to(device)
            pp_cfg_copy = dict(pp_cfg)
            pp_cfg_copy["prompt"] = pair["source_prompt"]
            prompt_processor = threestudio.find("sd3-prompt-processor")(pp_cfg_copy)

            with torch.no_grad():
                target_latent = guidance.encode_images(img_512)

            target = target_latent.clone().detach().requires_grad_(True)
            optimizer = torch.optim.AdamW([target], lr=2e-3, weight_decay=0)

            max_iters           = 1400
            n_accumulation      = 2
            prompt_utils        = prompt_processor()
            dummy_cam           = torch.zeros([1, 4, 4], device=device)
            dummy_elev          = torch.zeros([1], device=device)
            dummy_azim          = torch.zeros([1], device=device)
            dummy_dist          = torch.zeros([1], device=device)

            for step in range(max_iters * n_accumulation + 1):
                loss_dict = guidance(
                    noise_to_optimize=target,
                    rgb=target_latent,
                    prompt_utils=prompt_utils,
                    mvp_mtx=dummy_cam,
                    elevation=dummy_elev,
                    azimuth=dummy_azim,
                    camera_distances=dummy_dist,
                    c2w=dummy_cam.clone(),
                    rgb_as_latents=True,
                )
                loss = (loss_dict["loss_iRFDS"] +
                        loss_dict["loss_regularize"]) / n_accumulation
                loss.backward()
                if (step + 1) % n_accumulation == 0:
                    actual_step = (step + 1) // n_accumulation
                    guidance.update_step(epoch=0, global_step=actual_step)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            tar_pp_cfg = dict(pp_cfg)
            tar_pp_cfg["prompt"] = pair["target_prompt"]
            with torch.no_grad():
                out = pipe_sd3(
                    prompt=pair["target_prompt"],
                    latents=target.detach(),
                    num_inference_steps=15,
                    guidance_scale=2.0,
                    output_type="pil",
                )
            edited = out.images[0]
            del guidance, prompt_processor
            torch.cuda.empty_cache()

            fp = save_image(edited, out_dir, "irfds",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "irfds", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"iRFDS failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "irfds", rows)
    log.info("iRFDS done.")

def _run_ftedit(pairs, device, out_dir, log, evaluator):
    ftedit_root = str(METHODS_ROOT / "FTEdit")
    if ftedit_root not in sys.path:
        sys.path.insert(0, ftedit_root)

    from mmdit.sd35_pipeline import StableDiffusion3Pipeline as SD35Pipeline
    from inversion.flow_fixpoint_residual_new import Inversed_flow_fixpoint_residual
    from controller import attn_norm_ctrl_sd35

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = SD35Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.transformer.eval()
    pipe.vae.eval()

    saved_path = str(out_dir / "ftedit")
    Path(saved_path).mkdir(parents=True, exist_ok=True)

    invf = Inversed_flow_fixpoint_residual(
        pipe,
        steps=30,
        device=device,
        inv_cfg=1.0,
        recov_cfg=1.0,
        skip_steps=7,
        saved_path=saved_path,
    )
    rows = []

    for pair in tqdm(pairs, desc="FTEdit"):
        try:
            src_img = _safe_load_pil(pair["image_path"], log)
            if src_img is None:
                continue
            _img_path = _find_image(pair["image_path"]) or pair["image_path"]
            prompts = [pair["source_prompt"], pair["target_prompt"]]

            attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, None, None)
            all_latents = invf.euler_flow_inversion(
                prompt=pair["source_prompt"],
                image=_img_path,
                num_fixpoint_steps=3,
                average_step_ranges=(0, 5),
            )

            controller_ada  = attn_norm_ctrl_sd35.Adalayernorm_replace(
                prompts, 30, 1.0,
                pipe.tokenizer, pipe.tokenizer_3, device=device,
            )
            controller_attn = attn_norm_ctrl_sd35.SD3attentionreplace(
                prompts, 30, 1.0
            )
            attn_norm_ctrl_sd35.register_attention_control_sd35(
                pipe, controller_attn, controller_ada
            )

            _image1, image2 = invf.edit_img_with_residual(
                prompts, all_latents, controller_ada
            )

            if isinstance(image2, np.ndarray):
                arr = np.squeeze(image2)
                if arr.dtype != np.uint8:
                    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
                edited = Image.fromarray(arr)
            else:
                edited = image2

            fp = save_image(edited, out_dir, "ftedit",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "ftedit", m, str(fp)))
            log.info(f"  [FTEdit] {pair['base_name']} {pair['code']}  "
                     f"CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"FTEdit failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "ftedit", rows)

    del pipe, invf
    torch.cuda.empty_cache()
    return rows

def _run_splitflow(pairs, device, out_dir, log, evaluator):
    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3
    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD3_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD3_MODEL, torch_dtype=torch.float16
    ).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16
    )

    rows = []

    for pair in tqdm(pairs, desc="SplitFlow"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n"
                f"\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n"
                f"\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded  = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            intermed = re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = (x0_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x0_src = x0_src.to(device)

            x0_tar = SplitFlowSD3(
                pipe, scheduler, x0_src,
                pair["source_prompt"],
                tar_prompts,
                "",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5,
                edit_guidance_scale=13.5,
                n_min=0, n_max=33,
            )

            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, "splitflow",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "splitflow", m, str(fp)))
            log.info(f"  [SplitFlow] {pair['base_name']} {pair['code']}  "
                     f"CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"SplitFlow failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "splitflow", rows)
    del pipe, llm
    torch.cuda.empty_cache()
    return rows

def run_ftedit_splitflow(pairs: List[Dict], gpu_id: int, out_dir: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("gpu7")
    log.info(f"Starting FTEdit + SplitFlow on GPU {gpu_id}")
    evaluator = MetricsEvaluator(device)
    _run_ftedit(pairs, device, out_dir, log, evaluator)
    _run_splitflow(pairs, device, out_dir, log, evaluator)
    log.info("FTEdit + SplitFlow done.")

def _make_pnp_preprocess(device, model_key, num_steps=50):
    """Return a DDIM-inversion helper object for model_key."""
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    import torchvision.transforms as _T

    class _Preprocess:
        def __init__(self):
            self.device = device
            self.vae = AutoencoderKL.from_pretrained(
                model_key, subfolder="vae", torch_dtype=torch.float16).to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_key, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_key, subfolder="text_encoder",
                torch_dtype=torch.float16).to(device)
            self.unet = UNet2DConditionModel.from_pretrained(
                model_key, subfolder="unet",
                torch_dtype=torch.float16).to(device)
            self.scheduler = DDIMScheduler.from_pretrained(
                model_key, subfolder="scheduler")

        @torch.no_grad()
        def _cond(self, prompt):
            inp = self.tokenizer(
                prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt")
            return self.text_encoder(inp.input_ids.to(self.device))[0]

        @torch.no_grad()
        def encode_img(self, image_path):
            img = _T.Compose([_T.Resize(512), _T.ToTensor()])(
                Image.open(image_path).convert("RGB")
            ).unsqueeze(0).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                lat = self.vae.encode(2 * img - 1).latent_dist.mean * 0.18215
            return lat

        @torch.no_grad()
        def ddim_inversion(self, cond, latent):
            self.scheduler.set_timesteps(num_steps)
            rev_ts = list(reversed(self.scheduler.timesteps.tolist()))
            latent_list = [latent]
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                for i, t in enumerate(rev_ts):
                    cond_b = cond.repeat(latent.shape[0], 1, 1)
                    alpha_t    = self.scheduler.alphas_cumprod[t]
                    alpha_prev = (
                        self.scheduler.alphas_cumprod[rev_ts[i - 1]]
                        if i > 0 else self.scheduler.final_alpha_cumprod
                    )
                    mu, mu_prev = alpha_t ** 0.5, alpha_prev ** 0.5
                    sigma = (1 - alpha_t) ** 0.5
                    sigma_prev = (1 - alpha_prev) ** 0.5
                    eps = self.unet(latent, t,
                                   encoder_hidden_states=cond_b).sample
                    pred_x0 = (latent - sigma_prev * eps) / mu_prev
                    latent = mu * pred_x0 + sigma * eps
                    latent_list.append(latent)
            return latent_list

        @torch.no_grad()
        def ddim_sample(self, x, cond):
            self.scheduler.set_timesteps(num_steps)
            ts = self.scheduler.timesteps
            latent_list = []
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                for i, t in enumerate(ts):
                    cond_b  = cond.repeat(x.shape[0], 1, 1)
                    alpha_t = self.scheduler.alphas_cumprod[t]
                    alpha_prev = (
                        self.scheduler.alphas_cumprod[ts[i + 1].item()]
                        if i < len(ts) - 1 else self.scheduler.final_alpha_cumprod
                    )
                    mu, sigma = alpha_t ** 0.5, (1 - alpha_t) ** 0.5
                    mu_prev = alpha_prev ** 0.5
                    sigma_prev = (1 - alpha_prev) ** 0.5
                    eps = self.unet(x, t,
                                   encoder_hidden_states=cond_b).sample
                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    latent_list.append(x)
            return latent_list

        def extract_latents(self, image_path, src_prompt=""):
            """Return (inverted_x, recon_reversed):
            inverted_x[0]=z0 clean, inverted_x[-1]=zT noisy.
            recon_reversed[-1]=zT noisy (after reverse).
            """
            cond       = self._cond(src_prompt)
            latent     = self.encode_img(image_path)
            inverted_x = self.ddim_inversion(cond, latent)
            recon      = self.ddim_sample(inverted_x[-1], cond)
            recon.reverse()
            return inverted_x, recon

    return _Preprocess()

def _make_pnp_model(device, model_key, num_steps=50):
    """Return a PNP attention/conv injection object for model_key."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    class _PNP:
        def __init__(self):
            self.device = device
            pipe = StableDiffusionPipeline.from_pretrained(
                model_key, torch_dtype=torch.float16).to(device)
            self.vae          = pipe.vae
            self.tokenizer    = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.unet         = pipe.unet
            self.scheduler    = DDIMScheduler.from_pretrained(
                model_key, subfolder="scheduler")
            self.scheduler.set_timesteps(num_steps, device=device)
            self.n_timesteps  = num_steps
            del pipe
            torch.cuda.empty_cache()

        @torch.no_grad()
        def get_text_embeds(self, prompt, negative_prompt):
            inp = self.tokenizer(
                prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt")
            emb = self.text_encoder(inp.input_ids.to(self.device))[0]
            unc = self.tokenizer(
                negative_prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt")
            unc_emb = self.text_encoder(unc.input_ids.to(self.device))[0]
            return torch.cat([unc_emb, emb])

        @torch.no_grad()
        def decode_latent(self, latent):
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                latent = 1 / 0.18215 * latent
                img = self.vae.decode(latent).sample
                img = (img / 2 + 0.5).clamp(0, 1)
            return img

        def _register_time(self, t):
            setattr(self.unet.up_blocks[1].resnets[1], "t", t)
            for res, blocks in {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}.items():
                for blk in blocks:
                    setattr(
                        self.unet.up_blocks[res].attentions[blk]
                        .transformer_blocks[0].attn1, "t", t)
            for res, blocks in {0: [0, 1], 1: [0, 1], 2: [0, 1]}.items():
                for blk in blocks:
                    setattr(
                        self.unet.down_blocks[res].attentions[blk]
                        .transformer_blocks[0].attn1, "t", t)
            setattr(
                self.unet.mid_block.attentions[0].transformer_blocks[0].attn1,
                "t", t)

        def _register_attn(self, injection_schedule):
            def sa_forward(attn_mod):
                to_out = (attn_mod.to_out[0]
                          if isinstance(attn_mod.to_out, torch.nn.ModuleList)
                          else attn_mod.to_out)

                def forward(x, encoder_hidden_states=None,
                            attention_mask=None):
                    is_cross = encoder_hidden_states is not None
                    enc_hs   = encoder_hidden_states if is_cross else x
                    if (not is_cross
                            and attn_mod.injection_schedule is not None
                            and (attn_mod.t in attn_mod.injection_schedule
                                 or attn_mod.t == 1000)):
                        q = attn_mod.to_q(x)
                        k = attn_mod.to_k(enc_hs)
                        sb = q.shape[0] // 3
                        q[sb:2 * sb] = q[:sb]
                        k[sb:2 * sb] = k[:sb]
                        q[2 * sb:] = q[:sb]
                        k[2 * sb:] = k[:sb]
                    else:
                        q = attn_mod.to_q(x)
                        k = attn_mod.to_k(enc_hs)
                    q = attn_mod.head_to_batch_dim(q)
                    k = attn_mod.head_to_batch_dim(k)
                    v = attn_mod.to_v(enc_hs)
                    v = attn_mod.head_to_batch_dim(v)
                    sim = (torch.einsum("b i d, b j d -> b i j", q, k)
                           * attn_mod.scale)
                    attn = sim.softmax(dim=-1)
                    out  = torch.einsum("b i j, b j d -> b i d", attn, v)
                    return to_out(attn_mod.batch_to_head_dim(out))

                return forward

            for res, blocks in {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}.items():
                for blk in blocks:
                    m = (self.unet.up_blocks[res].attentions[blk]
                         .transformer_blocks[0].attn1)
                    m.forward = sa_forward(m)
                    setattr(m, "injection_schedule", injection_schedule)

        def _register_conv(self, injection_schedule):
            def conv_fwd(conv):
                def forward(inp, temb):
                    h = conv.norm1(inp)
                    h = conv.nonlinearity(h)
                    if conv.upsample is not None:
                        inp = conv.upsample(inp)
                        h   = conv.upsample(h)
                    elif conv.downsample is not None:
                        inp = conv.downsample(inp)
                        h   = conv.downsample(h)
                    h = conv.conv1(h)
                    if temb is not None:
                        t_e = conv.time_emb_proj(
                            conv.nonlinearity(temb))[:, :, None, None]
                        if conv.time_embedding_norm == "default":
                            h = h + t_e
                    h = conv.norm2(h)
                    if (temb is not None
                            and conv.time_embedding_norm == "scale_shift"):
                        scale, shift = torch.chunk(t_e, 2, dim=1)
                        h = h * (1 + scale) + shift
                    h = conv.nonlinearity(h)
                    h = conv.dropout(h)
                    h = conv.conv2(h)
                    if (conv.injection_schedule is not None
                            and (conv.t in conv.injection_schedule
                                 or conv.t == 1000)):
                        sb = h.shape[0] // 3
                        h[sb:2 * sb] = h[:sb]
                        h[2 * sb:] = h[:sb]
                    if conv.conv_shortcut is not None:
                        inp = conv.conv_shortcut(inp)
                    return (inp + h) / conv.output_scale_factor

                return forward

            c = self.unet.up_blocks[1].resnets[1]
            c.forward = conv_fwd(c)
            setattr(c, "injection_schedule", injection_schedule)

        @torch.no_grad()
        def denoise_step(self, x, t, guidance_scale, src_latent):
            lat_inp = torch.cat([src_latent, x, x])
            self._register_time(t.item())
            text_inp = torch.cat([self.pnp_guidance_embeds,
                                  self.text_embeds], dim=0)
            noise_pred = self.unet(lat_inp, t,
                                   encoder_hidden_states=text_inp)["sample"]
            _, noise_unc, noise_cond = noise_pred.chunk(3)
            noise_pred = noise_unc + guidance_scale * (noise_cond - noise_unc)
            return self.scheduler.step(noise_pred, t, x)["prev_sample"]

        def run_pnp(self, noisy_latent, target_prompt,
                    guidance_scale=7.5, pnp_f_t=0.8, pnp_attn_t=0.5):
            """noisy_latent: list [z0,...,zT]; zT = noisy_latent[-1]."""
            qk_t  = int(self.n_timesteps * pnp_attn_t)
            f_t   = int(self.n_timesteps * pnp_f_t)
            qk_ts = (self.scheduler.timesteps[:qk_t] if qk_t >= 0 else [])
            f_ts  = (self.scheduler.timesteps[:f_t]  if f_t  >= 0 else [])
            self._register_attn(qk_ts)
            self._register_conv(f_ts)
            self.text_embeds = self.get_text_embeds(
                target_prompt,
                "ugly, blurry, black, low res, unrealistic")
            self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
            x = noisy_latent[-1]
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                for i, t in enumerate(self.scheduler.timesteps):
                    x = self.denoise_step(x, t, guidance_scale,
                                          noisy_latent[-1 - i])
            return self.decode_latent(x)

    return _PNP()

def _pnp_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t[0].permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))

def _setup_pnp_repo(pnp_root: Path):
    """Add PnPInversion to sys.path and patch PIL.Image.ANTIALIAS."""
    import PIL.Image as _pil
    if not hasattr(_pil, "ANTIALIAS"):
        _pil.ANTIALIAS = _pil.LANCZOS
    sys.path = [p for p in sys.path if "PnPInversion" not in p]
    sys.path.insert(0, str(pnp_root / "core"))
    sys.path.insert(0, str(pnp_root))
    (pnp_root / "utils").mkdir(exist_ok=True)
    (pnp_root / "utils" / "__init__.py").touch(exist_ok=True)

def _swap_p2p_editor_to_model(editor, model_id: str, device):
    """Replace the ldm_stable pipeline inside a P2PEditor with model_id."""
    from models.p2p.scheduler_dev import DDIMSchedulerDev
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    del editor.ldm_stable
    torch.cuda.empty_cache()
    cfg = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler").config
    new_sched = DDIMSchedulerDev(
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample,
        set_alpha_to_one=cfg.set_alpha_to_one,
        prediction_type=cfg.prediction_type,
    )
    editor.ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=new_sched).to(device)
    editor.scheduler = new_sched
    editor.ldm_stable.scheduler.set_timesteps(editor.num_ddim_steps)

def run_new_ddim_sd14(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """DDIM+P2P and DDIM+PnP on CompVis/stable-diffusion-v1-4."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("ddim_sd14")
    log.info(f"Starting DDIM+P2P + DDIM+PnP (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    MODEL_KEY = "CompVis/stable-diffusion-v1-4"
    evaluator = MetricsEvaluator(str(device))

    from models.p2p_editor import P2PEditor
    editor = P2PEditor(["ddim+p2p"], device, num_ddim_steps=50)
    rows_p2p: List[Dict] = []
    for pair in tqdm(pairs, desc="DDIM+P2P (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _combined = editor(
                "ddim+p2p", image_path=_res,
                prompt_src=pair["source_prompt"],
                prompt_tar=pair["target_prompt"],
                guidance_scale=7.5,
                cross_replace_steps=0.4, self_replace_steps=0.6,
            )
            _pw = _combined.width // 4
            edited = _combined.crop((3 * _pw, 0, 4 * _pw, _combined.height))
            fp = save_image(edited, out_dir, "ddim_p2p",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows_p2p.append(_make_row(pair, "ddim_p2p", m, str(fp)))
            log.info(f"  [ddim+p2p] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"ddim+p2p failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "ddim_p2p", rows_p2p)
    del editor
    torch.cuda.empty_cache()

    preproc   = _make_pnp_preprocess(device, MODEL_KEY)
    pnp_model = _make_pnp_model(device, MODEL_KEY)
    rows_pnp: List[Dict] = []
    for pair in tqdm(pairs, desc="DDIM+PnP (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _, recon = preproc.extract_latents(_res,
                                               src_prompt=pair["source_prompt"])
            out_t  = pnp_model.run_pnp(recon, pair["target_prompt"],
                                        guidance_scale=7.5)
            edited = _pnp_tensor_to_pil(out_t)
            fp = save_image(edited, out_dir, "ddim_pnp",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows_pnp.append(_make_row(pair, "ddim_pnp", m, str(fp)))
            log.info(f"  [ddim+pnp] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"ddim+pnp failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "ddim_pnp", rows_pnp)
    log.info("DDIM+P2P + DDIM+PnP (SD 1.4) done.")

def run_nulltext_sd21(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """Null-text inversion + P2P on CompVis/stable-diffusion-v1-4 (SD 1.4)."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("nulltext_sd14")
    log.info(f"Starting Null-text+P2P (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    SD14_MODEL = "CompVis/stable-diffusion-v1-4"
    from models.p2p_editor import P2PEditor
    editor = P2PEditor(["null-text-inversion+p2p"], device, num_ddim_steps=50)
    _swap_p2p_editor_to_model(editor, SD14_MODEL, device)

    evaluator = MetricsEvaluator(str(device))
    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="Null-text+P2P (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _combined = editor(
                "null-text-inversion+p2p", image_path=_res,
                prompt_src=pair["source_prompt"],
                prompt_tar=pair["target_prompt"],
                guidance_scale=7.5,
                cross_replace_steps=0.4, self_replace_steps=0.6,
            )
            _pw = _combined.width // 4
            edited = _combined.crop((3 * _pw, 0, 4 * _pw, _combined.height))
            fp = save_image(edited, out_dir, "null_text_sd14",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "null_text_sd14", m, str(fp)))
            log.info(f"  [null-text+p2p SD1.4] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"null-text+p2p SD1.4 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "null_text_sd14", rows)
    log.info("Null-text+P2P (SD 1.4) done.")

def run_pnpinv_p2p_sd14(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """DirectInversion+P2P on CompVis/stable-diffusion-v1-4 (GPU 2)."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("pnpinv_p2p_sd14")
    log.info(f"Starting DirectInv+P2P (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    SD14_MODEL = "CompVis/stable-diffusion-v1-4"
    evaluator = MetricsEvaluator(str(device))

    from models.p2p_editor import P2PEditor
    editor = P2PEditor(["directinversion+p2p"], device, num_ddim_steps=50)
    _swap_p2p_editor_to_model(editor, SD14_MODEL, device)

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="DirectInv+P2P (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _combined = editor(
                "directinversion+p2p", image_path=_res,
                prompt_src=pair["source_prompt"],
                prompt_tar=pair["target_prompt"],
                guidance_scale=7.5,
                cross_replace_steps=0.4, self_replace_steps=0.6,
            )
            _pw = _combined.width // 4
            edited = _combined.crop((3 * _pw, 0, 4 * _pw, _combined.height))
            fp = save_image(edited, out_dir, "pnpinv_p2p_sd14",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "pnpinv_p2p_sd14", m, str(fp)))
            log.info(f"  [pnpinv+p2p SD1.4] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"pnpinv+p2p SD1.4 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "pnpinv_p2p_sd14", rows)
    log.info("DirectInv+P2P (SD 1.4) done.")

def run_pnpinv_pnp_sd14(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """DirectInversion+PnP on CompVis/stable-diffusion-v1-4 (GPU 5)."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("pnpinv_pnp_sd14")
    log.info(f"Starting DirectInv+PnP (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    SD14_MODEL = "CompVis/stable-diffusion-v1-4"
    evaluator = MetricsEvaluator(str(device))

    preproc   = _make_pnp_preprocess(device, SD14_MODEL)
    pnp_model = _make_pnp_model(device, SD14_MODEL)
    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="DirectInv+PnP (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            inverted_x, _ = preproc.extract_latents(
                _res, src_prompt=pair["source_prompt"])
            out_t  = pnp_model.run_pnp(inverted_x, pair["target_prompt"],
                                        guidance_scale=7.5)
            edited = _pnp_tensor_to_pil(out_t)
            fp = save_image(edited, out_dir, "pnpinv_pnp_sd14",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            rows.append(_make_row(pair, "pnpinv_pnp_sd14", m, str(fp)))
            log.info(f"  [pnpinv+pnp SD1.4] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"pnpinv+pnp SD1.4 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "pnpinv_pnp_sd14", rows)
    log.info("DirectInv+PnP (SD 1.4) done.")

def run_splitflow_sd35(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow on stabilityai/stable-diffusion-3.5-large.
    LLM: mistralai/Mistral-7B-Instruct-v0.3
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("splitflow_sd35")
    log.info(f"Starting SplitFlow (SD 3.5) on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="SplitFlow (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, edit_guidance_scale=13.5,
                n_min=0, n_max=33,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, "splitflow_sd35",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, "splitflow_sd35", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"SplitFlow SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "splitflow_sd35", rows)
    log.info("SplitFlow (SD 3.5) done.")

def run_flowedit_sd35_conflictaware_cosine(pairs: List[Dict], gpu_id: int, out_dir: Path,
                                     kappa_tar: float = 0.7, expr_name: str = "flowedit_sd35_conflictaware_cosine"):
    """FlowEdit Conflict-Aware on stabilityai/stable-diffusion-3.5-large.
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting FlowEdit (SD 3.5) on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3_ConflictAware_Cosine
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="FlowEdit (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img  = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3_ConflictAware_Cosine(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                kappa_tar=kappa_tar,
                n_min=0, n_max=33,
            )
            x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name,
                            pair["base_name"], pair["code"])
            # m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"FlowEdit SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows)
    log.info(f"Conflict-Aware FlowEdit (SD 3.5) done.")

def run_flowedit_sd35_conflictaware_relative(pairs: List[Dict], gpu_id: int, out_dir: Path,
                                     kappa_tar: float = 0.7, expr_name: str = "flowedit_sd35_conflictaware_relative"):
    """FlowEdit Conflict-Aware on stabilityai/stable-diffusion-3.5-large.
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting FlowEdit (SD 3.5) on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3_ConflictAware_Relative
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="FlowEdit (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img  = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3_ConflictAware_Relative(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                kappa_tar=kappa_tar,
                n_min=0, n_max=33,
            )
            x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name,
                            pair["base_name"], pair["code"])
            # m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"FlowEdit SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows)
    log.info(f"Conflict-Aware FlowEdit (SD 3.5) done.")

def run_flowedit_sd35_conflictaware_sigmatype(pairs: List[Dict], gpu_id: int, out_dir: Path,
                                     sigma_type: int = 1, expr_name: str = "flowedit_sd35_conflictaware_sigmatype"):
    """FlowEdit Conflict-Aware on stabilityai/stable-diffusion-3.5-large.
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting FlowEdit (SD 3.5) on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3_ConflictAware_SigmaType
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="FlowEdit (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img  = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3_ConflictAware_SigmaType(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                sigma_type=sigma_type,
                n_min=0, n_max=33,
            )
            x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name,
                            pair["base_name"], pair["code"])
            # m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"FlowEdit SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows)
    log.info(f"Conflict-Aware FlowEdit (SD 3.5) done.")

def run_ftedit_only(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FTEdit + AdaLN on SD 3.5-Large (no SplitFlow).
    num_steps=30, inv_cfg=1.0, recov_cfg=2.0, skip_steps=0,
    ly_ratio=1.0, attn_ratio=0.15, num_fixpoint_steps=3, average_step_ranges=(0,5).
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("ftedit_only")
    log.info(f"Starting FTEdit (SD 3.5) on GPU {gpu_id}")

    evaluator = MetricsEvaluator(device)
    _run_ftedit(pairs, device, out_dir, log, evaluator)
    log.info("FTEdit (SD 3.5) done.")

def run_irfds_sd35(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """iRFDS score distillation (SD3-medium, matching iRFDS_sd3.py hardcode).
    max_iters=1400, lr=2e-3, guidance_scale=2.0, num_inference_steps=15.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("irfds_sd35")
    log.info(f"Starting iRFDS (SD 3.5) on GPU {gpu_id}")

    irfds_root = str(METHODS_ROOT / "rectified_flow_prior")
    if irfds_root not in sys.path:
        sys.path.insert(0, irfds_root)
    # threestudio reads load/prompt_library.json relative to CWD
    os.chdir(irfds_root)

    import types as _types, builtins as _bi
    _C_EXT_STUBS = {"tinycudann", "igl", "envlight", "nvdiffrast", "nerfacc",
                    "open3d", "pymeshlab", "xatlas", "wandb", "cv2"}
    _real_import = _bi.__import__

    def _permissive_import(name, globs=None, locs=None, fromlist=(), level=0):
        try:
            return _real_import(name, globs, locs, fromlist, level)
        except (ImportError, ModuleNotFoundError):
            root = name.split(".")[0]
            if root in _C_EXT_STUBS:
                for i, part in enumerate(name.split(".")):
                    full = ".".join(name.split(".")[:i + 1])
                    if full not in sys.modules:
                        sys.modules[full] = _types.ModuleType(full)
                return sys.modules[name.split(".")[0]]
            raise

    _bi.__import__ = _permissive_import
    import importlib.machinery as _imach
    for _s in _C_EXT_STUBS:
        if _s not in sys.modules:
            _m = _types.ModuleType(_s)
            # Python 3.13: importlib.util.find_spec() raises ValueError if
            # __spec__ is None on a module already in sys.modules.
            _m.__spec__ = _imach.ModuleSpec(_s, None)
            sys.modules[_s] = _m
    sys.modules["tinycudann"].free_temporary_memory = lambda: None
    sys.modules["igl"].fast_winding_number_for_meshes = lambda *a, **kw: None
    sys.modules["igl"].point_mesh_squared_distance    = lambda *a, **kw: (None,) * 3
    sys.modules["igl"].read_obj                       = lambda *a, **kw: (None,) * 6
    _nvdr_torch = _types.ModuleType("nvdiffrast.torch")
    for _cls in ("RasterizeGLContext", "RasterizeCudaContext", "DepthPeelContext"):
        setattr(_nvdr_torch, _cls,
                type(_cls, (), {"__init__": lambda *a, **kw: None}))
    for _fn in ("rasterize", "interpolate", "antialias", "texture", "antialias_func"):
        setattr(_nvdr_torch, _fn, lambda *a, **kw: (None, None))
    sys.modules["nvdiffrast.torch"] = _nvdr_torch
    sys.modules["nvdiffrast"].torch = _nvdr_torch
    try:
        import threestudio
    finally:
        _bi.__import__ = _real_import

    from diffusers import StableDiffusion3Pipeline
    import torch.nn.functional as F
    import torchvision.transforms as T

    # iRFDS_sd3.py hardcodes SD3-medium internally for the score-distillation
    # transformer, so we must also use SD3-medium for the decoding pipeline to
    # avoid the model-mismatch grid artifacts seen with SD3.5-large.
    SD3_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe_sd3 = StableDiffusion3Pipeline.from_pretrained(
        SD3_MODEL, torch_dtype=torch.float16)
    pipe_sd3.enable_model_cpu_offload(gpu_id=0)
    pipe_sd3.set_progress_bar_config(disable=True)

    guidance_cfg = {
        "half_precision_weights":        True,
        "view_dependent_prompting":      False,
        "guidance_scale":                1.0,
        "pretrained_model_name_or_path": SD3_MODEL,
        "min_step_percent":              0.02,
        "max_step_percent":              0.98,
    }
    pp_cfg = {"pretrained_model_name_or_path": SD3_MODEL, "spawn": False}
    to_tensor = T.Compose([T.ToTensor()])
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="iRFDS (SD3.5)"):
        try:
            src_img = _safe_load_pil(pair["image_path"], log)
            if src_img is None:
                continue
            img_t   = to_tensor(src_img).unsqueeze(0).to(device)
            img_512 = F.interpolate(img_t, (512, 512), mode="bilinear",
                                    align_corners=False)

            pp_copy = dict(pp_cfg, prompt=pair["source_prompt"])
            guidance = threestudio.find("iRFDS-sd3")(guidance_cfg).to(device)
            guidance.camera_embedding = guidance.camera_embedding.to(device)
            prompt_processor = threestudio.find("sd3-prompt-processor")(pp_copy)

            with torch.no_grad():
                target_latent = guidance.encode_images(img_512)

            target    = target_latent.clone().detach().requires_grad_(True)
            optimizer = torch.optim.AdamW([target], lr=2e-3, weight_decay=0)

            max_iters      = 1400
            n_accumulation = 2
            prompt_utils   = prompt_processor()
            dummy_cam  = torch.zeros([1, 4, 4], device=device)
            dummy_elev = torch.zeros([1], device=device)
            dummy_azim = torch.zeros([1], device=device)
            dummy_dist = torch.zeros([1], device=device)

            for step in range(max_iters * n_accumulation + 1):
                loss_dict = guidance(
                    noise_to_optimize=target,
                    # forward() expects BHWC and does permute(0,3,1,2) internally;
                    # target_latent is BCHW so permute here to avoid shape mismatch.
                    rgb=target_latent.permute(0, 2, 3, 1),
                    prompt_utils=prompt_utils,
                    mvp_mtx=dummy_cam, elevation=dummy_elev,
                    azimuth=dummy_azim, camera_distances=dummy_dist,
                    c2w=dummy_cam.clone(), rgb_as_latents=True,
                )
                loss = (loss_dict["loss_iRFDS"]
                        + loss_dict["loss_regularize"]) / n_accumulation
                loss.backward()
                if (step + 1) % n_accumulation == 0:
                    actual_step = (step + 1) // n_accumulation
                    guidance.update_step(epoch=0, global_step=actual_step)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                out = pipe_sd3(
                    prompt=pair["target_prompt"],
                    latents=target.detach(),
                    num_inference_steps=15,
                    guidance_scale=2.0,
                    output_type="pil",
                )
            edited = out.images[0]
            del guidance, prompt_processor
            torch.cuda.empty_cache()

            fp = save_image(edited, out_dir, "irfds_sd35",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, "irfds_sd35", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"iRFDS SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "irfds_sd35", rows)
    log.info("iRFDS (SD 3.5) done.")

def run_splitflow_sd35(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow on stabilityai/stable-diffusion-3.5-large.
    LLM: mistralai/Mistral-7B-Instruct-v0.3
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("splitflow_sd35")
    log.info(f"Starting SplitFlow (SD 3.5) on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="SplitFlow (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, edit_guidance_scale=13.5,
                n_min=0, n_max=33,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, "splitflow_sd35",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, "splitflow_sd35", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"SplitFlow SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "splitflow_sd35", rows)
    log.info("SplitFlow (SD 3.5) done.")

def run_splitflow_sd35_conflictaware_cosine(pairs: List[Dict], gpu_id: int, out_dir: Path,
                                            kappa_tar: float = 0.7, expr_name: str = "splitflow_sd35_conflictaware_cosine"):
    """SplitFlow on stabilityai/stable-diffusion-3.5-large.
    LLM: mistralai/Mistral-7B-Instruct-v0.3
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting SplitFlow (SD 3.5) on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3_ConflictAware_Cosine
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="SplitFlow (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3_ConflictAware_Cosine(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale_base=3.5, tar_guidance_scale_base=13.5,
                kappa_tar=kappa_tar,
                n_max=33,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name,
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"SplitFlow SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "splitflow_sd35", rows)
    log.info("SplitFlow (SD 3.5) done.")

def run_splitflow_sd35_conflictaware_relative(pairs: List[Dict], gpu_id: int, out_dir: Path,
                                            kappa_tar: float = 0.7, expr_name: str = "splitflow_sd35_conflictaware_relative"):
    """SplitFlow on stabilityai/stable-diffusion-3.5-large.
    LLM: mistralai/Mistral-7B-Instruct-v0.3
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting SplitFlow (SD 3.5) on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3_ConflictAware_Relative
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="SplitFlow (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3_ConflictAware_Relative(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale_base=3.5, tar_guidance_scale_base=13.5,
                kappa_tar=kappa_tar,
                n_max=33,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name,
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"SplitFlow SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "splitflow_sd35", rows)
    log.info("SplitFlow (SD 3.5) done.")

# ── CFG Scheduler helpers ────────────────────────────────────────────────────

def _calc_v_sd3_cfg_zero(pipe, model_input, prompt_embeds, pooled_embeds,  # noqa: N802
                         guidance_scale, t, step_index, zero_init_steps=2):
    """
    SD3.5 transformer call with CFG-Zero* guidance [Fan et al., arXiv:2503.18886].

    model_input is a 2-item batch [uncond_latent, cond_latent].

    step_index < zero_init_steps:
        Returns torch.zeros_like(model_input[:half_batch]) — exactly as in the
        reference implementation — without calling the transformer. This avoids
        the large first-step guidance error when starting from pure noise t≈1.0.
    step_index >= zero_init_steps:
        Runs the transformer, splits [v_uncond, v_cond], applies star correction
        (s*) + CFG guidance via star_correction() from schedulers.py.
    """
    half_batch = model_input.shape[0] // 2
    if step_index < zero_init_steps:
        return torch.zeros_like(model_input[:half_batch])

    timestep = t.expand(model_input.shape[0])
    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    v_uncond, v_cond = noise_pred.chunk(2)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import star_correction
    return star_correction(v_uncond, v_cond, guidance_scale)


def _run_flowedit_sd35_cfg_zero(pairs, gpu_id, out_dir, expr_name,
                                          zero_init_steps=2):
    """
    FlowEdit SD3.5 with true CFG-Zero* zero-init, starting from pure noise
    (n_max = T_steps = 50) [Fan et al., arXiv:2503.18886].

    Implements the reference FlowEditSD3_CFGZero logic directly:
    - Source and target are processed in SEPARATE 2-item batches [uncond, cond],
      matching the reference script (not the joint 4-item batch of FlowEditSD3).
    - For step_index < zero_init_steps: both vt_src and vt_tar are zero tensors
      (no transformer call) so V_delta = 0 and zt_edit is unchanged.
    - For remaining steps: star correction (s*) + CFG guidance is applied.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    evaluator = MetricsEvaluator(
        device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    T_STEPS = 50
    SRC_CFG = 3.5
    TAR_CFG = 13.5
    N_AVG   = 1

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc=expr_name):
        try:
            fp_check = out_dir / expr_name / f"{pair['base_name']}_{pair['code']}.png"
            if fp_check.exists():
                log.info(f"  [skip] {pair['base_name']} {pair['code']} already exists")
                continue

            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            # ── Set up timesteps ────────────────────────────────────────────
            pipe.scheduler.set_timesteps(T_STEPS, device=device)
            timesteps = pipe.scheduler.timesteps

            # ── Encode prompts → [uncond, cond] pairs ───────────────────────
            pipe._guidance_scale = SRC_CFG
            src_pe, src_npe, src_ppe, src_nppe = pipe.encode_prompt(
                prompt=pair["source_prompt"], prompt_2=None, prompt_3=None,
                negative_prompt="", do_classifier_free_guidance=True,
                device=device)
            pipe._guidance_scale = TAR_CFG
            tar_pe, tar_npe, tar_ppe, tar_nppe = pipe.encode_prompt(
                prompt=pair["target_prompt"], prompt_2=None, prompt_3=None,
                negative_prompt="", do_classifier_free_guidance=True,
                device=device)

            src_embeds = torch.cat([src_npe, src_pe])
            src_pooled = torch.cat([src_nppe, src_ppe])
            tar_embeds = torch.cat([tar_npe, tar_pe])
            tar_pooled = torch.cat([tar_nppe, tar_ppe])
            del src_pe, src_npe, src_ppe, src_nppe
            del tar_pe, tar_npe, tar_ppe, tar_nppe
            torch.cuda.empty_cache()

            # ── FlowEdit denoising loop ─────────────────────────────────────
            zt_edit = x_src.clone()

            for step_i, t in enumerate(timesteps):
                torch.cuda.empty_cache()

                t_curr = (t / 1000.0).float()
                if step_i + 1 < len(timesteps):
                    t_prev = (timesteps[step_i + 1] / 1000.0).float()
                else:
                    t_prev = torch.zeros_like(t_curr)
                dt = t_prev - t_curr  # negative (high→low noise)

                v_delta_avg = torch.zeros_like(x_src)
                for _ in range(N_AVG):
                    noise  = torch.randn_like(x_src)
                    zt_src = (1 - t_curr) * x_src + t_curr * noise
                    zt_tar = zt_edit + zt_src - x_src
                    del noise

                    # Source velocity (2-item batch: [uncond_src, cond_src])
                    model_in_src = torch.cat([zt_src, zt_src])
                    del zt_src
                    vt_src = _calc_v_sd3_cfg_zero(
                        pipe, model_in_src, src_embeds, src_pooled,
                        SRC_CFG, t, step_i, zero_init_steps)
                    del model_in_src

                    # Target velocity (2-item batch: [uncond_tar, cond_tar])
                    model_in_tar = torch.cat([zt_tar, zt_tar])
                    del zt_tar
                    vt_tar = _calc_v_sd3_cfg_zero(
                        pipe, model_in_tar, tar_embeds, tar_pooled,
                        TAR_CFG, t, step_i, zero_init_steps)
                    del model_in_tar

                    v_delta_avg += (vt_tar - vt_src) / N_AVG
                    del vt_src, vt_tar

                zt_edit = zt_edit.to(torch.float32)
                zt_edit = zt_edit + v_delta_avg * dt
                zt_edit = zt_edit.to(v_delta_avg.dtype)
                del v_delta_avg

            # ── Decode ─────────────────────────────────────────────────────
            x_tar_denorm = (zt_edit / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"],
                                       pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows)
    log.info(f"{expr_name} done.")

def _run_flowedit_sd35_with_scheduler(pairs, gpu_id, out_dir, expr_name, cfg_scheduler):
    """Internal: FlowEdit SD3.5 with a cfg_scheduler function."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3_Scheduler
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc=expr_name):
        try:
            fp_check = out_dir / expr_name / f"{pair['base_name']}_{pair['code']}.png"
            if fp_check.exists():
                log.info(f"  [skip] {pair['base_name']} {pair['code']} already exists")
                continue
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3_Scheduler(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                n_min=0, n_max=33,
                cfg_scheduler=cfg_scheduler,
            )
            x_tar_denorm = x_tar / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows)
    log.info(f"{expr_name} done.")


def _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir, expr_name, cfg_scheduler):
    """Internal: SplitFlow SD3.5 with a cfg_scheduler function."""
    import random, re as _re
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3_Scheduler
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc=expr_name):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = _re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3_Scheduler(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, edit_guidance_scale=13.5,
                n_min=0, n_max=33,
                cfg_scheduler=cfg_scheduler,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows)
    log.info(f"{expr_name} done.")


def _run_ftedit_with_scheduler(pairs, gpu_id, out_dir, expr_name, cfg_scheduler):
    """Internal: FTEdit SD3.5 with a cfg_scheduler function."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    ftedit_root = str(METHODS_ROOT / "FTEdit")
    if ftedit_root not in sys.path:
        sys.path.insert(0, ftedit_root)

    from mmdit.sd35_pipeline import StableDiffusion3Pipeline as SD35Pipeline
    from inversion.flow_fixpoint_residual_new import Inversed_flow_fixpoint_residual
    from controller import attn_norm_ctrl_sd35

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = SD35Pipeline.from_pretrained(SD35_MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe.transformer.eval()
    pipe.vae.eval()

    saved_path = str(out_dir / expr_name)
    Path(saved_path).mkdir(parents=True, exist_ok=True)

    invf = Inversed_flow_fixpoint_residual(
        pipe, steps=30, device=device,
        inv_cfg=1.0, recov_cfg=1.0, skip_steps=7,
        saved_path=saved_path,
    )
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc=expr_name):
        try:
            src_img = _safe_load_pil(pair["image_path"], log)
            if src_img is None:
                continue
            _img_path = _find_image(pair["image_path"]) or pair["image_path"]
            prompts = [pair["source_prompt"], pair["target_prompt"]]

            attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, None, None)
            all_latents = invf.euler_flow_inversion(
                prompt=pair["source_prompt"],
                image=_img_path,
                num_fixpoint_steps=3,
                average_step_ranges=(0, 5),
            )

            controller_ada  = attn_norm_ctrl_sd35.Adalayernorm_replace(
                prompts, 30, 1.0,
                pipe.tokenizer, pipe.tokenizer_3, device=device,
            )
            controller_attn = attn_norm_ctrl_sd35.SD3attentionreplace(prompts, 30, 1.0)
            attn_norm_ctrl_sd35.register_attention_control_sd35(
                pipe, controller_attn, controller_ada
            )

            _image1, image2 = invf.edit_img_with_residual(
                prompts, all_latents, controller_ada,
                cfg_scheduler=cfg_scheduler,
            )

            if isinstance(image2, np.ndarray):
                arr = np.squeeze(image2)
                if arr.dtype != np.uint8:
                    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
                edited = Image.fromarray(arr)
            else:
                edited = image2

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  [{expr_name}] {pair['base_name']} {pair['code']}  CLIP={m['CLIP_whole']:.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows)
    log.info(f"{expr_name} done.")


# ── CFG Scheduler runners ────────────────────────────────────────────────────

def run_flowedit_sd35_interval(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FlowEdit SD3.5 + Interval CFG scheduler [Kynkäänniemi et al., NeurIPS 2024]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_interval
    _run_flowedit_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                      "flowedit_sd35_interval", scheduler_interval)

def run_flowedit_sd35_monotone(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FlowEdit SD3.5 + Monotone cosine-decay CFG scheduler [Wang et al., 2404.13040]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_monotone
    _run_flowedit_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                      "flowedit_sd35_monotone", scheduler_monotone)

def run_flowedit_sd35_zeroinit(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """
    FlowEdit SD3.5 + CFG-Zero* with true zero-init [Fan et al., arXiv:2503.18886].
    Implements the reference loop directly (separate 2-item batches, torch.zeros_like
    skip for first 2 steps). See _run_flowedit_sd35_cfg_zero.
    """
    _run_flowedit_sd35_cfg_zero(pairs, gpu_id, out_dir,
                                "flowedit_sd35_zeroinit", zero_init_steps=2)

def run_splitflow_sd35_interval(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow SD3.5 + Interval CFG scheduler [Kynkäänniemi et al., NeurIPS 2024]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_interval
    _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                       "splitflow_sd35_interval", scheduler_interval)

def run_splitflow_sd35_monotone(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow SD3.5 + Monotone cosine-decay CFG scheduler [Wang et al., 2404.13040]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_monotone
    _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                       "splitflow_sd35_monotone", scheduler_monotone)

def run_splitflow_sd35_zeroinit(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow SD3.5 + CFG-Zero* (zero-init + star correction) [Fan et al., 2503.18886]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_zero_init
    _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                       "splitflow_sd35_zeroinit", scheduler_zero_init)

def run_ftedit_interval(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FTEdit SD3.5 + Interval CFG scheduler [Kynkäänniemi et al., NeurIPS 2024]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_interval
    _run_ftedit_with_scheduler(pairs, gpu_id, out_dir, "ftedit_interval", scheduler_interval)

def run_ftedit_monotone(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FTEdit SD3.5 + Monotone cosine-decay CFG scheduler [Wang et al., 2404.13040]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_monotone
    _run_ftedit_with_scheduler(pairs, gpu_id, out_dir, "ftedit_monotone", scheduler_monotone)

def run_ftedit_zeroinit(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FTEdit SD3.5 + CFG-Zero* (zero-init + star correction) [Fan et al., 2503.18886]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_zero_init
    _run_ftedit_with_scheduler(pairs, gpu_id, out_dir, "ftedit_zeroinit", scheduler_zero_init)

def compute_fid(out_dir: Path, methods: List[str],
                source_imgs: List[str]) -> Dict[str, float]:
    """
    FID between the full set of edited images and the full set of source images.
    Uses clean-fid with mode='clean', images resized to 256×256.
    """
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("[FID] clean-fid not found — skipping.")
        return {}

    import tempfile, shutil

    ref_dir = out_dir / "_fid_ref"
    ref_dir.mkdir(exist_ok=True)
    valid_src = 0
    for p in source_imgs:
        try:
            resolved = _find_image(p)
            if resolved is None:
                print(f"[FID] source image not found, skipping: {p}")
                continue
            dst = ref_dir / Path(resolved).name
            if not dst.exists():
                img = Image.open(resolved).convert("RGB").resize((256, 256), Image.LANCZOS)
                img.save(dst)
            valid_src += 1
        except Exception as e:
            print(f"[FID] could not load source image {p}: {e}")
    if valid_src < 2:
        print(f"[FID] skipped — not enough source images ({valid_src} < 2)")
        return {}

    fid_scores = {}
    for method in methods:
        edited_folder = out_dir / method
        if not edited_folder.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            for f in edited_folder.glob("*.png"):
                img = Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)
                img.save(tmp_p / f.name)
            if not list(tmp_p.glob("*.png")):
                continue
            try:
                score = cleanfid.compute_fid(str(tmp_p), str(ref_dir),
                                             mode="clean", verbose=False)
                fid_scores[method] = score
            except Exception as e:
                print(f"[FID] {method}: {e}")

    return fid_scores

METRICS = ["CLIP_whole", "CLIP_edited", "LPIPS", "MSE", "PSNR", "SSIM", "StructDist"]

def merge_csvs(out_dir: Path) -> Path:
    all_rows = []
    for f in sorted(f for f in out_dir.glob("results_*.csv") if f.name != "results_merged.csv"):
        with open(f) as fh:
            all_rows.extend(list(csv.DictReader(fh)))
    merged = out_dir / "results_merged.csv"
    if all_rows:
        with open(merged, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_rows[0].keys(),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
    return merged

def print_summary(merged_csv: Path, fid_scores: Dict[str, float]):
    import io
    try:
        rows = []
        with open(merged_csv) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print("[Summary] merged CSV not found.")
        return

    from collections import defaultdict
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnts: Dict[str, int]              = defaultdict(int)
    for r in rows:
        m = r["method"]
        cnts[m] += 1
        for col in METRICS:
            try:
                sums[m][col] += float(r[col])
            except (ValueError, KeyError):
                pass

    methods = sorted(cnts.keys())
    avgs = {m: {c: sums[m][c] / cnts[m] for c in METRICS} for m in methods}

    col_w = 12
    hdr   = f"{'Method':<18}" + "".join(f"{c:>{col_w}}" for c in METRICS) + f"{'FID':>{col_w}}"
    sep   = "─" * len(hdr)
    print("\n" + sep)
    print("  BENCHMARK SUMMARY")
    print(sep)
    print(hdr)
    print(sep)
    for m in methods:
        row = f"{m:<18}"
        for c in METRICS:
            row += f"{avgs[m].get(c, float('nan')):>{col_w}.3f}"
        fid = fid_scores.get(m, float("nan"))
        row += f"{fid:>{col_w}.2f}"
        print(row)
    print(sep + "\n")

# _RUNNERS = [
#     # idx  function                  default GPU
#     (run_new_ddim_sd14,       0),   # 0  DDIM+P2P / DDIM+PnP   (SD1.4, ~4GB)
#     (run_nulltext_sd21,       1),   # 1  Null-Text              (SD2.1, ~8GB)
#     (run_pnpinv_p2p_sd14,    2),   # 2  PnP-Inv P2P            (SD1.4, ~4GB)
#     (run_flowedit_sd35,       3),   # 3  FlowEdit               (SD3.5, ~30GB)
#     (run_ftedit_only,         4),   # 4  FTEdit                 (SD3.5, ~30GB)
#     (run_pnpinv_pnp_sd14,    5),   # 5  PnP-Inv PnP/SDE        (SD1.4, ~4GB)
#     (run_splitflow_sd35,      6),   # 6  SplitFlow              (SD3.5, ~30GB)
#     (run_flowedit,            0),   # 7  FlowEdit               (FLUX.1-dev, ~30GB)
#     (run_fireflow,            1),   # 8  FireFlow               (FLUX.1-dev, ~30GB)
#     (run_rf_inversion,        2),   # 9  RF-Inversion           (FLUX.1-dev, ~30GB)
#     (run_rf_solver,           3),   # 10 RF-Solver              (FLUX.1-dev, ~30GB)
#     (run_irfds_sd35,          5),   # 11 iRFDS                  (SD3.5, very slow)
#     (run_cag,                 7),   # 12 CAG/Annealing Guidance (SD-based)
#     (run_irfds,               6),   # 13 iRFDS                  (FLUX, very slow)
# ]

# _METHOD_NAMES = [
#     ["ddim_p2p", "ddim_pnp"],   # 0
#     ["null_text_sd14"],          # 1
#     ["pnpinv_p2p_sd14"],         # 2
#     ["flowedit_sd35"],           # 3
#     ["ftedit"],                  # 4
#     ["pnpinv_pnp_sd14"],         # 5
#     ["splitflow_sd35"],          # 6
#     ["flowedit_flux"],           # 7
#     ["fireflow"],                # 8
#     ["rf_inversion"],            # 9
#     ["rf_solver"],               # 10
#     ["irfds_sd35"],              # 11
#     ["cag"],                     # 12
#     ["irfds_flux"],              # 13
# ]
# ---------- Baselines -----------
_RUNNERS = [
    # function  default GPU
    # (run_splitflow_sd35,      0),
    # (run_flowedit_sd35,       1),
    # (run_irfds_sd35, 2),
    (run_flowedit_sd35_interval, 0),
    (run_flowedit_sd35_monotone, 1),
    (run_flowedit_sd35_zeroinit, 2),
    (run_splitflow_sd35_interval, 3),
    (run_splitflow_sd35_monotone, 4),
    (run_splitflow_sd35_zeroinit, 5),
]

_METHOD_NAMES = [
    # ["splitflow_sd35"],
    # ["flowedit_sd35"],
    # ["irfds_sd35"],
    ["flowedit_sd35_interval"],
    ["flowedit_sd35_monotone"],
    ["flowedit_sd35_zeroinit"],
    ["splitflow_sd35_interval"],
    ["splitflow_sd35_monotone"],
    ["splitflow_sd35_zeroinit"],
]
# ---------- Proposed -----------
_RUNNERS = [
    # function  default GPU
    (run_flowedit_sd35_conflictaware_cosine,       0),
    (run_flowedit_sd35_conflictaware_relative,      1),
    (run_splitflow_sd35_conflictaware_cosine,       2),
    (run_splitflow_sd35_conflictaware_relative,     3),
]

_METHOD_NAMES = [
    ["flowedit_sd35_conflictaware_cosine"],
    ["flowedit_sd35_conflictaware_relative"],
    ["splitflow_sd35_conflictaware_cosine"],
    ["splitflow_sd35_conflictaware_relative"],
]

# ---------- Test -----------
# _RUNNERS = [
#     # function  default GPU
#     # (run_splitflow_sd35,      0), # Done by Yan, DVI2K and PIE
#     # (run_flowedit_sd35,       1), # Done by Yan, DVI2K and PIE
#     # (run_irfds_sd35, 0),          # Not start yet due to bug
#     # (run_pnpinv_p2p_sd14, 3)      # Not start yet due to bug
#     (run_flowedit_sd35_interval, 0), # Code is ready, not start
#     (run_flowedit_sd35_monotone, 1), # Code is ready, not start
#     (run_splitflow_sd35_interval, 2), # Code is ready, not start
#     (run_splitflow_sd35_monotone, 3), # Code is ready, not start
# ]

# _METHOD_NAMES = [
#     # ["splitflow_sd35"],
#     # ["flowedit_sd35"],
#     # ["irfds_sd35"],
#     # ["pnpinv_p2p_sd14"],
#     ["flowedit_sd35_interval"],
#     ["flowedit_sd35_monotone"],
#     ["splitflow_sd35_interval"],
#     ["splitflow_sd35_monotone"],
# ]
# ---------------------------
def _worker(fn, pairs, gpu_id, out_dir_str):
    """Top-level function for multiprocessing.spawn."""
    out_dir = Path(out_dir_str)
    try:
        fn(pairs, gpu_id, out_dir)
    except Exception:
        logging.getLogger(fn.__name__).error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Benchmark image-editing methods")
    parser.add_argument("--yaml",      default="Data/flowedit.yaml")
    parser.add_argument("--images",    default="Data/flowedit_data/")
    parser.add_argument("--outdir",    default=None)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs (e.g. 2 for testing)")
    parser.add_argument("--methods",   nargs="*", default=None,
                        help="Run only specific runners (0-6). "
                             "e.g. --methods 0 3 6 to run DDIM + FlowEdit + SplitFlow")
    parser.add_argument("--pie_bench",   action="store_true",
                        help="Use PIE-Bench dataset instead of FlowEdit YAML")
    parser.add_argument("--pie_mapping", default="Data/PIE-Bench_v1/mapping_file.json",
                        help="Path to PIE-Bench mapping_file.json")
    parser.add_argument("--pie_images",  default="Data/PIE-Bench_v1/annotation_images",
                        help="Path to PIE-Bench annotation_images directory")
    parser.add_argument("--gpu_map", nargs="*", type=int, default=None,
                        help="Override GPU IDs for selected runners (one per --methods entry). "
                             "e.g. --gpu_map 0 0 4 0 5")
    parser.add_argument("--monotonic_alpha", type=float, default=1.0,
                        help="Limit number of pairs (e.g. 2 for testing)")
    parser.add_argument("--monotonic_beta", type=float, default=1.0,
                        help="Limit number of pairs (e.g. 2 for testing)")

    args = parser.parse_args()

    if args.pie_bench:
        mapping_file = str(ROOT / args.pie_mapping)
        images_root  = str(ROOT / args.pie_images)
        pairs = load_pairs_pie(mapping_file, images_root, args.max_pairs)
        print(f"Output directory : {args.outdir or 'auto'}")
        print(f"PIE-Bench mapping: {mapping_file}")
        print(f"Images root      : {images_root}")
    else:
        yaml_path   = str(ROOT / args.yaml)
        images_root = str(ROOT / args.images)
        pairs = load_pairs(yaml_path, images_root, args.max_pairs)
        print(f"Output directory : {args.outdir or 'auto'}")
        print(f"YAML             : {yaml_path}")
        print(f"Images root      : {images_root}")

    if args.outdir is None:
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag     = "pie" if args.pie_bench else "flowedit"
        out_dir = ROOT / "outputs" / f"benchmark_{tag}_{stamp}"
    else:
        out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {out_dir}")
    print(f"Pairs            : {len(pairs)}")

    with open(out_dir / "config.json", "w") as f:
        json.dump({"pie_bench": args.pie_bench, "images": images_root,
                   "max_pairs": args.max_pairs, "num_pairs": len(pairs)}, f, indent=2)

    runner_indices = list(range(len(_RUNNERS)))
    if args.methods is not None:
        runner_indices = [int(x) for x in args.methods]

    # Build effective GPU IDs (--gpu_map overrides _RUNNERS defaults)
    gpu_ids = [_RUNNERS[idx][1] for idx in runner_indices]
    if args.gpu_map is not None:
        if len(args.gpu_map) != len(runner_indices):
            raise ValueError(f"--gpu_map has {len(args.gpu_map)} entries but "
                             f"{len(runner_indices)} runners selected")
        gpu_ids = args.gpu_map

    ctx = mp.get_context("spawn")
    procs = []
    for idx, gpu_id in zip(runner_indices, gpu_ids):
        fn, _ = _RUNNERS[idx]
        p = ctx.Process(
            target=_worker,
            args=(fn, pairs, gpu_id, str(out_dir)),
            name=fn.__name__,
        )
        p.start()
        procs.append(p)
        print(f"  Launched {fn.__name__} → GPU {gpu_id}  (pid={p.pid})")

    for p in procs:
        p.join()
        status = "OK" if p.exitcode == 0 else f"EXIT={p.exitcode}"
        print(f"  {p.name} finished [{status}]")

    merged = merge_csvs(out_dir)
    print(f"Merged CSV       : {merged}")

    all_methods = [m for idx in runner_indices for m in _METHOD_NAMES[idx]]
    unique_source = list({p["image_path"] for p in pairs})
    print("Computing FID…")
    fid_scores = compute_fid(out_dir, all_methods, unique_source)
    if fid_scores:
        fid_out = out_dir / "fid_scores.json"
        with open(fid_out, "w") as f:
            json.dump(fid_scores, f, indent=2)
        print(f"FID scores       : {fid_out}")

    print_summary(merged, fid_scores)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
