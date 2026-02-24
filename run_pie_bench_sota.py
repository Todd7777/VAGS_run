"""
benchmark_full_cag.py
══════════════════════════════════════════════════════════════════════════════
FlowEdit + Conflict-Aware Guidance (CAG) — full PIE-Bench evaluation.

Strategies (all from ECCV Formulations doc):
  0. Baseline           — constant λ = λ_base
  1. Exp_Forward_Gate   — Eq. 4 + 6, smooth sigmoid gate
  2. Exp_Forward        — §1.5, no gate (t_start=1 → g = 1−t)
  3. Exp_Reverse_Gate   — Eq. 5 + 6, reversed gate g_rev
  4. Exp_Boost          — Eq. 7-8, conflict boost
  5. TwoPhase           — Eq. 9-10, piecewise envelope h(t)
  6. Entropy            — Eq. 11-13, entropy-aware gate
  7. Locality           — Eq. 14-15, locality-aware (uses edit mask)
  8. Dual               — Eq. 16-19, dual-objective q(t, Δ)
  9. Quantile           — Eq. 20-21, quantile-ranked conflict
 10. Budget_Matched     — Eq. 22-23, Exp_Forward_Gate with κ tuned to match
                          baseline guidance budget

Metrics:
  StructDist×10³↓  PSNR↑  LPIPS×10³↓  MSE×10⁴↓  SSIM×10²↑
  CLIP-T-Whole↑  CLIP-T-Edited↑  CLIP-I↑  FID↓  Lambda_mean

Usage:
  python benchmark_full_cag.py

Dependencies:
  pip install diffusers transformers torch torchvision lpips scikit-image \
              scipy accelerate sentencepiece
"""

import os, gc, json, csv, math, random, warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import lpips as lpips_lib
from torchvision import transforms
from torchvision.models import inception_v3
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 0.  REPRODUCIBILITY & DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS & HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PIE_BENCH_ROOT = "Data/PIE-Bench_v1/annotation_images"
MASK_ROOT      = "Data/PIE-Bench_v1/annotation_masks"   # optional
MAPPING_FILE   = "Data/PIE-Bench_v1/mapping_file.json"
OUTPUT_DIR     = "outputs/PIE_BENCH_FULL_CAG"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── shared hyperparameters — kept identical to your working script ────────────
HP = dict(
    t_start    = 0.66,   # trajectory start time
    cfg_src    = 3.5,    # source CFG (fixed)
    base_cfg   = 13.5,   # λ_0 / λ_base
    kappa      = 4.0,    # modulation strength κ
    m          = 3.0,    # tanh scale m
    rho        = 0.5,    # conflict boost coefficient ρ  (Eq. 7)
    ta         = 0.5,    # TwoPhase upper boundary t_a  (Eq. 9)
    tb         = 0.2,    # TwoPhase lower boundary t_b  (Eq. 9)
    gamma      = 0.5,    # locality weight γ             (Eq. 14)
    lambda_min = 1.0,
    lambda_max = 40.0,
    steps      = 50,
    img_size   = 1024,
)

# Budget-matched variant: κ pre-tuned so mean(λ) ≈ base_cfg.
# Adjust empirically on your dataset; 2.1 is a reasonable starting point.
HP_BUDGET = {**HP, "kappa": 2.1}

ALL_STRATEGIES = [
    "Baseline",
    "Exp_Forward_Gate",
    "Exp_Forward",
    "Exp_Reverse_Gate",
    "Exp_Boost",
    "TwoPhase",
    "Entropy",
    "Locality",
    "Dual",
    "Quantile",
    "Budget_Matched",
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_pie_bench():
    image_index = {}
    for root, _, files in os.walk(PIE_BENCH_ROOT):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                stem = os.path.splitext(f)[0]
                image_index[stem] = os.path.join(root, f)

    with open(MAPPING_FILE) as fp:
        data = json.load(fp)

    dataset = []
    for key, ann in data.items():
        stem = os.path.splitext(os.path.basename(key))[0]
        if stem not in image_index:
            continue
        full_path = image_index[stem]
        category  = os.path.basename(os.path.dirname(full_path))

        mask_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(MASK_ROOT, category, stem + ext)
            if os.path.exists(cand):
                mask_path = cand
                break

        dataset.append(dict(
            full_path = full_path,
            base_name = stem,
            category  = category,
            source    = ann.get("original_prompt", ""),
            target    = ann.get("editing_prompt", ""),
            mask_path = mask_path,
        ))
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GUIDANCE SCHEDULER  ── all CAG strategies from ECCV doc
# ══════════════════════════════════════════════════════════════════════════════

class GuidanceScheduler:
    """
    Stateful per-image scheduler.
    Call .reset() before each new image/strategy pair.
    Call .get_lambda(t, vc_s, vc_t) at every diffusion step.
    """

    def __init__(self, strategy: str, hp: dict):
        self.strategy = strategy
        self.hp       = hp
        self._conflict_buf: list = []   # Δ_k history
        self._entropy_buf:  list = []   # H_k history
        self._lambda_buf:   list = []   # λ_k history (for budget check)

    def reset(self):
        self._conflict_buf.clear()
        self._entropy_buf.clear()
        self._lambda_buf.clear()

    # ── private helpers ───────────────────────────────────────────────────────

    def _delta(self, vc_s: torch.Tensor, vc_t: torch.Tensor) -> float:
        """Eq. 2 — normalised RMS conflict ‖Δ‖ / ‖v_src‖."""
        diff  = vc_t - vc_s
        n_d   = torch.norm(diff,  p=2).item()
        n_s   = torch.norm(vc_s, p=2).item()
        return n_d / (n_s + 1e-5)

    def _s(self, delta: float, m: float = None) -> float:
        """Eq. 3 — tanh saturation s(Δ_k) = tanh(Δ_k / m)."""
        return math.tanh(delta / (m or self.hp["m"]))

    def _gate(self, t: float) -> float:
        """Eq. 4 — linear late-stage gate g(t) ∈ [0, 1].
           g = 0 at t = t_start  (early/noisy)
           g = 1 at t = 0        (late/clean)
        """
        ts = self.hp["t_start"]
        return float(np.clip((ts - t) / ts, 0.0, 1.0))

    def _sig_gate(self, t: float, k: float = 12.0) -> float:
        """Smooth sigmoid version of g(t), avoids hard cliff."""
        p = self._gate(t)   # linear progress ∈ [0, 1]
        return 1.0 / (1.0 + math.exp(-k * (p - 0.5)))

    def _exp_core(self, s_val: float, g_val: float, kappa: float = None) -> float:
        """Eq. 6 — λ = clip(λ_0 · exp(κ · s · (2g−1)), λ_min, λ_max)."""
        hp    = self.hp
        kappa = kappa or hp["kappa"]
        lam   = hp["base_cfg"] * math.exp(kappa * s_val * (2.0 * g_val - 1.0))
        return float(np.clip(lam, hp["lambda_min"], hp["lambda_max"]))

    # ── main dispatch ─────────────────────────────────────────────────────────

    def get_lambda(
        self,
        t:        float,
        vc_s:     torch.Tensor,
        vc_t:     torch.Tensor,
        locality: float = 0.5,   # ℓ ∈ [0, 1] inverse mask area
    ) -> float:

        hp  = self.hp
        stg = self.strategy

        # compute conflict for all dynamic strategies
        delta = self._delta(vc_s, vc_t)
        self._conflict_buf.append(delta)

        # ── 0. Baseline ───────────────────────────────────────────────────────
        if stg == "Baseline":
            lam = hp["base_cfg"]

        # ── 1. Exp_Forward_Gate  (Eq. 4 + 6, smooth sigmoid gate) ────────────
        elif stg == "Exp_Forward_Gate":
            lam = self._exp_core(self._s(delta), self._sig_gate(t))

        # ── 2. Exp_Forward  (§1.5: t_start=1 ⇒ g(t) = 1−t) ──────────────────
        elif stg == "Exp_Forward":
            lam = self._exp_core(self._s(delta), 1.0 - t)

        # ── 3. Exp_Reverse_Gate  (Eq. 5 + 6, g_rev = 1 − g) ─────────────────
        elif stg == "Exp_Reverse_Gate":
            g_rev = 1.0 - self._sig_gate(t)             # Eq. 5
            lam   = self._exp_core(self._s(delta), g_rev)

        # ── 4. Exp_Boost  (Eq. 7-8: conflict boost) ──────────────────────────
        elif stg == "Exp_Boost":
            buf    = self._conflict_buf
            median = float(np.median(buf)) if len(buf) > 1 else delta
            b      = 1.0 + hp["rho"] * float(delta > median)  # Eq. 7
            dp     = b * delta                                  # Eq. 8  Δ'_k
            lam    = self._exp_core(self._s(dp), self._sig_gate(t))

        # ── 5. TwoPhase  (Eq. 9-10: piecewise envelope h(t)) ─────────────────
        elif stg == "TwoPhase":
            ta, tb = hp["ta"], hp["tb"]
            if   t >= ta: h = 0.0
            elif t <= tb: h = 1.0
            else:         h = (ta - t) / (ta - tb)            # Eq. 9
            lam = self._exp_core(self._s(delta), h)            # Eq. 10

        # ── 6. Entropy  (Eq. 11-13: entropy-aware gate) ──────────────────────
        elif stg == "Entropy":
            # proxy: variance of the target velocity field
            H_k = float(vc_t.var().item())
            self._entropy_buf.append(H_k)
            arr   = np.array(self._entropy_buf)
            med_H = float(np.median(arr))
            mad_H = float(np.median(np.abs(arr - med_H))) + 1e-6
            u_k   = 1.0 / (1.0 + math.exp(-((H_k - med_H) / mad_H)))  # Eq. 11  σ(·)
            g_ent = self._sig_gate(t) * (1.0 - u_k)                    # Eq. 12
            lam   = self._exp_core(self._s(delta), g_ent)              # Eq. 13

        # ── 7. Locality  (Eq. 14-15: locality-aware) ─────────────────────────
        elif stg == "Locality":
            s_loc = math.tanh(delta / hp["m"] * (1.0 + hp["gamma"] * locality))  # Eq. 14
            lam   = self._exp_core(s_loc, self._sig_gate(t))                      # Eq. 15

        # ── 8. Dual  (Eq. 16-19: dual-objective) ─────────────────────────────
        elif stg == "Dual":
            s_e = self._s(delta)                                    # Eq. 16
            s_p = 1.0 - s_e                                         # Eq. 17
            w   = self._sig_gate(t)                                 # ω(t) = g(t)
            q   = w * s_e - (1.0 - w) * s_p                        # Eq. 18
            lam = hp["base_cfg"] * math.exp(hp["kappa"] * q)       # Eq. 19
            lam = float(np.clip(lam, hp["lambda_min"], hp["lambda_max"]))

        # ── 9. Quantile  (Eq. 20-21: quantile-ranked conflict) ───────────────
        elif stg == "Quantile":
            buf = np.array(self._conflict_buf)
            r_k = float((buf < delta).mean())   # empirical quantile rank ∈ [0, 1]
            s_q = 2.0 * r_k - 1.0              # Eq. 20  ∈ [-1, 1]
            lam = self._exp_core(s_q, self._sig_gate(t))

        # ── 10. Budget_Matched  (Eq. 22: same as Exp_Forward_Gate, κ tuned) ──
        elif stg == "Budget_Matched":
            lam = self._exp_core(self._s(delta), self._sig_gate(t),
                                 kappa=HP_BUDGET["kappa"])

        else:
            raise ValueError(f"Unknown strategy: {stg!r}")

        self._lambda_buf.append(lam)
        return lam

    def mean_lambda(self) -> float:
        """Mean λ over full trajectory — guidance budget B (Eq. 22)."""
        return float(np.mean(self._lambda_buf)) if self._lambda_buf else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  METRIC EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class MetricEvaluator:
    def __init__(self, device):
        self.device    = device
        self.to_tensor = transforms.ToTensor()

        print("Loading CLIP (ViT-L/14) …")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(device).eval()
        self.clip_proc  = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")

        print("Loading LPIPS (VGG) …")
        self.lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()

        self._inception = None   # lazy init on first FID call

    # ── CLIP-T (image vs. target text) ───────────────────────────────────────
    def clip_t(self, image: Image.Image, prompt: str) -> float:
        """Whole-image CLIP-T score (matches PIE-Bench 'Whole' column)."""
        inputs = self.clip_proc(
            text=[prompt], images=image, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            out = self.clip_model(**inputs)
        # logits_per_image = 100 × cosine → return as-is to match PIE-Bench ×100 scale
        return out.logits_per_image.item()

    def clip_t_masked(
        self, image: Image.Image, prompt: str, mask: Image.Image
    ) -> float:
        """CLIP-T over edited region only (PIE-Bench 'Edited' column)."""
        if mask is None:
            return float("nan")
        mask_arr = np.array(mask.resize(image.size).convert("L")) > 127
        if mask_arr.sum() == 0:
            return float("nan")
        ys, xs = np.where(mask_arr)
        crop   = image.crop((xs.min(), ys.min(), xs.max(), ys.max()))
        return self.clip_t(crop, prompt)

    # ── CLIP-I (image vs. source image) ──────────────────────────────────────
    def clip_i(self, src: Image.Image, gen: Image.Image) -> float:
        """Cosine similarity between CLIP embeddings of source and edited image."""
        inputs = self.clip_proc(images=[src, gen], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.clip_model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        return float((feats[0] * feats[1]).sum().item())

    # ── LPIPS ─────────────────────────────────────────────────────────────────
    def lpips_dist(self, src: Image.Image, gen: Image.Image) -> float:
        """LPIPS perceptual distance ×10³ (matches PIE-Bench scale)."""
        t_s = self.to_tensor(src).to(self.device) * 2 - 1
        t_g = self.to_tensor(gen).to(self.device) * 2 - 1
        with torch.no_grad():
            d = self.lpips_fn(t_s.unsqueeze(0), t_g.unsqueeze(0))
        return d.item() * 1e3

    # ── Pixel-level metrics ───────────────────────────────────────────────────
    def pixel_metrics(self, src: Image.Image, gen: Image.Image):
        """Returns (MSE×10⁴, PSNR, SSIM×10², StructDist×10³)."""
        s = np.array(src.resize((512, 512))).astype(np.float32) / 255.0
        g = np.array(gen.resize((512, 512))).astype(np.float32) / 255.0

        mse  = float(np.mean((s - g) ** 2)) * 1e4
        psnr = float(peak_signal_noise_ratio(s, g, data_range=1.0))
        ssim = float(structural_similarity(
            s, g, data_range=1.0, channel_axis=2, win_size=7)) * 100.0

        # Structure Distance (edge-gradient based background metric)
        kx     = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        lum_w  = np.array([0.299, 0.587, 0.114])
        ls, lg = s @ lum_w, g @ lum_w
        gxs, gys = convolve(ls, kx), convolve(ls, kx.T)
        gxg, gyg = convolve(lg, kx), convolve(lg, kx.T)
        gs = np.sqrt(gxs**2 + gys**2 + 1e-8)
        gg = np.sqrt(gxg**2 + gyg**2 + 1e-8)
        sd = float(np.mean(np.abs(gs - gg) / (gs + gg + 1e-6))) * 1e3

        return mse, psnr, ssim, sd

    # ── Inception features for FID ────────────────────────────────────────────
    @torch.no_grad()
    def inception_feat(self, img: Image.Image) -> np.ndarray:
        if self._inception is None:
            print("Loading Inception-v3 (FID) …")
            self._inception = inception_v3(
                weights="IMAGENET1K_V1", transform_input=False
            ).to(self.device).eval()
            self._inception.fc = torch.nn.Identity()
            self._inc_tfm = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        t = self._inc_tfm(img).unsqueeze(0).to(self.device)
        return self._inception(t).squeeze().cpu().numpy()


def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """FID from two (N, 2048) feature matrices (numpy eigendecomposition)."""
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff           = mu_r - mu_f
    vals, vecs     = np.linalg.eigh(sigma_r @ sigma_f)
    sqrtm          = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * sqrtm))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FLOWEDIT PIPELINE  (velocity-difference coupling + CAG scheduler)
# ══════════════════════════════════════════════════════════════════════════════

class FlowEditPipeline:
    def __init__(self):
        print("Loading SD3-medium …")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=DTYPE,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)

    @torch.no_grad()
    def edit(
        self,
        init_image: Image.Image,
        src_prompt: str,
        tgt_prompt: str,
        strategy:   str,
        locality:   float = 0.5,
    ) -> tuple:
        """Returns (edited_image, mean_lambda)."""

        hp = HP_BUDGET if strategy == "Budget_Matched" else HP

        # ── encode source image ──────────────────────────────────────────────
        img_t  = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x0_src = (
            self.pipe.vae.encode(img_t).latent_dist.mode()
            - self.pipe.vae.config.shift_factor
        ) * self.pipe.vae.config.scaling_factor
        x0_src = x0_src.to(DTYPE)

        # ── text encodings ────────────────────────────────────────────────────
        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(
            src_prompt, src_prompt, src_prompt, device=DEVICE)
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(
            tgt_prompt, tgt_prompt, tgt_prompt, device=DEVICE)

        # ── timestep schedule ─────────────────────────────────────────────────
        t_start     = hp["t_start"]
        steps       = hp["steps"]
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps   = self.pipe.scheduler.timesteps
        start_index = int((1.0 - t_start) * steps)

        sched = GuidanceScheduler(strategy, hp)
        sched.reset()

        zt  = x0_src.clone()
        gen = torch.Generator(device=DEVICE).manual_seed(42)

        # ── denoising loop ────────────────────────────────────────────────────
        for i, t_tensor in enumerate(timesteps):
            if i < start_index:
                continue

            t  = t_tensor.item() / 1000.0
            dt = (
                timesteps[i + 1].item() / 1000.0 if i + 1 < len(timesteps) else 0.0
            ) - t

            noise  = torch.randn(*x0_src.shape, generator=gen,
                                 device=DEVICE, dtype=x0_src.dtype)
            zt_src = (1 - t) * x0_src + t * noise
            zt_tar = zt + zt_src - x0_src

            latents_in    = torch.cat([zt_src] * 2 + [zt_tar] * 2)
            prompt_embeds = torch.cat([ne_s, pe_s, ne_t, pe_t])
            pooled_embeds = torch.cat([npe_s, ppe_s, npe_t, ppe_t])

            pred = self.pipe.transformer(
                hidden_states         = latents_in,
                timestep              = t_tensor.expand(latents_in.shape[0]),
                encoder_hidden_states = prompt_embeds,
                pooled_projections    = pooled_embeds,
                return_dict           = False,
            )[0]

            vu_s, vc_s, vu_t, vc_t = pred.chunk(4)

            # ── dynamic guidance (core CAG step) ────────────────────────────
            lam_t = sched.get_lambda(t, vc_s, vc_t, locality=locality)

            v_src = vu_s + hp["cfg_src"] * (vc_s - vu_s)
            v_tgt = vu_t + lam_t         * (vc_t - vu_t)
            zt    = zt   + dt            * (v_tgt - v_src)

        # ── decode ────────────────────────────────────────────────────────────
        zt_dec = (zt / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        img_t  = self.pipe.vae.decode(zt_dec.to(torch.float32), return_dict=False)[0]
        img_t  = torch.clamp(img_t, -1.0, 1.0)
        result = self.pipe.image_processor.postprocess(img_t, output_type="pil")[0]

        return result, sched.mean_lambda()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dataset   = load_pie_bench()
    editor    = FlowEditPipeline()
    evaluator = MetricEvaluator(DEVICE)

    csv_path   = os.path.join(OUTPUT_DIR, "full_cag_results.csv")
    fieldnames = [
        "Image", "Category", "Strategy",
        "CLIP_T_Whole",    # image vs target prompt  (PIE-Bench "Whole")
        "CLIP_T_Edited",   # masked crop vs target prompt  (PIE-Bench "Edited")
        "CLIP_I",          # source-vs-edited image cosine similarity
        "LPIPS_x1e3",
        "MSE_x1e4",
        "PSNR_dB",
        "SSIM_x100",
        "StructDist_x1e3",
        "Lambda_mean",     # mean λ over trajectory — guidance budget (Eq. 22)
        "Filename",
    ]

    # Inception feature buffers for FID (filled in-loop, computed at the end)
    real_feats_buf              = []
    fake_feats_buf: dict[str, list] = defaultdict(list)

    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(dataset):
            print(f"[{idx + 1}/{len(dataset)}]  {item['base_name']}", flush=True)

            if not os.path.exists(item["full_path"]):
                continue

            init_img = load_image(item["full_path"]).resize(
                (HP["img_size"], HP["img_size"]))

            # ── edit mask ────────────────────────────────────────────────────
            mask_img   = None
            locality_v = 0.5     # default if no mask
            if item["mask_path"] and os.path.exists(item["mask_path"]):
                mask_img   = Image.open(item["mask_path"]).convert("L")
                mask_arr   = np.array(mask_img) > 127
                # ℓ = inverse mask fraction: global edit → ℓ≈0, local edit → ℓ≈1
                locality_v = 1.0 - float(mask_arr.mean())

            # inception features for real image (shared across all strategies)
            real_feats_buf.append(evaluator.inception_feat(init_img))

            for strat in ALL_STRATEGIES:
                set_seed(42)

                gen_img, lam_mean = editor.edit(
                    init_img,
                    item["source"],
                    item["target"],
                    strat,
                    locality=locality_v,
                )

                # ── metrics ──────────────────────────────────────────────────
                clip_t_whole  = evaluator.clip_t(gen_img, item["target"])
                clip_t_edited = evaluator.clip_t_masked(gen_img, item["target"], mask_img)
                clip_i_val    = evaluator.clip_i(init_img, gen_img)
                lpips_v       = evaluator.lpips_dist(init_img, gen_img)
                mse, psnr, ssim, sd = evaluator.pixel_metrics(init_img, gen_img)

                fake_feats_buf[strat].append(evaluator.inception_feat(gen_img))

                fname = f"{item['base_name']}_{strat}.jpg"
                gen_img.save(os.path.join(OUTPUT_DIR, fname))

                writer.writerow({
                    "Image":            item["base_name"],
                    "Category":         item["category"],
                    "Strategy":         strat,
                    "CLIP_T_Whole":     f"{clip_t_whole:.4f}",
                    "CLIP_T_Edited":    f"{clip_t_edited:.4f}",
                    "CLIP_I":           f"{clip_i_val:.4f}",
                    "LPIPS_x1e3":       f"{lpips_v:.4f}",
                    "MSE_x1e4":         f"{mse:.4f}",
                    "PSNR_dB":          f"{psnr:.4f}",
                    "SSIM_x100":        f"{ssim:.4f}",
                    "StructDist_x1e3":  f"{sd:.4f}",
                    "Lambda_mean":      f"{lam_mean:.4f}",
                    "Filename":         fname,
                })
                fout.flush()

                del gen_img
                torch.cuda.empty_cache()
                gc.collect()

    # ── FID (post-loop, over full dataset) ────────────────────────────────────
    print("\nComputing FID scores …")
    real_arr = np.stack(real_feats_buf)
    fid_scores: dict[str, float] = {}
    for strat, feats in fake_feats_buf.items():
        fid_scores[strat] = compute_fid(real_arr, np.stack(feats))
        print(f"  FID  [{strat:<22}] = {fid_scores[strat]:.2f}")

    # ── summary table ─────────────────────────────────────────────────────────
    df  = pd.read_csv(csv_path)
    num = [
        "CLIP_T_Whole", "CLIP_T_Edited", "CLIP_I",
        "LPIPS_x1e3", "MSE_x1e4", "PSNR_dB", "SSIM_x100",
        "StructDist_x1e3", "Lambda_mean",
    ]
    summary = df.groupby("Strategy")[num].mean().round(4)
    summary["FID"] = pd.Series(fid_scores).round(2)
    summary = summary.reindex([s for s in ALL_STRATEGIES if s in summary.index])

    summary_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
    summary.to_csv(summary_path)

    print("\n" + "═" * 120)
    print("  FULL CAG BENCHMARK  ·  SD3-medium  ·  PIE-Bench")
    print("═" * 120)
    print(summary.to_string())
    print("═" * 120)
    print(f"\n  Per-sample CSV   →  {csv_path}")
    print(f"  Summary table    →  {summary_path}")

    # reference values from the SplitFlow paper (SD3, Table 1)
    print("""
  ─── SotA reference (SD3, Table 1 of SplitFlow paper) ───────────────────────
  Method          StructDist  PSNR   LPIPS×10³  MSE×10⁴  SSIM×10²  CLIP-W  CLIP-E
  FlowEdit SD3      27.24    22.13   105.46      87.34    83.48     26.83   23.67
  SplitFlow SD3     25.96    22.45   102.14      81.99    83.91     26.96   23.83
  ─────────────────────────────────────────────────────────────────────────────
  Note: CLIP-W/CLIP-E in paper are ×100 cosine logits (same scale as CLIP_T_Whole here).
    """)


if __name__ == "__main__":
    main()