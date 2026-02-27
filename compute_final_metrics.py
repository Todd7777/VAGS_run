"""
compute_final_metrics.py
========================
Post-hoc computation of ALL benchmark metrics on generated images.
Run this AFTER the benchmark finishes (or on partial results).

Metrics computed
----------------
  CLIP_whole      – CLIP(edited, target_prompt)         [0-100, matches CLIPScore/paper]
  CLIP_edited     – CLIP(edited_masked, target_prompt)  [0-100, Otsu-masked edited region]
  CLIP_I_cos      – cosine_sim(CLIP_img(src), CLIP_img(edited))   [image-image ∈ [-1,1]]
  CLIP_I_score    – cos × logit_scale  (raw logit, ≈ 20-30 for similar images)
  LPIPS_alex      – LPIPS AlexNet  × 1e3  (↓ = more preserved)
  LPIPS_vgg       – LPIPS VGG16    × 1e3  (↓ = more preserved)
  LPIPS_squeeze   – LPIPS SqueezeNet raw [0-1]  ← matches SplitFlow/FTEdit paper
  MSE             – pixel MSE      × 1e4  (↓ = closer to src)
  PSNR            – Peak SNR dB           (↑ = closer to src)
  SSIM            – structural similarity × 1e2  (↑ = more preserved)
  StructDist_ours – Sobel-gradient-magnitude diff × 1e3  (↓ = structure preserved)
  StructDist_pie  – DINO ViT-B/8 key-self-sim MSE × 1e3 (↓ = structure preserved)
                    preprocessing matches reference: [0,255] tensor (no ToTensor())
  FID             – Fréchet Inception Distance per method (clean-fid, 256 × 256)

Usage
-----

  python compute_final_metrics.py

  python compute_final_metrics.py \\
      --outdir outputs/benchmark_20260226_160112 \\
      --yaml   Data/flowedit.yaml \\
      --images Data/flowedit_data/ \\
      --gpu    7

  python compute_final_metrics.py --no-dino --no-fid
"""

import os, sys, csv, json, re, yaml, argparse, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.filters import threshold_otsu
from tqdm import tqdm

ROOT = Path(__file__).parent

def _get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",  type=str, default=None,
                    help="Benchmark output dir (default: latest outputs/benchmark_*)")
    ap.add_argument("--yaml",    type=str, default="Data/flowedit.yaml")
    ap.add_argument("--images",  type=str, default="Data/flowedit_data/")
    ap.add_argument("--gpu",     type=int, default=7)
    ap.add_argument("--no-dino", action="store_true",
                    help="Skip StructDist_pie (DINO) — faster")
    ap.add_argument("--no-fid",  action="store_true",
                    help="Skip FID computation")
    ap.add_argument("--fid-only", action="store_true",
                    help="Only compute FID (skip all per-image metrics)")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Restrict to specific method folders, e.g. --methods flowedit_sd35 ftedit")
    return ap.parse_args()

def _latest_outdir() -> Path:
    dirs = sorted((ROOT / "outputs").glob("benchmark_*"), reverse=True)
    if not dirs:
        raise FileNotFoundError("No benchmark_* dir found in outputs/")
    return dirs[0]

def _find_image(path: str):
    p = Path(path)
    if p.exists():
        return str(p)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        c = p.with_suffix(ext)
        if c.exists():
            return str(c)
    return None

def load_pairs(yaml_path: str, images_root: str):
    with open(yaml_path) as f:
        entries = list(yaml.safe_load_all(f))[0]
    pairs = []
    for entry in entries:
        img_name   = Path(entry["init_img"]).name
        img_path   = str(Path(images_root) / img_name)
        resolved   = _find_image(img_path)
        if resolved:
            img_path = resolved
        base_name  = Path(img_name).stem
        src_prompt = str(entry["source_prompt"]).strip()
        for tp, code in zip(entry["target_prompts"], entry["target_codes"]):
            pairs.append({
                "image_path":    img_path,
                "base_name":     base_name,
                "source_prompt": src_prompt,
                "target_prompt": str(tp).strip(),
                "code":          str(code).strip(),
            })
    return pairs

class AllMetrics:
    """Loads all models lazily; call compute() per image pair."""

    def __init__(self, device: str, use_dino: bool = True):
        self.device   = device
        self.use_dino = use_dino
        self._clip_ready  = False
        self._lpips_ready = False
        self._dino_ready  = False

    def _load_clip(self):
        if self._clip_ready:
            return
        from transformers import CLIPModel, CLIPProcessor
        print("  Loading CLIP (ViT-L/14) …")
        self._clip  = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device).eval()
        self._cproc = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        self._logit_scale = self._clip.logit_scale.exp().item()
        self._clip_ready  = True

    @torch.no_grad()
    def _clip_image_feat(self, img: Image.Image) -> torch.Tensor:
        """Return L2-normalised CLIP image embedding (1, D)."""
        inp = self._cproc(images=img, return_tensors="pt").to(self.device)
        feat = self._clip.get_image_features(**inp)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def clip_whole(self, edited: Image.Image, prompt: str) -> float:
        """CLIP(edited, prompt) — raw logit [0-100] matching CLIPScore torchmetrics scale."""
        self._load_clip()
        inp = self._cproc(text=[prompt], images=edited,
                          return_tensors="pt", padding=True).to(self.device)
        return self._clip(**inp).logits_per_image.item()

    @torch.no_grad()
    def clip_edited(self, src: Image.Image, edited: Image.Image,
                    prompt: str) -> float:
        """CLIP on the auto-masked (Otsu) edited region — raw logit [0-100]."""
        self._load_clip()
        s = np.array(src.convert("RGB").resize((512, 512))).astype(np.float32)
        e = np.array(edited.convert("RGB").resize((512, 512))).astype(np.float32)
        diff = np.mean(np.abs(s - e), axis=2)
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
        crop = edited.convert("RGB").resize((512, 512)).crop(
            (x1, y1, x2 + 1, y2 + 1))
        inp = self._cproc(text=[prompt], images=crop,
                          return_tensors="pt", padding=True).to(self.device)
        return self._clip(**inp).logits_per_image.item()

    @torch.no_grad()
    def clip_image_image(self, src: Image.Image,
                         edited: Image.Image) -> tuple[float, float]:
        """
        Returns (CLIP_I_cos, CLIP_I_score).
          CLIP_I_cos   – cosine similarity of L2-norm embeddings ∈ [-1, 1]
          CLIP_I_score – raw logit = cos × logit_scale  (≈ 20-30 for similar images)
        """
        self._load_clip()
        f_src  = self._clip_image_feat(src)
        f_edit = self._clip_image_feat(edited)
        cos    = (f_src * f_edit).sum().item()
        score  = cos * self._logit_scale
        return cos, score

    def _load_lpips(self):
        if self._lpips_ready:
            return
        import lpips as _lpips
        print("  Loading LPIPS (AlexNet + VGG + SqueezeNet) …")
        self._lpips_alex   = _lpips.LPIPS(net="alex").to(self.device).eval()
        self._lpips_vgg    = _lpips.LPIPS(net="vgg").to(self.device).eval()
        self._lpips_squeeze = _lpips.LPIPS(net="squeeze").to(self.device).eval()
        self._lpips_ready = True

    def _to_tensor11(self, img: Image.Image) -> torch.Tensor:
        """(1,3,H,W) float32 in [-1,1]."""
        a = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1

    @torch.no_grad()
    def lpips_both(self, src: Image.Image,
                   edited: Image.Image) -> tuple[float, float, float]:
        """Returns (LPIPS_alex × 1e3, LPIPS_vgg × 1e3, LPIPS_squeeze raw [0-1]).
        LPIPS_squeeze is raw (not ×1e3) to match paper (SplitFlow/FTEdit MetricsCalculator)."""
        self._load_lpips()
        t_src  = self._to_tensor11(src)
        t_edit = self._to_tensor11(edited)
        alex   = self._lpips_alex(t_src, t_edit).item() * 1e3
        vgg    = self._lpips_vgg(t_src, t_edit).item() * 1e3
        squeeze = self._lpips_squeeze(t_src, t_edit).item()
        return alex, vgg, squeeze

    @staticmethod
    def _np512(img: Image.Image) -> np.ndarray:
        return np.array(img.convert("RGB").resize(
            (512, 512), Image.LANCZOS)).astype(np.float32) / 255.0

    @staticmethod
    def _np_native(img: Image.Image) -> np.ndarray:
        """No resize — matches reference calculate_mse (original resolution)."""
        return np.array(img.convert("RGB")).astype(np.float32) / 255.0

    def pixel_metrics(self, src: Image.Image,
                      edited: Image.Image) -> dict[str, float]:
        s, e = self._np512(src), self._np512(edited)
        mse_raw = float(np.mean((s - e) ** 2))
        mse     = mse_raw * 1e4
        rmse    = float(np.sqrt(mse_raw))
        psnr = float(peak_signal_noise_ratio(s, e, data_range=1.0))
        ssim = float(structural_similarity(
            s, e, data_range=1.0, channel_axis=2, win_size=11) * 1e2)

        sn, en  = self._np_native(src), self._np_native(edited)
        if sn.shape == en.shape:
            mse_native = float(np.mean((sn - en) ** 2))
        else:
            en2 = np.array(edited.convert("RGB").resize(
                src.size, Image.LANCZOS)).astype(np.float32) / 255.0
            mse_native = float(np.mean((sn - en2) ** 2))

        return {"MSE": mse, "MSE_raw": mse_raw, "RMSE": rmse,
                "MSE_native": mse_native, "PSNR": psnr, "SSIM": ssim}

    def struct_dist_ours(self, src: Image.Image,
                         edited: Image.Image) -> float:
        s, e = self._np512(src), self._np512(edited)
        w    = np.array([0.299, 0.587, 0.114])
        ls, le = s @ w, e @ w
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gx_s, gy_s = convolve(ls, kx), convolve(ls, kx.T)
        gx_e, gy_e = convolve(le, kx), convolve(le, kx.T)
        g_s = np.sqrt(gx_s ** 2 + gy_s ** 2 + 1e-8)
        g_e = np.sqrt(gx_e ** 2 + gy_e ** 2 + 1e-8)
        return float(np.mean(np.abs(g_s - g_e) / (g_s + g_e + 1e-6)) * 1e3)

    def _load_dino(self):
        if self._dino_ready or not self.use_dino:
            return
        from torchvision import transforms
        print("  Loading DINO ViT-B/8 …")
        try:
            self._dino = torch.hub.load(
                "facebookresearch/dino:main", "dino_vitb8",
                verbose=False).to(self.device).eval()
        except Exception as e:
            print(f"  [WARN] Could not load DINO: {e} — StructDist_pie will be NaN")
            self.use_dino = False
            return
        imagenet_norm = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self._dino_tf = transforms.Compose([
            transforms.Resize(224, max_size=480),
            imagenet_norm,
        ])
        self._hook_outputs = []
        last_block = self._dino.blocks[-1]

        def _hook(mod, inp, out):
            self._hook_outputs.append(out)

        last_block.attn.qkv.register_forward_hook(_hook)
        self._dino_ready = True

    @torch.no_grad()
    def _dino_key_self_sim(self, img: Image.Image) -> torch.Tensor:
        """DINO key self-similarity matrix from last block, layer 11.
        Preprocessing matches reference: PIL→numpy[0,255]→tensor, Resize, ImageNet norm."""
        img_np = np.array(img.convert("RGB")).astype(np.float32)
        t_raw  = torch.from_numpy(img_np.transpose(2, 0, 1)).to(self.device)
        t = self._dino_tf(t_raw).unsqueeze(0)
        self._hook_outputs.clear()
        _ = self._dino(t)
        qkv = self._hook_outputs[0]
        B, N, _ = qkv.shape
        head_num = self._dino.blocks[-1].attn.num_heads
        embed_dim = qkv.shape[-1] // 3
        d_head = embed_dim // head_num
        qkv_r = qkv.reshape(B, N, 3, head_num, d_head).permute(2, 0, 3, 1, 4)
        keys = qkv_r[1, 0]
        h, t_n, d = keys.shape
        concat_k = keys.transpose(0, 1).reshape(t_n, h * d)
        norm1 = concat_k.norm(dim=1, keepdim=True)
        factor = torch.clamp(norm1 @ norm1.T, min=1e-8)
        ssim_map = (concat_k @ concat_k.T) / factor
        return ssim_map

    @torch.no_grad()
    def struct_dist_pie(self, src: Image.Image,
                        edited: Image.Image) -> float:
        """DINO ViT-B/8 structure distance (PIE-Bench). Returns value × 1e3."""
        self._load_dino()
        if not self._dino_ready:
            return float("nan")
        ssim_src  = self._dino_key_self_sim(src)
        ssim_edit = self._dino_key_self_sim(edited)
        dist = F.mse_loss(ssim_edit, ssim_src).item()
        return dist * 1e3

    @staticmethod
    def _auto_crop_panel(src: Image.Image, edited: Image.Image) -> Image.Image:
        """
        Detect and crop 4-panel P2PEditor output [instruct|src|recon|edited].
        If edited width ≈ 4× src width → take the rightmost quarter.
        Also handles 2-panel (2× width) → take right half.
        """
        ew, eh = edited.size
        sw, sh = src.size
        ratio = ew / max(sw, 1)
        if 3.5 <= ratio <= 4.5:
            panel_w = ew // 4
            edited = edited.crop((3 * panel_w, 0, 4 * panel_w, eh))
        elif 1.8 <= ratio <= 2.2:
            panel_w = ew // 2
            edited = edited.crop((panel_w, 0, ew, eh))
        return edited

    def compute(self, src: Image.Image, edited: Image.Image,
                target_prompt: str) -> dict[str, float]:
        edited = self._auto_crop_panel(src, edited)
        if edited.size != src.size:
            edited = edited.resize(src.size, Image.LANCZOS)

        m = {}

        m["CLIP_whole"]  = self.clip_whole(edited, target_prompt)
        m["CLIP_edited"] = self.clip_edited(src, edited, target_prompt)

        cos, score = self.clip_image_image(src, edited)
        m["CLIP_I_cos"]   = cos
        m["CLIP_I_score"] = score

        alex, vgg, squeeze = self.lpips_both(src, edited)
        m["LPIPS_alex"]    = alex
        m["LPIPS_vgg"]     = vgg
        m["LPIPS_squeeze"] = squeeze

        m.update(self.pixel_metrics(src, edited))

        m["StructDist_ours"] = self.struct_dist_ours(src, edited)
        m["StructDist_pie"]  = self.struct_dist_pie(src, edited)

        return m

def compute_fid_all(out_dir: Path, method_folders: list[str],
                    source_imgs: list[str],
                    device: str = "cuda") -> dict[str, float]:
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("[FID] clean-fid not found — pip install clean-fid")
        return {}

    import tempfile, shutil
    import torch
    _device = torch.device(device)

    ref_dir = out_dir / "_fid_ref_256"
    ref_dir.mkdir(exist_ok=True)
    for p in source_imgs:
        res = _find_image(p)
        if res is None:
            continue
        dst = ref_dir / Path(res).name
        if not dst.with_suffix(".png").exists():
            img = Image.open(res).convert("RGB").resize((256, 256), Image.LANCZOS)
            img.save(dst.with_suffix(".png"))

    n_ref = len(list(ref_dir.glob("*.png")))
    if n_ref < 2:
        print(f"[FID] only {n_ref} source images — skipping")
        return {}

    scores = {}
    for method in method_folders:
        edited_folder = out_dir / method
        if not edited_folder.exists():
            continue
        pngs = list(edited_folder.glob("*.png"))
        if len(pngs) < 2:
            print(f"[FID] {method}: only {len(pngs)} images — skipping")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            for f in pngs:
                img = Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)
                img.save(tmp_p / f.name)
            try:
                print(f"  FID {method} ({len(pngs)} images) …")
                score = cleanfid.compute_fid(
                    str(tmp_p), str(ref_dir), mode="clean",
                    device=_device, verbose=False)
                scores[method] = score
                print(f"    → {score:.2f}")
            except Exception as e:
                print(f"  [FID] {method}: {e}")
    return scores

METRIC_COLS = [
    "CLIP_whole", "CLIP_edited",
    "CLIP_I_cos", "CLIP_I_score",
    "LPIPS_alex", "LPIPS_vgg", "LPIPS_squeeze",
    "MSE", "MSE_raw", "MSE_native", "RMSE",
    "PSNR", "SSIM",
    "StructDist_ours", "StructDist_pie",
]
CSV_COLS = ["method", "image", "code", "target_prompt"] + METRIC_COLS + ["file"]

def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def merge_all_csvs(out_dir: Path) -> Path:
    merged = out_dir / "final_results_merged.csv"
    all_rows = []
    for f in sorted(out_dir.glob("final_results_*.csv")):
        if f == merged:
            continue
        with open(f) as fh:
            all_rows.extend(list(csv.DictReader(fh)))
    if all_rows:
        with open(merged, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_rows[0].keys(),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
    return merged

def print_summary(merged_csv: Path, fid_scores: dict[str, float]):
    with open(merged_csv) as f:
        rows = list(csv.DictReader(f))

    sums = defaultdict(lambda: defaultdict(float))
    cnts = defaultdict(int)
    for r in rows:
        m = r["method"]
        cnts[m] += 1
        for col in METRIC_COLS:
            try:
                sums[m][col] += float(r[col])
            except (ValueError, KeyError):
                pass

    methods = sorted(cnts.keys())
    avgs = {m: {c: sums[m][c] / cnts[m] for c in METRIC_COLS}
            for m in methods}

    col_w = 13
    hdr   = f"{'Method':<20}" + "".join(f"{c:>{col_w}}" for c in METRIC_COLS) + f"{'FID':>{col_w}}"
    sep   = "─" * len(hdr)
    print("\n" + sep)
    print("  FINAL BENCHMARK SUMMARY")
    print(sep)
    print(hdr)
    print(sep)
    for m in methods:
        row = f"{m:<20}"
        for c in METRIC_COLS:
            v = avgs[m].get(c, float("nan"))
            row += f"{v:>{col_w}.4f}"
        fid = fid_scores.get(m, float("nan"))
        row += f"{fid:>{col_w}.2f}"
        print(row)
    print(sep)

    summary = {m: {**avgs[m], "FID": fid_scores.get(m, float("nan")),
                   "n_pairs": cnts[m]}
               for m in methods}
    summary_path = merged_csv.parent / "final_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary → {summary_path}\n")

def main():
    args = _get_args()

    if args.outdir is None:
        out_dir = _latest_outdir()
    else:
        out_dir = Path(args.outdir)
    yaml_path   = str(ROOT / args.yaml)
    images_root = str(ROOT / args.images)

    print(f"Output dir  : {out_dir}")
    print(f"YAML        : {yaml_path}")
    print(f"Images root : {images_root}")
    print(f"GPU         : {args.gpu}")

    device = f"cuda:{args.gpu}"
    pairs  = load_pairs(yaml_path, images_root)
    print(f"Pairs loaded: {len(pairs)}")

    pair_lookup = {(p["base_name"], p["code"]): p for p in pairs}

    if args.methods:
        method_folders = [m for m in args.methods
                          if (out_dir / m).is_dir()]
    else:
        method_folders = sorted([
            d.name for d in out_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ])

    print(f"Methods     : {method_folders}\n")

    all_source_imgs = list({p["image_path"] for p in pairs})

    if args.fid_only:
        print("FID-only mode — skipping per-image metrics.\n")
        fid_scores = compute_fid_all(out_dir, method_folders, all_source_imgs, device=device)
        fid_path = out_dir / "final_fid_scores.json"
        with open(fid_path, "w") as f:
            json.dump(fid_scores, f, indent=2)
        print(f"\nFID scores saved: {fid_path}")
        for m, s in fid_scores.items():
            print(f"  {m:<25} FID = {s:.2f}")
        summary_path = out_dir / "final_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            for m, s in fid_scores.items():
                if m in summary:
                    summary[m]["FID"] = s
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary updated : {summary_path}")
        return

    ev = AllMetrics(device=device, use_dino=not args.no_dino)

    for method in method_folders:
        method_dir = out_dir / method
        pngs = sorted(method_dir.glob("*.png"))
        if not pngs:
            print(f"[{method}] No images found — skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  {method}  ({len(pngs)} images)")
        print(f"{'='*60}")

        rows = []
        for img_path in tqdm(pngs, desc=method):
            stem = img_path.stem
            matched_pair = None
            for (bn, code), pair in pair_lookup.items():
                expected_stem = f"{bn}_{code}"
                if stem == expected_stem:
                    matched_pair = pair
                    break

            if matched_pair is None:
                print(f"  [WARN] no pair found for {stem} — skipping")
                continue

            src_resolved = _find_image(matched_pair["image_path"])
            if src_resolved is None:
                print(f"  [WARN] source image not found: {matched_pair['image_path']}")
                continue

            try:
                src_img    = Image.open(src_resolved).convert("RGB")
                edited_img = Image.open(img_path).convert("RGB")

                metrics = ev.compute(src_img, edited_img,
                                     matched_pair["target_prompt"])
                rows.append({
                    "method":        method,
                    "image":         matched_pair["base_name"],
                    "code":          matched_pair["code"],
                    "target_prompt": matched_pair["target_prompt"],
                    **metrics,
                    "file":          str(img_path),
                })
            except Exception:
                print(f"  [ERR] {stem}:\n{traceback.format_exc()}")

        csv_path = out_dir / f"final_results_{method}.csv"
        write_csv(csv_path, rows)
        print(f"  → {csv_path}  ({len(rows)} rows)")

        if rows:
            print(f"  Running averages ({len(rows)} pairs):")
            for col in METRIC_COLS:
                vals = [float(r[col]) for r in rows
                        if r.get(col) not in (None, "", "nan")]
                if vals:
                    print(f"    {col:<18} {np.mean(vals):.4f}")

    merged = merge_all_csvs(out_dir)
    print(f"\nMerged CSV  : {merged}")

    fid_scores = {}
    if not args.no_fid:
        print("\nComputing FID …")
        fid_scores = compute_fid_all(out_dir, method_folders, all_source_imgs, device=device)
        fid_path = out_dir / "final_fid_scores.json"
        with open(fid_path, "w") as f:
            json.dump(fid_scores, f, indent=2)
        print(f"FID saved   : {fid_path}")

    if merged.exists():
        print_summary(merged, fid_scores)

if __name__ == "__main__":
    main()
