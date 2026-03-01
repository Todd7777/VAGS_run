"""
compute_all_metrics.py — compute DIV2K-style edit metrics on pre-generated images.

Metrics (no pixel-level metrics — global edits without masks):
  CLIP_T_whole  : CLIP cosine similarity edited image ↔ target prompt
  CLIP_I_cos    : CLIP image–image cosine similarity src ↔ edited
  DINO_sim      : DINO ViT-B/8 cosine similarity src ↔ edited
  LPIPS_alex    : LPIPS AlexNet   × 1e3
  LPIPS_vgg     : LPIPS VGG       × 1e3
  LPIPS_squeeze : LPIPS SqueezeNet × 1e3
  DreamSim      : DreamSim perceptual distance (lower = more similar)

Usage:
  python compute_all_metrics.py \\
      --outdir outputs/benchmark_20260226_160112 \\
      --yaml   Data/flowedit.yaml \\
      --images Data/flowedit_data/ \\
      [--methods flowedit_sd35 splitflow_sd35] \\
      [--max_pairs 5] \\
      [--device cuda:0]
"""
import argparse
import csv
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",    required=True,  type=Path)
    p.add_argument("--yaml",      required=True,  type=Path)
    p.add_argument("--images",    required=True,  type=Path)
    p.add_argument("--methods",   nargs="+",      default=None)
    p.add_argument("--max_pairs", type=int,       default=None)
    p.add_argument("--device",    default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_yaml_pairs(yaml_path: Path, images_root: Path):
    """Return list of dicts: {image_stem, src_path, target_prompt, code}."""
    with open(yaml_path) as f:
        entries = yaml.safe_load(f)
    pairs = []
    for entry in entries:
        init_img = entry["init_img"]
        src_path = images_root.parent / init_img if not (images_root / Path(init_img).name).exists() else images_root / Path(init_img).name
        if not src_path.exists():
            src_path = images_root / Path(init_img).name
        stem = Path(init_img).stem
        prompts = entry.get("target_prompts", [])
        codes   = entry.get("target_codes",   [])
        for prompt, code in zip(prompts, codes):
            pairs.append({
                "image":          stem,
                "src_path":       src_path,
                "target_prompt":  prompt.strip(),
                "code":           code,
            })
    return pairs


class MetricModels:
    def __init__(self, device: str):
        self.device = device
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print("Loading CLIP ViT-L/14 …")
        import clip as openai_clip
        self._clip_model, self._clip_preprocess = openai_clip.load(
            "ViT-L/14", device=self.device)
        self._clip_model.eval()

        print("Loading DINO ViT-B/8 …")
        self._dino = torch.hub.load(
            "facebookresearch/dino:main", "dino_vitb8",
            verbose=False).to(self.device).eval()
        self._dino_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3,1,1)
        self._dino_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3,1,1)

        print("Loading LPIPS (alex / vgg / squeeze) …")
        import lpips as _lp
        self._lpips_alex    = _lp.LPIPS(net="alex").to(self.device).eval()
        self._lpips_vgg     = _lp.LPIPS(net="vgg").to(self.device).eval()
        self._lpips_squeeze = _lp.LPIPS(net="squeeze").to(self.device).eval()

        print("Loading DreamSim …")
        from dreamsim import dreamsim as _dreamsim
        self._dreamsim_model, self._dreamsim_preprocess = _dreamsim(
            pretrained=True, device=self.device)
        self._dreamsim_model.eval()

        print("Loading Todd MetricsCalculator (PSNR/MSE/SSIM/StructDist) …")
        import sys as _sys
        _mc_dir = str(Path(__file__).parent / "methods/PnPInversion/evaluation")
        if _mc_dir not in _sys.path:
            _sys.path.insert(0, _mc_dir)
        from matrics_calculator import MetricsCalculator
        self._mc = MetricsCalculator(self.device)

        self._loaded = True
        print("All models loaded.\n")

    def _to_tensor_01(self, img: Image.Image, size: int) -> torch.Tensor:
        arr = np.array(img.convert("RGB").resize((size, size), Image.LANCZOS),
                       dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _to_tensor_11(self, t01: torch.Tensor) -> torch.Tensor:
        return t01 * 2 - 1

    @torch.no_grad()
    def compute(self, src: Image.Image, edited: Image.Image,
                target_prompt: str) -> dict:
        self.load()
        import clip as openai_clip

        src_224    = self._to_tensor_01(src,    224)
        edited_224 = self._to_tensor_01(edited, 224)

        # CLIP image features
        def clip_img_feat(t01):
            t_clip = self._clip_preprocess(
                Image.fromarray((t01.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            ).unsqueeze(0).to(self.device)
            return self._clip_model.encode_image(t_clip).float()

        src_clip_feat    = clip_img_feat(src_224)
        edited_clip_feat = clip_img_feat(edited_224)

        # CLIP text features
        text_tok  = openai_clip.tokenize([target_prompt], truncate=True).to(self.device)
        text_feat = self._clip_model.encode_text(text_tok).float()

        # CLIP_T_whole: cosine(edited_img_feat, text_feat)
        clip_t = float(torch.nn.functional.cosine_similarity(
            edited_clip_feat, text_feat).item())

        # CLIP_I_cos: cosine(src_img_feat, edited_img_feat)
        clip_i = float(torch.nn.functional.cosine_similarity(
            src_clip_feat, edited_clip_feat).item())

        # DINO similarity
        def dino_feat(t01):
            t = (t01 - self._dino_mean) / self._dino_std
            return self._dino(t)

        src_dino    = dino_feat(src_224)
        edited_dino = dino_feat(edited_224)
        dino_sim = float(torch.nn.functional.cosine_similarity(
            src_dino, edited_dino).item())

        # LPIPS all variants (×1e3)
        src_512    = self._to_tensor_01(src,    512)
        edited_512 = self._to_tensor_01(edited, 512)
        s11 = self._to_tensor_11(src_512)
        t11 = self._to_tensor_11(edited_512)
        lpips_alex    = float(self._lpips_alex(s11, t11).item())    * 1e3
        lpips_vgg     = float(self._lpips_vgg(s11, t11).item())     * 1e3
        lpips_squeeze = float(self._lpips_squeeze(s11, t11).item()) * 1e3

        # DreamSim
        ds_src    = self._dreamsim_preprocess(src.convert("RGB")).to(self.device)
        ds_edited = self._dreamsim_preprocess(edited.convert("RGB")).to(self.device)
        dreamsim_v = float(self._dreamsim_model(ds_src, ds_edited).item())

        # Todd MetricsCalculator — mask=None, both resized to 512×512 to match shapes
        src512    = src.convert("RGB").resize((512, 512), Image.LANCZOS)
        edited512 = edited.convert("RGB").resize((512, 512), Image.LANCZOS)
        psnr_v   = float(self._mc.calculate_psnr(src512, edited512, None, None))
        mse_v    = float(self._mc.calculate_mse(src512, edited512, None, None))
        ssim_v   = float(self._mc.calculate_ssim(src512, edited512, None, None))
        struct_v = float(self._mc.calculate_structure_distance(src512, edited512, None, None))

        return {
            "CLIP_T_whole":  clip_t,
            "CLIP_I_cos":    clip_i,
            "DINO_sim":      dino_sim,
            "LPIPS_alex":    lpips_alex,
            "LPIPS_vgg":     lpips_vgg,
            "LPIPS_squeeze": lpips_squeeze,
            "DreamSim":      dreamsim_v,
            "PSNR":          psnr_v,
            "MSE":           mse_v,
            "SSIM":          ssim_v,
            "StructDist":    struct_v,
        }


def discover_methods(outdir: Path, requested: list | None) -> list[str]:
    methods = sorted(
        d.name for d in outdir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    if requested:
        missing = [m for m in requested if m not in methods]
        if missing:
            print(f"[WARN] Requested methods not found in outdir: {missing}")
        methods = [m for m in requested if m in methods]
    return methods


METRIC_KEYS = ["CLIP_T_whole", "CLIP_I_cos", "DINO_sim",
               "LPIPS_alex", "LPIPS_vgg", "LPIPS_squeeze", "DreamSim",
               "PSNR", "MSE", "SSIM", "StructDist"]


def run_method(method: str, outdir: Path, pairs: list, models: MetricModels,
               max_pairs: int | None) -> list[dict]:
    method_dir = outdir / method
    rows = []
    skipped = 0
    pairs_to_run = pairs[:max_pairs] if max_pairs else pairs

    for pair in tqdm(pairs_to_run, desc=method, ncols=100):
        img_name = f"{pair['image']}_{pair['code']}.png"
        edited_path = method_dir / img_name
        if not edited_path.exists():
            skipped += 1
            if skipped <= 3:
                print(f"  [SKIP] {img_name}")
            continue

        try:
            src    = Image.open(pair["src_path"]).convert("RGB")
            edited = Image.open(edited_path).convert("RGB")
            m = models.compute(src, edited, pair["target_prompt"])
            rows.append({
                "method":  method,
                "image":   pair["image"],
                "code":    pair["code"],
                **m,
                "file":    str(edited_path),
            })
        except Exception as e:
            print(f"  [ERR] {img_name}: {e}")

    if skipped > 3:
        print(f"  [SKIP] … {skipped} images total not found")
    return rows


def _model_group(method: str) -> str:
    """Extract backbone model tag for grouping."""
    m = method.lower()
    if any(x in m for x in ("flux", "ftedit", "fireflow", "rf_inv", "rf_sol",
                             "rf_inversion", "rf_solver", "cag")):
        return "0_flux"
    if m in ("flowedit",) or m.startswith("flowedit_flux"):
        return "0_flux"
    if "sd35" in m or "sd3" in m or m in ("flowedit_sd35", "splitflow_sd35"):
        return "1_sd35"
    if m.startswith("flowedit") and "sd" not in m:
        return "0_flux"
    if any(x in m for x in ("sd14", "sd1", "ddim", "pnpinv", "pnp_inv",
                             "null_text", "p2p", "pnp")):
        return "2_sd14"
    return "3_other"


def print_summary(all_rows: list[dict]):
    by_method = defaultdict(list)
    for r in all_rows:
        by_method[r["method"]].append(r)

    sorted_methods = sorted(by_method, key=lambda m: (_model_group(m), m))

    header = f"{'Method':<28}{'N':>6}" + "".join(f"{k:>14}" for k in METRIC_KEYS)
    sep = "=" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    current_group = None
    for method in sorted_methods:
        group = _model_group(method)
        if group != current_group:
            if current_group is not None:
                print("-" * len(header))
            current_group = group
        vals = by_method[method]
        avgs = {k: float(np.nanmean([r[k] for r in vals])) for k in METRIC_KEYS}
        print(f"{method:<28}{len(vals):>6}" + "".join(f"{avgs[k]:>14.4f}" for k in METRIC_KEYS))
    print(sep)


def save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fields = ["method", "image", "code"] + METRIC_KEYS + ["file"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()

    outdir = args.outdir.expanduser().resolve()
    yaml_p = args.yaml.expanduser().resolve()
    img_root = args.images.expanduser().resolve()

    if not outdir.exists():
        raise SystemExit(f"outdir not found: {outdir}")

    pairs = load_yaml_pairs(yaml_p, img_root)
    print(f"Loaded {len(pairs)} pairs from {yaml_p.name}")

    methods = discover_methods(outdir, args.methods)
    if not methods:
        raise SystemExit("No method directories found.")
    print(f"Methods to evaluate: {methods}\n")

    models = MetricModels(args.device)
    all_rows = []

    for method in methods:
        rows = run_method(method, outdir, pairs, models,
                          max_pairs=args.max_pairs)
        all_rows.extend(rows)

        if rows:
            csv_path = outdir / f"results_{method}.csv"
            save_csv(rows, csv_path)
            print(f"  Saved {len(rows)} rows → {csv_path}")

    if all_rows:
        merged_path = outdir / "results_merged.csv"
        save_csv(all_rows, merged_path)
        print(f"\nMerged CSV → {merged_path}")
        print_summary(all_rows)


if __name__ == "__main__":
    main()
