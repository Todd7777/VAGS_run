"""
merge_results.py — Fusion CSVs + FID + table finale PIE-Bench
"""
import os, argparse, glob
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="outputs/PIE_BENCH_PARALLEL")
args = parser.parse_args()
D = args.output_dir

# ══════════════════════════════════════════════════════════════════════════════
# 1. FUSION + DÉDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════
files = sorted(glob.glob(os.path.join(D, "results_gpu*.csv")))
if not files: print(f"Aucun CSV dans {D}"); exit(1)

print(f"{len(files)} fichier(s) trouvés")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset=["Image", "Strategy"], keep="first")
if before != len(df):
    print(f"⚠  {before - len(df)} doublons supprimés")

n_img, n_strat = df["Image"].nunique(), df["Strategy"].nunique()
print(f"Total : {len(df)} lignes ({n_img} images × {n_strat} stratégies)")
if len(df) < n_img * n_strat:
    print(f"⚠  {n_img*n_strat - len(df)} combinaisons manquantes")

df.to_csv(os.path.join(D, "all_results_merged.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 2. FID  (Inception-v3, calculé sur les images sauvegardées)
# ══════════════════════════════════════════════════════════════════════════════
def compute_fid(real_feats, fake_feats):
    """Standard FID using scipy sqrtm for numerical stability."""
    from scipy.linalg import sqrtm as scipy_sqrtm
    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sig_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff  = mu_r - mu_f
    covmean = scipy_sqrtm(sig_r @ sig_f)
    # numerical issues → small imaginary parts, discard them
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2*covmean))

fid_scores = {}
try:
    import torch
    from torchvision import transforms
    from torchvision.models import inception_v3
    from PIL import Image as PILImage

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nCalcul FID sur {dev}…")

    net = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(dev).eval()
    net.fc = torch.nn.Identity()
    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    @torch.no_grad()
    def feat(path):
        img = PILImage.open(path).convert("RGB")
        return net(tfm(img).unsqueeze(0).to(dev)).squeeze().cpu().numpy()

    strategies = [s for s in [
        "Baseline","Exp_Forward_Gate","Exp_Forward","Exp_Reverse_Gate",
        "Exp_Boost","TwoPhase","Entropy","Locality","Dual","Quantile","Budget_Matched"
    ] if s in df["Strategy"].values]

    # real = Baseline images as reference distribution
    real_feats, fake_feats_dict = [], {s: [] for s in strategies}

    for img_name in df["Image"].unique():
        for s in strategies:
            path = os.path.join(D, f"{img_name}_{s}.jpg")
            if not os.path.exists(path): continue
            f = feat(path)
            fake_feats_dict[s].append(f)
            if s == "Baseline":
                real_feats.append(f)  # Baseline = référence

    if len(real_feats) >= 2:
        real_arr = np.stack(real_feats)
        for s, fl in fake_feats_dict.items():
            if len(fl) >= 2:
                fid_scores[s] = round(compute_fid(real_arr, np.stack(fl)), 2)
                print(f"  FID [{s:<22}] = {fid_scores[s]:.2f}")
    else:
        print("  Pas assez d'images pour FID")
except Exception as e:
    print(f"  FID ignoré : {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
num_cols = [c for c in [
    "StructDist_1e3","PSNR_dB","LPIPS_1e3","MSE_1e4","SSIM_100",
    "CLIP_W","CLIP_E","CLIP_I","Lambda_mean"
] if c in df.columns]

summary = df.groupby("Strategy")[num_cols].mean().round(4)
if fid_scores:
    summary["FID"] = pd.Series(fid_scores).round(2)

ORDER = ["Baseline","Exp_Forward_Gate","Exp_Forward","Exp_Reverse_Gate",
         "Exp_Boost","TwoPhase","Entropy","Locality","Dual","Quantile","Budget_Matched"]
summary = summary.reindex([s for s in ORDER if s in summary.index])
summary.to_csv(os.path.join(D, "summary_table.csv"))

# ══════════════════════════════════════════════════════════════════════════════
# 4. AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════
arrows = {"StructDist_1e3":"↓","PSNR_dB":"↑","LPIPS_1e3":"↓","MSE_1e4":"↓",
          "SSIM_100":"↑","CLIP_W":"↑","CLIP_E":"↑","CLIP_I":"↑","Lambda_mean":"","FID":"↓"}

all_cols = num_cols + (["FID"] if fid_scores else [])
W = 14

print("\n" + "═"*((W+3)*len(all_cols)+24))
print(f"  RÉSULTATS FINAUX — {n_img} images · {n_strat} stratégies · SD3-medium")
print("═"*((W+3)*len(all_cols)+24))

# header
hdr = f"  {'Strategy':<22}"
for c in all_cols: hdr += f" | {(c+arrows.get(c,''))[:W]:>{W}}"
print(hdr)
print("  " + "-"*22 + ("--+-" + "-"*W)*len(all_cols))

for strat, row in summary.iterrows():
    line = f"  {strat:<22}"
    for c in all_cols:
        v = row.get(c, float("nan"))
        line += f" | {v:>{W}.4f}" if not (isinstance(v,float) and np.isnan(v)) else f" | {'nan':>{W}}"
    print(line)

print("═"*((W+3)*len(all_cols)+24))
print("""
  ── Référence SotA (SD3) ──────────────────────────────────────────────
  FlowEdit   SD=27.24 PSNR=22.13 LPIPS=105.46 MSE=87.34 SSIM=83.48 CLIP-W=26.83 CLIP-E=23.67
  SplitFlow  SD=25.96 PSNR=22.45 LPIPS=102.14 MSE=81.99 SSIM=83.91 CLIP-W=26.96 CLIP-E=23.83
""")

# breakdown par catégorie
if "Category" in df.columns:
    print("── PSNR par catégorie ───────────────────────────────────────────────")
    cat = df.pivot_table(values="PSNR_dB", index="Category",
                         columns="Strategy", aggfunc="mean").round(2)
    cat = cat[[s for s in ORDER if s in cat.columns]]
    print(cat.to_string())

print(f"\n  CSV      → {os.path.join(D,'all_results_merged.csv')}")
print(f"  Résumé   → {os.path.join(D,'summary_table.csv')}")