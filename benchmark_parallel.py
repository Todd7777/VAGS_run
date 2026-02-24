"""
benchmark_parallel.py
══════════════════════════════════════════════════════════════════════════════
Version parallélisable du benchmark CAG.
Chaque instance gère 1 chunk du dataset sur 1 GPU.

Usage mono-GPU :
    python benchmark_parallel.py --gpu_id 0

Usage 8 GPUs (via launch_parallel.sh) :
    CUDA_VISIBLE_DEVICES=1 python benchmark_parallel.py --gpu_id 1 --num_chunks 7 --chunk_idx 0
    CUDA_VISIBLE_DEVICES=2 python benchmark_parallel.py --gpu_id 2 --num_chunks 7 --chunk_idx 1
    ...

Test rapide (2 images) :
    CUDA_VISIBLE_DEVICES=1 python benchmark_parallel.py --gpu_id 1 --test_run
"""

import os, gc, json, csv, math, random, argparse, time

# Doit être setté AVANT import torch (fait par le shell via CUDA_VISIBLE_DEVICES=X)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import lpips as lpips_lib
from torchvision import transforms
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# 0.  ARGS
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",      type=int,  default=0)
parser.add_argument("--num_chunks",  type=int,  default=1)
parser.add_argument("--chunk_idx",   type=int,  default=0)
parser.add_argument("--test_run",    action="store_true",
                    help="Ne traite que 2 images (vérification rapide)")
args = parser.parse_args()

# CUDA_VISIBLE_DEVICES setté par le shell → cuda:0 == le GPU assigné
DEVICE = torch.device("cuda:0")
DTYPE  = torch.float16

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS & HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PIE_BENCH_ROOT = "Data/PIE-Bench_v1/annotation_images"
MASK_ROOT      = "Data/PIE-Bench_v1/annotation_masks"
MAPPING_FILE   = "Data/PIE-Bench_v1/mapping_file.json"
OUTPUT_DIR     = "outputs/PIE_BENCH_PARALLEL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HP = dict(
    t_start    = 0.66,
    cfg_src    = 3.5,
    base_cfg   = 13.5,
    kappa      = 4.0,
    m          = 3.0,
    lambda_min = 1.0,
    lambda_max = 40.0,
    steps      = 50,
    img_size   = 1024,
)

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
# 2.  DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset_chunk():
    image_index = {}
    for root, _, files in os.walk(PIE_BENCH_ROOT):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                stem = os.path.splitext(f)[0]
                image_index[stem] = os.path.join(root, f)

    with open(MAPPING_FILE) as fp:
        mapping = list(json.load(fp).items())

    chunk_size = math.ceil(len(mapping) / args.num_chunks)
    start      = args.chunk_idx * chunk_size
    end        = min(start + chunk_size, len(mapping))
    sub_map    = mapping[start:end]

    if args.test_run:
        sub_map = sub_map[:2]
        print(f"[GPU {args.gpu_id}] TEST MODE — 2 images seulement")

    dataset = []
    for key, ann in sub_map:
        stem = os.path.splitext(os.path.basename(key))[0]
        if stem not in image_index:
            continue
        full_path = image_index[stem]
        category  = os.path.basename(os.path.dirname(full_path))

        mask_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            for cand in [
                os.path.join(MASK_ROOT, category, stem + ext),
                os.path.join(MASK_ROOT, stem + ext),
                full_path.replace("annotation_images", "annotation_masks"),
            ]:
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path:
                break

        dataset.append(dict(
            full_path = full_path,
            base_name = stem,
            category  = category,
            source    = ann.get("original_prompt", ""),
            target    = ann.get("editing_prompt",  ""),
            mask_path = mask_path,
        ))

    print(f"[GPU {args.gpu_id}] Chunk {args.chunk_idx}/{args.num_chunks} "
          f"— {len(dataset)} images ({start}→{end-1})")
    return dataset

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CAG GUIDANCE
# ══════════════════════════════════════════════════════════════════════════════

def get_lambda(strategy, t, vc_s, vc_t, mask_area=1.0,
               conflict_history=None, entropy_history=None):

    base  = HP["base_cfg"]
    kappa = HP["kappa"]
    m     = HP["m"]
    ts    = HP["t_start"]

    delta = torch.sqrt(torch.mean((vc_t - vc_s) ** 2)).item()
    s_d   = math.tanh(delta / m)
    g_t   = float(np.clip((ts - t) / ts, 0.0, 1.0))

    # safe exp — clamp argument to avoid OverflowError
    def safe_exp(x): return math.exp(max(-500.0, min(500.0, float(x))))
    def sig(x, k=12.0): return 1.0 / (1.0 + safe_exp(-k * (x - 0.5)))
    g_sig = sig(g_t)

    lam = base

    if strategy == "Baseline":
        lam = base

    elif strategy == "Exp_Forward_Gate":
        lam = base * safe_exp(kappa * s_d * (2 * g_sig - 1))

    elif strategy == "Exp_Forward":
        lam = base * safe_exp(kappa * s_d * (2 * (1 - t) - 1))

    elif strategy == "Exp_Reverse_Gate":
        lam = base * safe_exp(kappa * s_d * (2 * (1 - g_sig) - 1))

    elif strategy == "Exp_Boost":
        median = float(np.median(conflict_history)) if conflict_history else delta
        b      = 1.5 if delta > median else 1.0
        lam    = base * safe_exp(kappa * math.tanh(b * delta / m) * (2 * g_sig - 1))

    elif strategy == "TwoPhase":
        ta, tb = 0.5, 0.2
        if   t >= ta: h = 0.0
        elif t <= tb: h = 1.0
        else:         h = (ta - t) / (ta - tb)
        lam = base * safe_exp(kappa * s_d * (2 * h - 1))

    elif strategy == "Entropy":
        ent   = float(vc_t.var().item())
        arr   = np.array(entropy_history + [ent]) if entropy_history else np.array([ent])
        med_e = float(np.median(arr))
        mad_e = float(np.median(np.abs(arr - med_e))) + 1e-6
        u_k   = 1.0 / (1.0 + safe_exp(-((ent - med_e) / mad_e)))
        g_ent = g_sig * (1.0 - u_k)
        lam   = base * safe_exp(kappa * s_d * (2 * g_ent - 1))
        if entropy_history is not None:
            entropy_history.append(ent)

    elif strategy == "Locality":
        gamma = 0.5
        s_loc = math.tanh(delta / m * (1.0 + gamma * (1.0 - mask_area)))
        lam   = base * safe_exp(kappa * s_loc * (2 * g_sig - 1))

    elif strategy == "Dual":
        s_e = s_d
        s_p = 1.0 - s_e
        q   = g_sig * s_e - (1.0 - g_sig) * s_p
        lam = base * safe_exp(kappa * q)

    elif strategy == "Quantile":
        buf = np.array(conflict_history + [delta]) if conflict_history else np.array([delta])
        r_k = float((buf < delta).mean())
        s_q = 2.0 * r_k - 1.0
        lam = base * safe_exp(kappa * s_q * (2 * g_sig - 1))

    elif strategy == "Budget_Matched":
        lam = base * safe_exp(2.1 * s_d * (2 * g_sig - 1))

    return float(np.clip(lam, HP["lambda_min"], HP["lambda_max"]))

# ══════════════════════════════════════════════════════════════════════════════
# 4.  METRIC EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to("cuda:0").eval()
        self.clip_proc  = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=True)
        self.lpips_fn   = lpips_lib.LPIPS(net="vgg").to("cuda:0").eval()
        self.to_tensor  = transforms.ToTensor()

    @torch.no_grad()
    def compute(self, src, gen, prompt, mask=None):
        if src.size != gen.size:
            gen = gen.resize(src.size)

        inp    = self.clip_proc(text=[prompt], images=gen,
                                return_tensors="pt", padding=True).to("cuda:0")
        clip_w = self.clip_model(**inp).logits_per_image.item()

        clip_e = float("nan")
        if mask is not None:
            m_np = np.array(mask.resize(gen.size).convert("L")) / 255.0
            if m_np.sum() > 100:
                overlay = (np.array(gen) * np.stack([m_np]*3, axis=-1)).astype(np.uint8)
                inp_e   = self.clip_proc(text=[prompt],
                                         images=Image.fromarray(overlay),
                                         return_tensors="pt", padding=True).to("cuda:0")
                clip_e  = self.clip_model(**inp_e).logits_per_image.item()

        inp_i  = self.clip_proc(images=[src, gen], return_tensors="pt").to("cuda:0")
        feats  = F.normalize(self.clip_model.get_image_features(**inp_i), dim=-1)
        clip_i = float((feats[0] * feats[1]).sum().item())

        s = np.array(src.resize((512, 512))).astype(np.float32) / 255.0
        g = np.array(gen.resize((512, 512))).astype(np.float32) / 255.0

        mse    = float(np.mean((s - g) ** 2)) * 1e4
        psnr   = float(peak_signal_noise_ratio(s, g, data_range=1.0))
        ssim   = float(structural_similarity(
                     s, g, data_range=1.0, channel_axis=2, win_size=7)) * 100.0
        lpips_v = self.lpips_fn(
            self.to_tensor(src).to("cuda:0").unsqueeze(0) * 2 - 1,
            self.to_tensor(gen).to("cuda:0").unsqueeze(0) * 2 - 1,
        ).item() * 1e3

        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        lw = np.array([0.299, 0.587, 0.114])
        ls, lg = s @ lw, g @ lw
        gs = np.sqrt(convolve(ls, kx)**2 + convolve(ls, kx.T)**2 + 1e-8)
        gg = np.sqrt(convolve(lg, kx)**2 + convolve(lg, kx.T)**2 + 1e-8)
        sd = float(np.mean(np.abs(gs - gg) / (gs + gg + 1e-6))) * 1e3

        return dict(clip_w=clip_w, clip_e=clip_e, clip_i=clip_i,
                    mse=mse, psnr=psnr, ssim=ssim, lpips=lpips_v, sd=sd)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    def __init__(self):
        print(f"[GPU {args.gpu_id}] Chargement SD3-medium (visible={args.gpu_id}) …")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=DTYPE,
        ).to("cuda:0")
        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

    def encode_source(self, init_image):
        img_t  = self.pipe.image_processor.preprocess(init_image).to("cuda:0", dtype=torch.float32)
        x0_src = (
            self.pipe.vae.encode(img_t).latent_dist.mode()
            - self.pipe.vae.config.shift_factor
        ) * self.pipe.vae.config.scaling_factor
        return x0_src.to(DTYPE)

    def encode_prompts(self, src_p, tgt_p):
        # T5 → GPU uniquement pour l'encodage, puis CPU pour libérer ~10GB VRAM
        if hasattr(self.pipe, 'text_encoder_3') and self.pipe.text_encoder_3 is not None:
            self.pipe.text_encoder_3.to("cuda:0")
        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(src_p, src_p, src_p, device="cuda:0")
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(tgt_p, tgt_p, tgt_p, device="cuda:0")
        if hasattr(self.pipe, 'text_encoder_3') and self.pipe.text_encoder_3 is not None:
            self.pipe.text_encoder_3.to("cpu")
        torch.cuda.empty_cache()
        return (pe_s, ne_s, ppe_s, npe_s), (pe_t, ne_t, ppe_t, npe_t)

    @torch.no_grad()
    def edit(self, x0_src, src_embeds, tgt_embeds, strategy, mask_area=1.0):
        pe_s, ne_s, ppe_s, npe_s = src_embeds
        pe_t, ne_t, ppe_t, npe_t = tgt_embeds

        t_start     = HP["t_start"]
        steps       = HP["steps"]
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps   = self.pipe.scheduler.timesteps
        start_index = int((1.0 - t_start) * steps)

        zt = x0_src.clone()
        conflict_history, entropy_history, lambda_history = [], [], []

        strat_seed = 42 + hash(strategy) % 10000
        rng = torch.Generator(device="cuda:0").manual_seed(strat_seed)

        for i, t_tensor in enumerate(timesteps):
            if i < start_index:
                continue

            t  = t_tensor.item() / 1000.0
            dt = (timesteps[i+1].item()/1000.0 if i+1 < len(timesteps) else 0.0) - t

            noise  = torch.randn(*x0_src.shape, generator=rng,
                                 device="cuda:0", dtype=x0_src.dtype)
            zt_src = (1 - t) * x0_src + t * noise
            zt_tar = zt + zt_src - x0_src

            pred = self.pipe.transformer(
                hidden_states         = torch.cat([zt_src]*2 + [zt_tar]*2),
                timestep              = t_tensor.expand(4).to("cuda:0"),
                encoder_hidden_states = torch.cat([ne_s, pe_s, ne_t, pe_t]),
                pooled_projections    = torch.cat([npe_s, ppe_s, npe_t, ppe_t]),
                return_dict           = False,
            )[0]

            vu_s, vc_s, vu_t, vc_t = pred.chunk(4)

            lam = get_lambda(strategy, t, vc_s, vc_t,
                             mask_area=mask_area,
                             conflict_history=conflict_history,
                             entropy_history=entropy_history)
            lambda_history.append(lam)
            conflict_history.append(torch.sqrt(torch.mean((vc_t - vc_s)**2)).item())

            v_src = vu_s + HP["cfg_src"] * (vc_s - vu_s)
            v_tgt = vu_t + lam           * (vc_t - vu_t)
            zt    = zt   + dt            * (v_tgt - v_src)

        zt_dec = (zt / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        raw    = self.pipe.vae.decode(zt_dec.to("cuda:0", dtype=torch.float32), return_dict=False)[0]
        result = self.pipe.image_processor.postprocess(
            torch.clamp(raw, -1.0, 1.0), output_type="pil")[0]

        return result, float(np.mean(lambda_history))

# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dataset   = load_dataset_chunk()
    pipe      = Pipeline()
    evaluator = Evaluator()

    csv_path   = os.path.join(OUTPUT_DIR, f"results_gpu{args.gpu_id}_chunk{args.chunk_idx}.csv")
    fieldnames = ["GPU","Chunk","Image","Category","Strategy",
                  "CLIP_W","CLIP_E","CLIP_I",
                  "LPIPS_1e3","MSE_1e4","PSNR_dB","SSIM_100","StructDist_1e3","Lambda_mean"]

    col_hdr = (f"  {'Strategy':<22} | {'λ':>5} | {'CLIP-W':>6} | {'CLIP-I':>6} | "
               f"{'PSNR':>5} | {'SSIM':>5} | {'LPIPS':>6} | {'MSE':>6} | {'SD':>6}")
    col_sep = "  " + "-"*22 + "-+-" + ("-"*6+"-+-")*8

    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(dataset):
            t0 = time.time()
            print(f"\n[GPU {args.gpu_id}] [{idx+1}/{len(dataset)}]  "
                  f"{item['base_name']}  ({item['category']})", flush=True)
            print(col_hdr); print(col_sep)

            init_img         = load_image(item["full_path"]).resize((HP["img_size"], HP["img_size"]))
            x0_src           = pipe.encode_source(init_img)
            src_emb, tgt_emb = pipe.encode_prompts(item["source"], item["target"])

            mask_img, mask_area = None, 1.0
            if item["mask_path"] and os.path.exists(item["mask_path"]):
                mask_img  = Image.open(item["mask_path"]).convert("L")
                mask_area = float(np.mean(np.array(mask_img) / 255.0))

            for strat in ALL_STRATEGIES:
                set_seed(42)
                gen_img, lam_mean = pipe.edit(x0_src, src_emb, tgt_emb, strat, mask_area)
                m = evaluator.compute(init_img, gen_img, item["target"], mask_img)

                gen_img.save(os.path.join(OUTPUT_DIR, f"{item['base_name']}_{strat}.jpg"))
                writer.writerow({
                    "GPU": args.gpu_id, "Chunk": args.chunk_idx,
                    "Image": item["base_name"], "Category": item["category"],
                    "Strategy": strat,
                    "CLIP_W":  f"{m['clip_w']:.4f}", "CLIP_E":  f"{m['clip_e']:.4f}",
                    "CLIP_I":  f"{m['clip_i']:.4f}", "LPIPS_1e3": f"{m['lpips']:.4f}",
                    "MSE_1e4": f"{m['mse']:.4f}",   "PSNR_dB":  f"{m['psnr']:.4f}",
                    "SSIM_100":f"{m['ssim']:.4f}",  "StructDist_1e3": f"{m['sd']:.4f}",
                    "Lambda_mean": f"{lam_mean:.4f}",
                })
                fout.flush()

                print(f"  {strat:<22} | {lam_mean:5.1f} | "
                      f"{m['clip_w']:6.2f} | {m['clip_i']:6.3f} | "
                      f"{m['psnr']:5.2f} | {m['ssim']:5.2f} | "
                      f"{m['lpips']:6.1f} | {m['mse']:6.1f} | {m['sd']:6.1f}",
                      flush=True)

                del gen_img
                torch.cuda.empty_cache()
                gc.collect()

            # Libérer tenseurs partagés avant la prochaine image
            for t in list(src_emb) + list(tgt_emb): del t
            del x0_src, src_emb, tgt_emb
            torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - t0
            print(f"  → {elapsed:.0f}s ({elapsed/len(ALL_STRATEGIES):.0f}s/stratégie)", flush=True)

    print(f"\n[GPU {args.gpu_id}] Terminé → {csv_path}")


if __name__ == "__main__":
    main()