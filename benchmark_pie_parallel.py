import os, gc, json, csv, math, random, argparse
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import lpips as lpips_lib
from torchvision import transforms
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve

# ==========================================
# 1. CONFIGURATION & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Parallel PIE-Bench Evaluation")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--num_chunks", type=int, default=1)
parser.add_argument("--chunk_idx", type=int, default=0)
parser.add_argument("--test_run", action="store_true")
args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
DEVICE = torch.device("cuda")
DTYPE = torch.float16 

PIE_BENCH_ROOT = "Data/PIE-Bench_v1/annotation_images"
MASK_ROOT      = "Data/PIE-Bench_v1/annotation_masks"
MAPPING_FILE   = "Data/PIE-Bench_v1/mapping_file.json"
OUTPUT_DIR     = "outputs/PIE_BENCH_FULL_SOTA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. INDEXEUR & EVALUATEUR
# ==========================================
def build_image_index(root_dir):
    index = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                index[os.path.splitext(f)[0]] = os.path.join(root, f)
    return index

class MasterEvaluator:
    def __init__(self, device):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.lpips_loss = lpips_lib.LPIPS(net='vgg').to(device).eval()
        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def compute_all(self, src_img, gen_img, target_prompt, mask=None):
        if src_img.size != gen_img.size: gen_img = gen_img.resize(src_img.size)
        inputs = self.clip_processor(text=[target_prompt], images=gen_img, return_tensors="pt", padding=True).to(self.device)
        clip_w = self.clip_model(**inputs).logits_per_image.item()
        
        clip_e = float('nan')
        if mask is not None:
            m_np = np.array(mask.resize(gen_img.size).convert("L")) / 255.0
            if m_np.sum() > 0:
                ed_reg = Image.fromarray((np.array(gen_img) * np.stack([m_np]*3, axis=-1)).astype(np.uint8))
                inputs_ed = self.clip_processor(text=[target_prompt], images=ed_reg, return_tensors="pt", padding=True).to(self.device)
                clip_e = self.clip_model(**inputs_ed).logits_per_image.item()

        s, g = np.array(src_img.resize((512,512)))/255., np.array(gen_img.resize((512,512)))/255.
        mse = float(np.mean((s - g)**2)) * 10000.0
        ssim = float(structural_similarity(s, g, data_range=1.0, channel_axis=2)) * 100.0
        lpips_v = self.lpips_loss(self.to_tensor(src_img).to(self.device).unsqueeze(0)*2-1, 
                                  self.to_tensor(gen_img).to(self.device).unsqueeze(0)*2-1).item() * 1000.0
        return clip_w, clip_e, mse, ssim, lpips_v

# ==========================================
# 3. CAG ENGINE (TODD'S 11 STRATEGIES)
# ==========================================
def get_cag_lambda(strategy, t, t_start, vc_s, vc_t, mask_area=1.0):
    cfg_base, kappa, m = 13.5, 4.0, 3.0
    delta = torch.sqrt(torch.mean((vc_t - vc_s)**2)).item()
    s_d = np.tanh(delta / m)
    g_t = np.clip((t_start - t) / t_start, 0.0, 1.0)
    
    if strategy == "Baseline": return cfg_base
    if strategy == "Exp_Forward_Gate": return cfg_base * np.exp(kappa * s_d * (2 * g_t - 1))
    if strategy == "Exp_Forward":      return cfg_base * np.exp(kappa * s_d * (2 * (1-t) - 1))
    if strategy == "Exp_Reverse_Gate": return cfg_base * np.exp(kappa * s_d * (2 * (1-g_t) - 1))
    if strategy == "Exp_Boost":        return cfg_base * np.exp(kappa * np.tanh(1.5 * delta / m) * (2 * g_t - 1))
    if strategy == "TwoPhase":
        h = 0.0 if t >= 0.5 else (1.0 if t <= 0.2 else (0.5 - t) / 0.3)
        return cfg_base * np.exp(kappa * s_d * (2 * h - 1))
    if strategy == "Entropy":
        ent = float(vc_t.var().item())
        return cfg_base * np.exp(kappa * s_d * (1.0 - np.tanh(ent)))
    if strategy == "Locality":
        s_loc = np.tanh((delta/m) * (1.0 + 0.5*(1.0-mask_area)))
        return cfg_base * np.exp(kappa * s_loc * (2 * g_t - 1))
    if strategy == "Dual":
        q = g_t * s_d - (1.0 - g_t) * (1.0 - s_d)
        return cfg_base * np.exp(kappa * q)
    if strategy == "Quantile":
        return cfg_base * np.exp(kappa * (2 * s_d - 1) * g_t)
    if strategy == "Budget_Matched":
        return cfg_base * np.exp(2.1 * s_d * (2 * g_t - 1))
    return cfg_base

# ==========================================
# 4. RUNNER
# ==========================================
class PIEBenchRunner:
    def __init__(self):
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=DTYPE)
        self.pipe.enable_model_cpu_offload()
        self.evaluator = MasterEvaluator(DEVICE)

    @torch.no_grad()
    def edit(self, init_image, src_p, tgt_p, strategy, mask_area=1.0):
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=DTYPE)
        x0 = (self.pipe.vae.encode(img_t).latent_dist.mode() - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(src_p, src_p, src_p, device=DEVICE)
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(tgt_p, tgt_p, tgt_p, device=DEVICE)
        
        t_start, steps = 0.66, 50
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps, zt = self.pipe.scheduler.timesteps, x0.clone()

        for i, t_tens in enumerate(timesteps):
            if i < int((1.0 - t_start) * steps): continue
            t, dt = t_tens.item() / 1000.0, (timesteps[i+1].item()/1000.0 if i+1 < len(timesteps) else 0.0) - (t_tens.item()/1000.0)
            zt_src = (1 - t) * x0 + t * torch.randn_like(x0)
            zt_tar = zt + zt_src - x0
            out = self.pipe.transformer(hidden_states=torch.cat([zt_src]*2 + [zt_tar]*2), timestep=t_tens.expand(4), 
                                        encoder_hidden_states=torch.cat([ne_s, pe_s, ne_t, pe_t]), 
                                        pooled_projections=torch.cat([npe_s, ppe_s, npe_t, ppe_t]), return_dict=False)[0]
            vu_s, vc_s, vu_t, vc_t = out.chunk(4)
            lam = np.clip(get_cag_lambda(strategy, t, t_start, vc_s, vc_t, mask_area), 1.0, 40.0)
            zt = zt + dt * ((vu_t + lam * (vc_t - vu_t)) - (vu_s + 3.5 * (vc_s - vu_s)))

        res = self.pipe.vae.decode((zt / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(torch.clamp(res, -1.0, 1.0), output_type="pil")[0]

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    img_idx = build_image_index(PIE_BENCH_ROOT)
    with open(MAPPING_FILE, 'r') as f: mapping = list(json.load(f).items())
    
    chunk_size = math.ceil(len(mapping) / args.num_chunks)
    sub_map = mapping[args.chunk_idx * chunk_size : (args.chunk_idx + 1) * chunk_size]
    if args.test_run: sub_map = sub_map[:2]

    runner = PIEBenchRunner()
    strats = ["Baseline", "Exp_Forward_Gate", "Exp_Forward", "Exp_Reverse_Gate", "Exp_Boost", "TwoPhase", "Entropy", "Locality", "Dual", "Quantile", "Budget_Matched"]
    
    csv_path = os.path.join(OUTPUT_DIR, f"results_gpu{args.gpu_id}_chunk{args.chunk_idx}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Strategy', 'Image', 'CLIP_W', 'CLIP_E', 'MSE_1e4', 'SSIM_100', 'LPIPS_1e3'])
        writer.writeheader()

        for key, ann in sub_map:
            if key not in img_idx: continue
            img_path = img_idx[key]
            src_img = load_image(img_path).resize((1024, 1024))
            mask_img = load_image(img_path.replace("annotation_images", "annotation_masks")) if os.path.exists(img_path.replace("annotation_images", "annotation_masks")) else None
            m_area = np.mean(np.array(mask_img.convert("L"))/255.0) if mask_img else 1.0

            for s in strats:
                print(f"GPU {args.gpu_id} | {key} | {s}")
                gen = runner.edit(src_img, ann["original_prompt"], ann["editing_prompt"], s, m_area)
                m = runner.evaluator.compute_all(src_img, gen, ann["editing_prompt"], mask_img)
                writer.writerow({'Strategy': s, 'Image': key, 'CLIP_W': f"{m[0]:.4f}", 'CLIP_E': f"{m[1]:.4f}", 'MSE_1e4': f"{m[2]:.4f}", 'SSIM_100': f"{m[3]:.4f}", 'LPIPS_1e3': f"{m[4]:.4f}"})
                f.flush()
                gen.save(os.path.join(OUTPUT_DIR, f"{s}_{key}.jpg"))
                del gen; torch.cuda.empty_cache(); gc.collect()