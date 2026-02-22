import os
import torch
import numpy as np
import csv
import yaml
import pandas as pd
import random
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import lpips
from torchvision import transforms
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

# Ensure deterministic results for scientific comparisons
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Hardware setup: SD3 Medium runs best in Float16 on CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

# Data paths using the standard DIV2K structure
YAML_PATH = "Data/flowedit.yaml"
IMAGES_ROOT = "Data/Images"
OUTPUT_DIR = "outputs/DIV2K_SOTA_EVAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MasterEvaluator:
    """
    Centralized metrics engine for SOTA tables.
    Calculates semantic alignment (CLIP), visual quality (FID), and background preservation (LPIPS/MSE/StructDist).
    """
    def __init__(self, device):
        self.device = device
        # CLIP-ViT-L/14 is the standard for high-resolution semantic similarity
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # Perceptual loss (VGG-based) used to measure how much the background 'feels' changed
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
        # FID measures the distribution distance between real and generated images
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        self.to_tensor = transforms.ToTensor()

    def get_clip_t_score(self, image, prompt):
        """Text-to-Image alignment: How well does the image follow the prompt?"""
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()

    def get_clip_t_edited(self, image, prompt, mask):
        """Measures CLIP similarity strictly inside the edited area defined by a mask."""
        if mask is None: return float('nan')
        img_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.resize(image.size).convert("L")) / 255.0
        mask_3d = np.stack([mask_np]*3, axis=-1)
        edited_region = Image.fromarray((img_np * mask_3d).astype(np.uint8))
        return self.get_clip_t_score(edited_region, prompt)

    def get_clip_i_score(self, img_source, img_generated):
        """Image-to-Image alignment: Measures if the global semantics are preserved."""
        inputs_src = self.clip_processor(images=img_source, return_tensors="pt").to(self.device)
        inputs_gen = self.clip_processor(images=img_generated, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat_src = self.clip_model.get_image_features(**inputs_src)
            feat_gen = self.clip_model.get_image_features(**inputs_gen)
        feat_src = feat_src / feat_src.norm(p=2, dim=-1, keepdim=True)
        feat_gen = feat_gen / feat_gen.norm(p=2, dim=-1, keepdim=True)
        return (feat_src @ feat_gen.T).item() * 100.0

    def get_lpips_distance(self, img_source, img_generated):
        """Perceptual distance: Standard scale is x1000 for academic tables."""
        t_src = self.to_tensor(img_source).to(self.device) * 2 - 1
        t_gen = self.to_tensor(img_generated).to(self.device) * 2 - 1
        with torch.no_grad():
            dist = self.lpips_loss(t_src.unsqueeze(0), t_gen.unsqueeze(0))
        return dist.item() * 1000.0

    def get_structural_metrics(self, img_source, img_generated):
        """Calculates pixel-wise and gradient-based background preservation metrics."""
        src = np.array(img_source.resize((512, 512))).astype(np.float32) / 255.0
        gen = np.array(img_generated.resize((512, 512))).astype(np.float32) / 255.0
        mse = float(np.mean((src - gen) ** 2)) * 10000.0
        psnr = float(peak_signal_noise_ratio(src, gen, data_range=1.0))
        ssim = float(structural_similarity(src, gen, data_range=1.0, channel_axis=2, win_size=7)) * 100.0
        
        # StructDist: Measures how much the image edges/gradients have moved
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        lum_w = np.array([0.299, 0.587, 0.114])
        lum_src = np.dot(src, lum_w); lum_gen = np.dot(gen, lum_w)
        gx_s = convolve(lum_src, kx); gy_s = convolve(lum_src, kx.T)
        gx_g = convolve(lum_gen, kx); gy_g = convolve(lum_gen, kx.T)
        grad_src = np.sqrt(gx_s**2 + gy_s**2 + 1e-8); grad_gen = np.sqrt(gx_g**2 + gy_g**2 + 1e-8)
        struct_dist = float(np.mean(np.abs(grad_src - grad_gen) / (grad_src + grad_gen + 1e-6))) * 1000.0
        return mse, psnr, ssim, struct_dist

    def update_fid(self, img_source, img_generated):
        """Buffers images to compute the Frechet Inception Distance at the end of the batch."""
        t_src = (self.to_tensor(img_source.resize((299, 299))) * 255).byte().unsqueeze(0).to(self.device)
        t_gen = (self.to_tensor(img_generated.resize((299, 299))) * 255).byte().unsqueeze(0).to(self.device)
        self.fid_metric.update(t_src, real=True); self.fid_metric.update(t_gen, real=False)

    def compute_fid(self):
        """Finalizes FID calculation and resets the state for the next strategy."""
        try:
            val = self.fid_metric.compute().item()
            self.fid_metric.reset()
            return val
        except: return float('nan')

def get_dynamic_cfg(strategy, t, t_start, vc_s, vc_t, base_cfg=7.5, kappa=4.0, m=3.0):
    """
    Implements the Dynamic Guidance Logic from Todd's PDF.
    Modulates the CFG scale based on the conflict signal (delta_k) and time progress.
    """
    # delta_k is the RMS of the difference between target and source predictions
    delta_k = torch.sqrt(torch.mean((vc_t - vc_s) ** 2)).item()
    s_delta = np.tanh(delta_k / m) # Modulation sensitivity
    
    # Linear time gates: g_t goes 0->1 (forward), g_rev goes 1->0 (reverse)
    g_t = np.clip((t_start - t) / t_start, 0.0, 1.0)
    g_rev = 1.0 - g_t
    
    if strategy == "Baseline": return base_cfg
    
    # Exponential CAG variants: Use time-gating to suppress noise early and boost details later
    if strategy == "Exp CAG (Forward+Gate)": 
        return np.clip(base_cfg * np.exp(kappa * s_delta * (2 * g_t - 1)), 1.0, 20.0)
    
    if strategy == "Exp CAG (Reverse+Gate)": 
        return np.clip(base_cfg * np.exp(kappa * s_delta * (2 * g_rev - 1)), 1.0, 20.0)
    
    if strategy == "Dual-Objective CAG":
        # Balances editing (s_delta) and preservation (1 - s_delta) across the timeline
        q = g_t * s_delta - (1.0 - g_t) * (1.0 - s_delta)
        return np.clip(base_cfg * np.exp(kappa * q), 1.0, 20.0)
    
    if strategy == "Two-Phase CAG":
        # Piecewise scheduling: Freeze early stages, Surge in the middle, Refine at the end
        h_t = 0.0 if t >= 0.7 else (1.0 if t <= 0.3 else (0.7 - t) / 0.4)
        return np.clip(base_cfg * np.exp(kappa * s_delta * (2 * h_t - 1)), 1.0, 20.0)
        
    return base_cfg

class ComparisonEditor:
    """Core SD3 Image-to-Image pipeline modified with flow-based guidance."""
    def __init__(self):
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=DTYPE
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)

    @torch.no_grad()
    def process(self, init_image, src_p, tgt_p, strategy):
        # Image encoding to latent space
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x0_src = (self.pipe.vae.encode(img_t).latent_dist.mode() - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        x0_src = x0_src.to(dtype=DTYPE)
        
        # Text encoding for both Source and Target prompts
        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(src_p, src_p, src_p, device=DEVICE)
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(tgt_p, tgt_p, tgt_p, device=DEVICE)
        
        t_start, steps = 0.85, 50
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = self.pipe.scheduler.timesteps
        zt = x0_src.clone() # Initialize target latent at source latent

        for i, t_tens in enumerate(timesteps):
            if i < int((1.0 - t_start) * steps): continue
            t_val = t_tens.item() / 1000.0
            dt = (timesteps[i+1].item()/1000.0 if i+1 < len(timesteps) else 0.0) - t_val
            
            # Flow-matching logic: generate zt_src (noise path) and align zt_tar
            zt_src = (1 - t_val) * x0_src + t_val * torch.randn_like(x0_src)
            zt_tar = zt + zt_src - x0_src
            
            lat_in = torch.cat([zt_src]*2 + [zt_tar]*2)
            p_emb = torch.cat([ne_s, pe_s, ne_t, pe_t])
            pool = torch.cat([npe_s, ppe_s, npe_t, ppe_t])
            
            # Transformer predicts the velocity/noise vector
            out = self.pipe.transformer(hidden_states=lat_in, timestep=t_tens.expand(4), encoder_hidden_states=p_emb, pooled_projections=pool, return_dict=False)[0]
            vu_s, vc_s, vu_t, vc_t = out.chunk(4)
            
            # Compute the dynamic Guidance Scale (lambda)
            cur_lam = get_dynamic_cfg(strategy, t_val, t_start, vc_s, vc_t)
            
            # Euler update for the target latent path
            zt = zt + dt * ((vu_t + cur_lam * (vc_t - vu_t)) - (vu_s + 3.5 * (vc_s - vu_s)))
            
        # Decode the final optimized latent back to pixel space
        res = self.pipe.vae.decode((zt / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(torch.clamp(res, -1.0, 1.0), output_type="pil")[0]

if __name__ == "__main__":
    # Load metadata from DIV2K YAML file
    with open(YAML_PATH, 'r') as f: data = list(yaml.safe_load_all(f))[0]
    editor, evaluator = ComparisonEditor(), MasterEvaluator(DEVICE)
    strats = ["Baseline", "Exp CAG (Forward+Gate)", "Exp CAG (Reverse+Gate)", "Dual-Objective CAG", "Two-Phase CAG"]
    
    csv_f = open(os.path.join(OUTPUT_DIR, "div2k_results.csv"), 'w', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=['Image', 'Strategy', 'CLIP_Whole', 'CLIP_Edited', 'CLIP_I', 'LPIPS', 'MSE', 'PSNR', 'SSIM', 'StructDist'])
    writer.writeheader()
    
    for strat in strats:
        for entry in data[:15]: # Process 15 images per strategy as a subset
            img_p = os.path.join(IMAGES_ROOT, os.path.basename(entry['init_img']))
            if not os.path.exists(img_p): continue
            
            # Load and process image
            img = load_image(img_p).resize((1024,1024))
            tgt = entry['target_prompts'][0]
            res = editor.process(img, entry['source_prompt'], tgt, strat)
            
            # Evaluate the results
            evaluator.update_fid(img, res)
            m = evaluator.get_structural_metrics(img, res)
            
            writer.writerow({
                'Image': os.path.basename(img_p), 
                'Strategy': strat, 
                'CLIP_Whole': evaluator.get_clip_t_score(res, tgt), 
                'CLIP_Edited': evaluator.get_clip_t_edited(res, tgt, None), 
                'CLIP_I': evaluator.get_clip_i_score(img, res), 
                'LPIPS': evaluator.get_lpips_distance(img, res), 
                'MSE': m[0], 'PSNR': m[1], 'SSIM': m[2], 'StructDist': m[3]
            })
            csv_f.flush()
            res.save(os.path.join(OUTPUT_DIR, f"{strat}_{os.path.basename(img_p)}"))
            
        # Calculate and print the FID score for the entire set of images under this strategy
        print(f"FID Final Score for {strat}: {evaluator.compute_fid()}")