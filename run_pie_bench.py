import os
import gc
import torch
import numpy as np
import csv
import json
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

PIE_BENCH_ROOT = "Data/PIE-Bench_v1/annotation_images"
MAPPING_FILE = "Data/PIE-Bench_v1/mapping_file.json"
OUTPUT_DIR = "outputs/PIE_BENCH_EVAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pie_bench():
    image_index = {}
    for root, dirs, files in os.walk(PIE_BENCH_ROOT):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(file)[0]
                image_index[base_name] = os.path.join(root, file)
                
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
    
    dataset = []
    for key, annotations in data.items():
        src_prompt = annotations.get("original_prompt", "")
        tgt_prompt = annotations.get("editing_prompt", "")
        base_key = os.path.splitext(os.path.basename(key))[0]
        
        if base_key in image_index:
            full_path = image_index[base_key]
            category = os.path.basename(os.path.dirname(full_path))
            
            dataset.append({
                "full_path": full_path,
                "base_name": base_key,
                "category": category,
                "source": src_prompt,
                "target": tgt_prompt
            })
            
    return dataset

class MetricEvaluator:
    def __init__(self, device):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
        self.to_tensor = transforms.ToTensor()

    def get_clip_score(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(): 
            outputs = self.clip_model(**inputs)
        return (outputs.logits_per_image.item() / 100.0) * 100.0

    def get_lpips_distance(self, img_source, img_generated):
        t_src = self.to_tensor(img_source).to(self.device) * 2 - 1
        t_gen = self.to_tensor(img_generated).to(self.device) * 2 - 1
        with torch.no_grad(): 
            dist = self.lpips_loss(t_src.unsqueeze(0), t_gen.unsqueeze(0))
        return dist.item() * 1000.0

    def get_structural_metrics(self, img_source, img_generated):
        src = np.array(img_source.resize((512, 512))).astype(np.float32) / 255.0
        gen = np.array(img_generated.resize((512, 512))).astype(np.float32) / 255.0

        mse = float(np.mean((src - gen) ** 2)) * 10000.0
        psnr = float(peak_signal_noise_ratio(src, gen, data_range=1.0))
        ssim = float(structural_similarity(src, gen, data_range=1.0, channel_axis=2, win_size=7)) * 100.0

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        lum_w = np.array([0.299, 0.587, 0.114])
        lum_src = np.dot(src, lum_w)
        lum_gen = np.dot(gen, lum_w)
        gx_s = convolve(lum_src, kx)
        gy_s = convolve(lum_src, kx.T)
        gx_g = convolve(lum_gen, kx)
        gy_g = convolve(lum_gen, kx.T)
        grad_src = np.sqrt(gx_s**2 + gy_s**2 + 1e-8)
        grad_gen = np.sqrt(gx_g**2 + gy_g**2 + 1e-8)
        grad_diff = np.abs(grad_src - grad_gen) / (grad_src + grad_gen + 1e-6)
        struct_dist = float(np.mean(grad_diff)) * 1000.0

        return mse, psnr, ssim, struct_dist

class ComparisonEditor:
    def __init__(self):
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=DTYPE
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)

    @torch.no_grad()
    def process(self, init_image, src_prompt, tgt_prompt, strategy):
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x0_src = (self.pipe.vae.encode(img_t).latent_dist.mode() - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        x0_src = x0_src.to(dtype=DTYPE)

        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(src_prompt, src_prompt, src_prompt, device=DEVICE)
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(tgt_prompt, tgt_prompt, tgt_prompt, device=DEVICE)

        t_start  = 0.66
        base_cfg = 13.5
        kappa    = 4.0
        m        = 3.0
        steps    = 50

        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps   = self.pipe.scheduler.timesteps
        start_index = int((1.0 - t_start) * steps)

        zt = x0_src.clone()
        generator = torch.Generator(device=DEVICE).manual_seed(42)

        for i, t_tensor in enumerate(timesteps):
            if i < start_index: continue

            t  = t_tensor.item() / 1000.0
            dt = (timesteps[i+1].item() / 1000.0 if i+1 < len(timesteps) else 0.0) - t
            
            noise = torch.randn(*x0_src.shape, generator=generator, device=DEVICE, dtype=x0_src.dtype)
            
            zt_src = (1 - t) * x0_src + t * noise
            zt_tar = zt + zt_src - x0_src

            latents_in    = torch.cat([zt_src]*2 + [zt_tar]*2)
            prompt_embeds = torch.cat([ne_s, pe_s, ne_t, pe_t])
            pooled_embeds = torch.cat([npe_s, ppe_s, npe_t, ppe_t])

            noise_pred = self.pipe.transformer(
                hidden_states=latents_in,
                timestep=t_tensor.expand(latents_in.shape[0]),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False
            )[0]

            vu_s, vc_s, vu_t, vc_t = noise_pred.chunk(4)
            current_lambda = base_cfg

            if strategy == "Exponential_CAG":
                diff = vc_t - vc_s
                norm_diff = torch.norm(diff, p=2, dim=(1,2,3)).mean().item()
                norm_src = torch.norm(vc_s, p=2, dim=(1,2,3)).mean().item()
                relative_conflict = norm_diff / (norm_src + 1e-5)
                
                progress = (t_start - t) / t_start
                sigma = 1 / (1 + np.exp(-12 * (progress - 0.5)))
                control_term = 2 * sigma - 1
                conflict_score = np.tanh(relative_conflict / m)
                modulation = np.exp(kappa * control_term * conflict_score)
                current_lambda = base_cfg * modulation

            v_source_final = vu_s + 3.5 * (vc_s - vu_s)
            v_target_final = vu_t + current_lambda * (vc_t - vu_t)
            zt = zt + dt * (v_target_final - v_source_final)

        zt_dec  = (zt / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image   = self.pipe.vae.decode(zt_dec.to(torch.float32), return_dict=False)[0]
        image   = torch.clamp(image, -1.0, 1.0)
        res_img = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        return res_img

def analyze_and_print_results(csv_path):
    try:
        df = pd.read_csv(csv_path)
        metrics = ['CLIP', 'LPIPS', 'MSE_x1e4', 'PSNR_dB', 'SSIM_x100', 'StructDist_x1e3']
        summary = df.groupby('Strategy')[metrics].mean()
        print(summary.round(4).to_string())
        summary_path = csv_path.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path)
    except Exception as e:
        pass

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    DATASET = load_pie_bench()
    
    editor    = ComparisonEditor()
    evaluator = MetricEvaluator(DEVICE)

    STRATEGIES = ["Baseline", "Exponential_CAG"]
    MAX_IMAGES = len(DATASET)

    csv_path = os.path.join(OUTPUT_DIR, "pie_bench_results.csv")
    fieldnames = ['Image', 'Category', 'Strategy', 'CLIP', 'LPIPS', 'MSE_x1e4', 'PSNR_dB', 'SSIM_x100', 'StructDist_x1e3', 'Filename']

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(DATASET):
            if idx >= MAX_IMAGES: 
                break

            full_path = item["full_path"]

            if not os.path.exists(full_path):
                continue

            init_img  = load_image(full_path).resize((1024, 1024))
            base_name = item["base_name"]
            tgt_prompt = item["target"]

            for strat in STRATEGIES:
                
                res_img = editor.process(init_img, item["source"], tgt_prompt, strat)

                clip_s = evaluator.get_clip_score(res_img, tgt_prompt)
                lpips_d = evaluator.get_lpips_distance(init_img, res_img)
                mse, psnr, ssim, struct_dist = evaluator.get_structural_metrics(init_img, res_img)

                fname = f"{base_name}_{strat}.jpg"
                res_img.save(os.path.join(OUTPUT_DIR, fname))

                writer.writerow({
                    'Image':           base_name,
                    'Category':        item["category"],
                    'Strategy':        strat,
                    'CLIP':            f"{clip_s:.4f}",
                    'LPIPS':           f"{lpips_d:.4f}",
                    'MSE_x1e4':        f"{mse:.4f}",
                    'PSNR_dB':         f"{psnr:.4f}",
                    'SSIM_x100':       f"{ssim:.4f}",
                    'StructDist_x1e3': f"{struct_dist:.4f}",
                    'Filename':        fname,
                })
                csv_file.flush()
                
                del res_img
                torch.cuda.empty_cache()
                gc.collect()

    analyze_and_print_results(csv_path)