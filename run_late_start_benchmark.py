import os
import torch
import pandas as pd
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# Diffusers & Models
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
import open_clip
import lpips

# ==================== CONFIGURATION ULTIME ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

# Chemins
YAML_PATH = "Data/flowedit.yaml"
IMAGES_ROOT = "Data/Images"
OUTPUT_DIR = "outputs/benchmark_ultimate_late_start" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== MOTEUR MÉTRIQUES ====================
class MetricsCalculator:
    def __init__(self):
        self.lpips = lpips.LPIPS(net='alex').to(DEVICE).eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.clip_model = self.clip_model.to(DEVICE).eval()
        self.to_tensor = T.ToTensor()

    @torch.no_grad()
    def compute(self, img_ref, img_out, target_txt):
        if img_ref.size != img_out.size: img_out = img_out.resize(img_ref.size)
        ref_t = self.to_tensor(img_ref).unsqueeze(0).to(DEVICE)
        out_t = self.to_tensor(img_out).unsqueeze(0).to(DEVICE)
        lp_score = self.lpips(T.Resize((256,256))(ref_t), T.Resize((256,256))(out_t)).item()
        
        image_input = self.clip_preprocess(img_out).unsqueeze(0).to(DEVICE)
        text_input = open_clip.tokenize([target_txt]).to(DEVICE)
        image_features = self.clip_model.encode_image(image_input)
        text_features = self.clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_score = (image_features @ text_features.T).item()
        return {"LPIPS": round(lp_score, 4), "CLIP": round(clip_score, 4)}

def get_dynamic_cfg(base_scale, progress, mode="constant"):
    if mode == "constant": return base_scale
    if mode == "increasing": return base_scale * (0.6 + 0.8 * progress) 
    return base_scale

# ==================== EDITEUR LATE START ====================
class HybridFlowEditor:
    def __init__(self):
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=DTYPE
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)
        self.metrics = MetricsCalculator()

    def get_embeddings(self, prompt):
        pe, ne, ppe, npe = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            device=DEVICE
        )
        return {"pe": pe, "ne": ne, "ppe": ppe, "npe": npe}

    @torch.no_grad()
    def process(self, init_image, source_prompt, target_prompt, 
                start_ratio=0.9, base_cfg=13.5, 
                inject_start=0.2, inject_end=1.0): # LES PARAMETRES GAGNANTS
        
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x0_src = (self.pipe.vae.encode(img_t).latent_dist.mode() - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        x0_src = x0_src.to(dtype=DTYPE)

        emb_src = self.get_embeddings(source_prompt)
        emb_tgt = self.get_embeddings(target_prompt)

        steps = 50
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = self.pipe.scheduler.timesteps
        start_index = int((1.0 - start_ratio) * steps)
        
        zt_edit = x0_src.clone()

        for i, t in enumerate(timesteps):
            if i < start_index: continue
            
            t_curr = t.item() / 1000.0
            t_next = timesteps[i+1].item() / 1000.0 if i+1 < len(timesteps) else 0.0
            dt = t_next - t_curr
            
            current_step_in_edit = i - start_index
            total_steps_in_edit = steps - start_index
            progress = current_step_in_edit / total_steps_in_edit if total_steps_in_edit > 0 else 1.0
            progress = max(0.0, min(1.0, progress))

            # --- LATE START LOGIC ---
            if inject_start <= progress <= inject_end:
                pe_active, ppe_active = emb_tgt["pe"], emb_tgt["ppe"]
                is_editing = True
            else:
                pe_active, ppe_active = emb_src["pe"], emb_src["ppe"]
                is_editing = False

            noise = torch.randn_like(x0_src)
            zt_src = (1 - t_curr) * x0_src + t_curr * noise
            zt_tar = zt_edit + zt_src - x0_src

            if is_editing:
                tgt_cfg = get_dynamic_cfg(base_cfg, progress, "increasing")
            else:
                tgt_cfg = 3.5 
            
            latents_in = torch.cat([zt_src]*2 + [zt_tar]*2)
            prompt_embeds = torch.cat([emb_src["ne"], emb_src["pe"], emb_src["ne"], pe_active])
            pooled_embeds = torch.cat([emb_src["npe"], emb_src["ppe"], emb_src["npe"], ppe_active])

            noise_pred = self.pipe.transformer(
                hidden_states=latents_in, timestep=t.expand(latents_in.shape[0]),
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled_embeds,
                return_dict=False
            )[0]

            src_u, src_t, tar_u, tar_t = noise_pred.chunk(4)
            v_src = src_u + 3.5 * (src_t - src_u)
            v_tar = tar_u + tgt_cfg * (tar_t - tar_u)

            zt_edit = zt_edit + dt * (v_tar - v_src)

        zt_dec = (zt_edit / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(zt_dec.to(torch.float32), return_dict=False)[0]
        image = torch.clamp(image, -1.0, 1.0)
        res_img = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        return res_img, self.metrics.compute(init_image, res_img, target_prompt)

# ==================== MAIN ====================
def run_ultimate_benchmark():
    if not os.path.exists(YAML_PATH): return

    with open(YAML_PATH, 'r') as f:
        full_data = yaml.safe_load(f)
        if isinstance(full_data, list) and isinstance(full_data[0], list):
            full_data = [item for sublist in full_data for item in sublist]
            
    editor = HybridFlowEditor()
    results = []
    
    # LA CONFIG CHAMPIONNE
    BEST_CONFIG = {
        "start_ratio": 0.9, 
        "base_cfg": 13.5,
        "inject_start": 0.2, 
        "inject_end": 1.0
    }
    
    print(f"[INFO] Lancement Benchmark ULTIME : {BEST_CONFIG}")
    
    total_ops = sum(len(e.get('target_prompts', [])) for e in full_data if e)
    pbar = tqdm(total=total_ops)
    
    for entry in full_data:
        if not entry: continue
        rel_path = entry.get('init_img') or entry.get('input_img')
        if not rel_path: 
            pbar.update(len(entry.get('target_prompts', [])))
            continue
            
        candidates = [rel_path, os.path.join(IMAGES_ROOT, os.path.basename(rel_path)), os.path.join("FlowEdit", rel_path)]
        img_path = next((p for p in candidates if os.path.exists(p)), None)
        
        if not img_path:
            pbar.update(len(entry.get('target_prompts', [])))
            continue

        try:
            init_img = load_image(img_path).resize((1024,1024))
        except:
            pbar.update(len(entry.get('target_prompts', [])))
            continue

        src_p = entry['source_prompt']
        for tgt_p in entry['target_prompts']:
            try:
                res_img, scores = editor.process(init_img, src_p, tgt_p, **BEST_CONFIG)
                
                # Save minimaliste
                h = abs(hash(tgt_p)) % 10000
                res_img.save(os.path.join(OUTPUT_DIR, f"{os.path.basename(img_path).split('.')[0]}_{h}.jpg"))
                
                results.append({
                    "Image": os.path.basename(img_path),
                    "Target": tgt_p[:30],
                    "LPIPS": scores['LPIPS'],
                    "CLIP": scores['CLIP']
                })
            except Exception as e:
                print(e)
            pbar.update(1)
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    if not df.empty:
        print("\n" + "="*60)
        print(" RÉSULTATS ULTIMES (LateStart 0.2) ")
        print("="*60)
        print(f"LPIPS Moyen : {df['LPIPS'].mean():.4f}")
        print(f"CLIP Moyen  : {df['CLIP'].mean():.4f}")
        df.to_csv(os.path.join(OUTPUT_DIR, "final_ultimate_results.csv"), index=False)

if __name__ == "__main__":
    run_ultimate_benchmark()