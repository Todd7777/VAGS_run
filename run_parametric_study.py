import os
import torch
import itertools
import csv
import numpy as np
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import lpips
from torchvision import transforms

# ==================== CONFIGURATION ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

# On teste sur l'image la plus difficile (Ours) car qui peut le plus peut le moins
INPUT_PATH = "Data/Images/bear_grass.png" 
if not os.path.exists(INPUT_PATH): INPUT_PATH = "Data/Images/flowedit_data/bear_grass.png"

OUTPUT_DIR = "outputs/wang_parametric_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCE_PROMPT = "A large brown bear walking through a grassy field."
TARGET_PROMPT = "A large polar bear with thick white fur walking through a green grassy field."

# ==================== GRILLE D'OPTIMISATION WANG ====================
GRID_PARAMS = {
    # Kappa : La puissance de l'exponentielle
    # 1.0 = Doux, 2.0 = Fort (notre test précédent), 3.0 = Très Agressif
    "kappa": [1.5, 2.0, 2.5, 3.0], 
    
    # M : Le diviseur dans le tanh (Sensibilité)
    # 3.0 = Sature vite (réactif), 5.0 = Standard, 8.0 = Progressif
    "m": [3.0, 5.0, 8.0],
    
    # T_start : Le moment de départ
    "t_start": [0.80, 0.85, 0.90]
}

# ==================== EVALUATEUR ====================
class MetricEvaluator:
    def __init__(self, device):
        print("[METRICS] Loading CLIP & LPIPS...")
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
        self.to_tensor = transforms.ToTensor()

    def get_clip_score(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(): outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item() / 100.0 

    def get_lpips_distance(self, img_source, img_generated):
        t_src = self.to_tensor(img_source).to(self.device) * 2 - 1
        t_gen = self.to_tensor(img_generated).to(self.device) * 2 - 1
        with torch.no_grad(): dist = self.lpips_loss(t_src.unsqueeze(0), t_gen.unsqueeze(0))
        return dist.item()

# ==================== MOTEUR WANG ====================
class WangEditor:
    def __init__(self):
        print(f"[INIT] Loading SD3 Pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=DTYPE
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)

    def get_sigmoidal_gate(self, t, t_start):
        if t > t_start: return 0.0
        progress = (t_start - t) / t_start 
        # Centré à 0.5 pour avoir une suppression au début (-1) et un boost à la fin (+1)
        return 1 / (1 + np.exp(-12 * (progress - 0.5))) 

    @torch.no_grad()
    def process(self, init_image, kappa, m, t_start):
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x0_src = (self.pipe.vae.encode(img_t).latent_dist.mode() - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        x0_src = x0_src.to(dtype=DTYPE)

        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(SOURCE_PROMPT, SOURCE_PROMPT, SOURCE_PROMPT, device=DEVICE)
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(TARGET_PROMPT, TARGET_PROMPT, TARGET_PROMPT, device=DEVICE)
        
        # On fixe le CFG de base à 7.5 pour isoler l'effet de Wang
        base_cfg = 7.5 
        steps = 50
        
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = self.pipe.scheduler.timesteps
        start_index = int((1.0 - t_start) * steps)
        
        zt = x0_src.clone()

        for i, t_tensor in enumerate(timesteps):
            if i < start_index: continue
            
            t = t_tensor.item() / 1000.0
            dt = (timesteps[i+1].item() / 1000.0 if i+1 < len(timesteps) else 0.0) - t
            noise = torch.randn_like(x0_src)
            zt_src = (1 - t) * x0_src + t * noise
            zt_tar = zt + zt_src - x0_src 

            latents_in = torch.cat([zt_src]*2 + [zt_tar]*2)
            prompt_embeds = torch.cat([ne_s, pe_s, ne_t, pe_t])
            pooled_embeds = torch.cat([npe_s, ppe_s, npe_t, ppe_t])

            noise_pred = self.pipe.transformer(
                hidden_states=latents_in, timestep=t_tensor.expand(latents_in.shape[0]), 
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled_embeds, return_dict=False
            )[0]
            
            vu_s, vc_s, vu_t, vc_t = noise_pred.chunk(4)

            # --- FORMULE DE WANG ---
            v_s_vec = vc_s
            v_t_vec = vc_t
            diff = v_t_vec - v_s_vec
            norm_diff = torch.norm(diff, p=2, dim=(1,2,3)).mean().item()
            norm_src = torch.norm(v_s_vec, p=2, dim=(1,2,3)).mean().item()
            relative_conflict = norm_diff / (norm_src + 1e-5)

            sigma = self.get_sigmoidal_gate(t, t_start)
            
            # Terme de contrôle : -1 (début) -> +1 (fin)
            control_term = 2 * sigma - 1 
            
            # Score de conflit borné par tanh
            conflict_score = np.tanh(relative_conflict / m)
            
            # Modulation Exponentielle
            modulation = np.exp(kappa * control_term * conflict_score)
            current_lambda = base_cfg * modulation

            v_source_final = vu_s + 3.5 * (vc_s - vu_s)
            v_target_final = vu_t + current_lambda * (vc_t - vu_t)
            
            zt = zt + dt * (v_target_final - v_source_final)

        zt_dec = (zt / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(zt_dec.to(torch.float32), return_dict=False)[0]
        image = torch.clamp(image, -1.0, 1.0)
        res_img = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        return res_img

if __name__ == "__main__":
    editor = WangEditor()
    evaluator = MetricEvaluator(DEVICE)
    
    init_img = load_image(INPUT_PATH).resize((1024,1024))
    
    keys, values = zip(*GRID_PARAMS.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"[START] Etude Paramétrique Wang : {len(combinations)} configurations.")
    
    csv_path = os.path.join(OUTPUT_DIR, "wang_optimization_results.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['kappa', 'm', 't_start', 'CLIP', 'LPIPS', 'Filename']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, config in enumerate(combinations):
            print(f"[{i+1}/{len(combinations)}] Kappa={config['kappa']} | M={config['m']} | t={config['t_start']}")
            
            res = editor.process(init_img, **config)
            
            clip_s = evaluator.get_clip_score(res, TARGET_PROMPT)
            lpips_d = evaluator.get_lpips_distance(init_img, res)
            
            print(f"   -> CLIP: {clip_s:.4f} | LPIPS: {lpips_d:.4f}")
            
            fname = f"wang_k{config['kappa']}_m{config['m']}_t{config['t_start']}.jpg"
            res.save(os.path.join(OUTPUT_DIR, fname))
            
            writer.writerow({
                'kappa': config['kappa'],
                'm': config['m'],
                't_start': config['t_start'],
                'CLIP': f"{clip_s:.4f}",
                'LPIPS': f"{lpips_d:.4f}",
                'Filename': fname
            })
            csv_file.flush()
            torch.cuda.empty_cache()

    print(f"\n[DONE] Résultats dans {OUTPUT_DIR}")