import os
import torch
import pandas as pd
import numpy as np
import yaml
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
import open_clip
import lpips
import torchvision.transforms as T
import itertools

# ==================== CONFIGURATION ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

# Chemins
YAML_PATH = "Data/flowedit.yaml"
IMAGES_ROOT = "Data/Images"
OUTPUT_DIR = "outputs/sign_flip_optimization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== METRICS ====================
class MetricsCalculator:
    def __init__(self):
        self.lpips = lpips.LPIPS(net='alex').to(DEVICE).eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.clip_model = self.clip_model.to(DEVICE).eval()
        self.to_tensor = T.ToTensor()

    @torch.no_grad()
    def compute(self, img_ref, img_out, target_txt):
        if img_ref.size != img_out.size: img_out = img_out.resize(img_ref.size)
        
        # Cast safe FP32
        ref_t = self.to_tensor(img_ref).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        out_t = self.to_tensor(img_out).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        lp_score = self.lpips(T.Resize((256,256))(ref_t), T.Resize((256,256))(out_t)).item()
        
        image_input = self.clip_preprocess(img_out).unsqueeze(0).to(DEVICE)
        text_input = open_clip.tokenize([target_txt]).to(DEVICE)
        image_features = self.clip_model.encode_image(image_input)
        text_features = self.clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_score = (image_features @ text_features.T).item()
        return {"LPIPS": round(lp_score, 4), "CLIP": round(clip_score, 4)}

# ==================== SIGN FLIP EDITOR ====================
class SignFlipEditor:
    def __init__(self):
        print("[INIT] Loading SD3 (FP16)...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=DTYPE
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)
        self.metrics = MetricsCalculator()

    def get_embeddings(self, prompt):
        pe, ne, ppe, npe = self.pipe.encode_prompt(prompt, prompt, prompt, device=DEVICE)
        return {k: v.to(dtype=DTYPE) for k, v in zip(["pe", "ne", "ppe", "npe"], [pe, ne, ppe, npe])}

    @torch.no_grad()
    def process(self, init_image, source_prompt, target_prompt, 
                t0, beta_e, beta_l, w_min=3.5, w_max=25.0):
        
        # Setup
        start_ratio = 0.9 # On fixe le start à 0.9 (Régime Unlocked)
        
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

        avg_w = 0
        
        for i, t in enumerate(timesteps):
            if i < start_index: continue
            
            t_curr = t.item() / 1000.0
            t_next = timesteps[i+1].item() / 1000.0 if i+1 < len(timesteps) else 0.0
            dt = t_next - t_curr
            
            # Progress : 0.0 (Début) -> 1.0 (Fin)
            progress = (i - start_index) / (steps - start_index) if (steps-start_index) > 0 else 1.0
            progress = max(0.0, min(1.0, progress))

            noise = torch.randn_like(x0_src)
            zt_src = (1 - t_curr) * x0_src + t_curr * noise
            zt_tar = zt_edit + zt_src - x0_src

            # On a besoin de v_source et v_target pour calculer s(t)
            latents_in = torch.cat([zt_src]*2 + [zt_tar]*2)
            
            # On conditionne : Src_Neg, Src_Pos, Tgt_Neg, Tgt_Pos
            # On utilise le prompt cible pour zt_tar, mais on a besoin de mesurer l'écart
            prompt_embeds = torch.cat([emb_src["ne"], emb_src["pe"], emb_tgt["ne"], emb_tgt["pe"]])
            pooled_embeds = torch.cat([emb_src["npe"], emb_src["ppe"], emb_tgt["npe"], emb_tgt["ppe"]])

            noise_pred = self.pipe.transformer(
                hidden_states=latents_in, timestep=t.expand(latents_in.shape[0]), 
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled_embeds, return_dict=False
            )[0]
            
            src_u, src_t, tar_u, tar_t = noise_pred.chunk(4)

            # --- CALCUL DU CONTROLLEUR (Sign Flip) ---
            
            # 1. Metric s(t) : Normalized Discrepancy
            v_src_vec = src_t.float()
            v_tgt_vec = tar_t.float()
            
            norm_diff = torch.norm(v_tgt_vec - v_src_vec, p=2, dim=(1,2,3)) # Norme par image
            norm_src = torch.norm(v_src_vec, p=2, dim=(1,2,3))
            
            # s(t)
            s_t = norm_diff / (norm_src + 1e-5)
            s_t_scalar = s_t.mean().item()
            
            # 2. Piecewise Controller
            # Note : Progress va de 0 à 1. t0 est le point de bascule.
            
            if progress < t0:
                # EARLY PHASE : Negative Association (Noise Suppression)
                # Formule : w_min + beta_e * (1 / (s_t + eps))
                # Plus le conflit est grand (s_t haut), plus le guidage est bas.
                raw_w = w_min + beta_e * (1.0 / (s_t_scalar + 0.1))
            else:
                # LATE PHASE : Positive Association (Boost)
                # Formule : w_min + beta_l * s_t
                # Plus le conflit est grand, plus on booste.
                raw_w = w_min + beta_l * s_t_scalar
            
            # Clamping
            w_t = min(max(raw_w, w_min), w_max)
            avg_w += w_t

            # Application
            # v_edit = v_src + w(t) * (v_tgt - v_src)
            # Attention : pour l'application, on utilise le v_src standard (guidance 3.5)
            # pour la partie structurelle, et notre w_t pour le saut vers la cible.
            
            v_src_safe = src_u + 3.5 * (src_t - src_u)
            v_tar_guided = tar_u + w_t * (tar_t - tar_u)
            
            # Update Euler
            zt_edit = zt_edit + dt * (v_tar_guided - v_src_safe)

        # Decode
        zt_dec = (zt_edit / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(zt_dec.to(torch.float32), return_dict=False)[0]
        image = torch.clamp(image, -1.0, 1.0)
        final_w = avg_w / (steps - start_index)
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0], final_w

# ==================== GRID SEARCH ====================
def run_grid_search():
    # PARAMETRES A OPTIMISER
    # -----------------------------------------------
    # t0 (Switch Point) : Quand passer de la sécurité à l'action ?
    # 0.3 = à 30% du process. 0.5 = à la moitié.
    SWITCH_POINTS = [0.2, 0.4] 
    
    # Beta Early (Suppression) : 
    # Formule : 3.5 + Beta_E / s(t). Si s(t)~1.0, on veut rester vers 3.5.
    # Si s(t) monte à 10 (bruit), on veut que ça baisse.
    # Plus Beta_E est petit, plus on est sévère sur le bruit ? 
    # Non, si Beta_E=0, w = w_min. 
    # Essayons des valeurs qui modulent autour de w_min.
    BETAS_EARLY = [1.0, 5.0] 
    
    # Beta Late (Boost) :
    # Formule : 3.5 + Beta_L * s(t). Si s(t)~1.0, on veut booster vers 15.0 ?
    # Donc Beta_L doit être assez fort (genre 5 à 10).
    BETAS_LATE = [5.0, 10.0, 15.0]
    # -----------------------------------------------

    if not os.path.exists(YAML_PATH): return
    with open(YAML_PATH, 'r') as f: full_data = list(yaml.safe_load_all(f))[0]
    subset = full_data[:5] # Subset rapide
    
    editor = SignFlipEditor()
    results = []

    print(f"[INFO] Grid Search: {len(SWITCH_POINTS)*len(BETAS_EARLY)*len(BETAS_LATE)} combos on {len(subset)} images.")

    for i, entry in enumerate(subset):
        img_path = os.path.join(IMAGES_ROOT, os.path.basename(entry['init_img']))
        if not os.path.exists(img_path): continue
        init_img = load_image(img_path).resize((1024,1024))
        src_p, tgt_p = entry['source_prompt'], entry['target_prompts'][0]

        print(f"\n--- Img {i}: {os.path.basename(img_path)} ---")
        
        # Grid Search Loop
        for t0, be, bl in itertools.product(SWITCH_POINTS, BETAS_EARLY, BETAS_LATE):
            config_id = f"T{t0}_BE{be}_BL{bl}"
            try:
                res, avg_w = editor.process(
                    init_img, src_p, tgt_p, 
                    t0=t0, beta_e=be, beta_l=bl, w_min=1.0
                )
                
                score = editor.metrics.compute(init_img, res, tgt_p)
                
                # Save visual
                res.save(os.path.join(OUTPUT_DIR, f"Img{i}_{config_id}.jpg"))
                
                print(f"   [{config_id}]: LPIPS={score['LPIPS']:.3f}, CLIP={score['CLIP']:.3f} (AvgW={avg_w:.1f})")
                
                results.append({
                    "Image": os.path.basename(img_path),
                    "t0": t0, "Beta_E": be, "Beta_L": bl,
                    "LPIPS": score['LPIPS'], "CLIP": score['CLIP'], "Avg_W": avg_w
                })
            except Exception as e:
                print(f"ERR {config_id}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        print("\n" + "="*60)
        print(" SIGN FLIP OPTIMIZATION RESULTS ")
        print("="*60)
        # On cherche le meilleur CLIP pour un LPIPS < 0.20
        summary = df.groupby(["t0", "Beta_E", "Beta_L"])[["LPIPS", "CLIP"]].mean()
        # Score hybride pour trier : CLIP - LPIPS (simple heuristic)
        summary["Score"] = summary["CLIP"] - summary["LPIPS"]
        print(summary.sort_values("Score", ascending=False))
        df.to_csv(os.path.join(OUTPUT_DIR, "sign_flip_results.csv"), index=False)

if __name__ == "__main__":
    run_grid_search()