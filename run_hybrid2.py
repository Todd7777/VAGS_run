import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision.transforms as T

from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
import open_clip
import lpips

# LPIPS setup
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

# CLIP setup
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model = clip_model.to(DEVICE)

# ==================== CONFIGURATION ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda")
DTYPE = torch.float16
OUTDIR = "outputs/benchmark_matrix_v25"
os.makedirs(OUTDIR, exist_ok=True)

print(f"[INFO] Device: {DEVICE}")

# ==================== MATH KERNELS ====================
def lerp(v0, v1, t):
    """Interpolation Linéaire (Classique)"""
    return (1.0 - t) * v0 + t * v1

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """Interpolation Sphérique (Géodésique)"""
    v0_f = v0.to(dtype=torch.float32)
    v1_f = v1.to(dtype=torch.float32)
    v0_norm = v0_f / torch.norm(v0_f, dim=-1, keepdim=True)
    v1_norm = v1_f / torch.norm(v1_f, dim=-1, keepdim=True)
    dot = torch.sum(v0_norm * v1_norm, dim=-1)
    
    if torch.abs(dot).max() > DOT_THRESHOLD:
        res = (1.0 - t) * v0_f + t * v1_f
    else:
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = torch.sin(theta_t) / sin_theta_0
        s0, s1 = s0.unsqueeze(-1), s1.unsqueeze(-1)
        res = s0 * v0_f + s1 * v1_f
    return res.to(dtype=DTYPE)

def get_cfg_value(base_cfg, progress, mode="constant"):
    """Gestion du CFG Dynamique"""
    if mode == "constant":
        return base_cfg
    elif mode == "increasing":
        # 0.5x -> 1.5x (Commence doux, finit fort)
        return base_cfg * (1.0 + progress)
    elif mode == "decreasing":
        # 1.5x -> 0.5x (Commence fort, finit doux)
        return base_cfg * (1.5 - progress)
    elif mode == "bell":
        # Fort au milieu
        fac = np.sin(progress * np.pi)
        return base_cfg * (0.8 + 0.9 * fac)
    return base_cfg

# ==================== UTILS ====================
def load_img(path):
    if not os.path.exists(path): return Image.new("RGB", (1024,1024), "gray")
    return load_image(path).resize((1024,1024))

@torch.no_grad()
def compute_metrics(img_ref, img_out):
    # Convert PIL → torch
    to_tensor = T.ToTensor()
    ref_t = to_tensor(img_ref).unsqueeze(0).to(DEVICE)
    out_t = to_tensor(img_out).unsqueeze(0).to(DEVICE)

    # Resize LPIPS to 256
    ref_lp = T.Resize((256,256))(ref_t)
    out_lp = T.Resize((256,256))(out_t)
    
    lp = lpips_fn(ref_lp, out_lp).item()

    # CLIP similarity
    ref_c = clip_preprocess(img_ref).unsqueeze(0).to(DEVICE)
    out_c = clip_preprocess(img_out).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        e1 = clip_model.encode_image(ref_c)
        e2 = clip_model.encode_image(out_c)
    clip_sim = torch.nn.functional.cosine_similarity(e1, e2).item()

    return {"lpips": lp, "clip": clip_sim}

def save_matrix(results, filename):
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    
    w, h = results[0]['img'].size
    canvas = Image.new("RGB", (w * cols, h * rows + 50 * rows), "white")
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("arial.ttf", 40)
    except: font = ImageFont.load_default()

    for i, res in enumerate(results):
        r = i // cols
        c = i % cols
        x = c * w
        y = r * (h + 50)
        
        canvas.paste(res['img'], (x, y + 50))
        # Titre centré
        text_w = draw.textlength(res['label'], font=font)
        draw.text((x + (w - text_w)/2, y + 5), res['label'], fill="black", font=font)
    
    path = os.path.join(OUTDIR, filename)
    canvas.save(path)
    print(f"[SAVE] {path}")

# ==================== ENGINE ====================
class FlowEditor:
    def __init__(self):
        print("[LOAD] SD3 Pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=DTYPE
        ).to(DEVICE)
        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
        self.pipe.set_progress_bar_config(disable=True)

    def get_embeddings(self, prompt):
        neg = "bad quality, blurry, distortion, lowres, ugly"
        pe, ne, ppe, npe = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt=neg, negative_prompt_2=neg, negative_prompt_3=neg,
            device=DEVICE
        )
        return {"pe": pe, "ne": ne, "ppe": ppe, "npe": npe}

    def calc_v(self, latents, prompt_embeds, pooled_embeds, t, guidance_scale):
        ts = t.expand(latents.shape[0])
        noise_pred = self.pipe.transformer(
            hidden_states=latents, timestep=ts,
            encoder_hidden_states=prompt_embeds, pooled_projections=pooled_embeds,
            return_dict=False
        )[0]
        unc, txt = noise_pred.chunk(2)
        return unc + guidance_scale * (txt - unc)

    @torch.no_grad()
    def run_edit(self, init_image, source_prompt, target_chain, 
                 mode="baseline",
                 interp_type="linear",
                 cfg_type="constant",
                 base_cfg=7.5,
                 steps=50):
        
        # 1. Encodage Image
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x_src = self.pipe.vae.encode(img_t).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        x_src = x_src.to(dtype=DTYPE)

        # 2. Embeddings
        emb_src = self.get_embeddings(source_prompt)
        
        if mode == "baseline":
            # Baseline = Saut direct vers la fin de la chaine
            emb_targets = [self.get_embeddings(target_chain[-1])] # Liste de 1 élément
        else:
            # Chain = On utilise toute la chaine
            emb_targets = [self.get_embeddings(p) for p in target_chain]

        # 3. Scheduler
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = self.pipe.scheduler.timesteps
        
        # Init Trajectoire
        zt_edit = x_src.clone()
        
        # 4. Boucle ODE
        for i, t in enumerate(tqdm(timesteps, desc=f"{mode}|{interp_type}|{cfg_type}")):
            t_curr = t / 1000.0
            t_next = timesteps[i+1] / 1000.0 if i + 1 < len(timesteps) else 0.0
            dt = t_next - t_curr
            progress = i / len(timesteps)

            # A. Latents Couplés (Source Reconstruction)
            noise = torch.randn_like(x_src) # Bruit fixe ou aléatoire ? FlowEdit use random per step usually
            # Pour la cohérence, on peut utiliser un générateur fixe si on veut
            zt_src = (1 - t_curr) * x_src + t_curr * noise
            
            # zt_tar est couplé à zt_edit (L'innovation FlowEdit)
            zt_tar = zt_edit + zt_src - x_src

            # B. Interpolation des Prompts
            if mode == "baseline":
                # Pas d'interpolation, on vise la cible directement
                pe_tar = emb_targets[0]["pe"]
                ppe_tar = emb_targets[0]["ppe"]
            else:
                # Interpolation continue le long de la chaîne
                n_stages = len(emb_targets)
                float_idx = progress * (n_stages - 1)
                idx_s = int(float_idx)
                idx_e = min(idx_s + 1, n_stages - 1)
                alpha = float_idx - idx_s
                
                start = emb_targets[idx_s]
                end = emb_targets[idx_e]
                
                if interp_type == "slerp":
                    pe_tar = slerp(start["pe"], end["pe"], alpha)
                    ppe_tar = slerp(start["ppe"], end["ppe"], alpha)
                else: # Linear
                    pe_tar = lerp(start["pe"], end["pe"], alpha)
                    ppe_tar = lerp(start["ppe"], end["ppe"], alpha)

            # C. Calcul CFG Dynamique
            current_cfg = get_cfg_value(base_cfg, progress, cfg_type)

            # D. Vitesses
            # V_src (Guidance fixe souvent plus faible pour la reconstruction)
            in_src = torch.cat([zt_src] * 2)
            emb_s = torch.cat([emb_src["ne"], emb_src["pe"]])
            pool_s = torch.cat([emb_src["npe"], emb_src["ppe"]])
            v_src = self.calc_v(in_src, emb_s, pool_s, t, 1.5) 
            
            # V_tar
            in_tar = torch.cat([zt_tar] * 2)
            # On utilise le négatif source pour la cible
            emb_t = torch.cat([emb_src["ne"], pe_tar])
            pool_t = torch.cat([emb_src["npe"], ppe_tar])
            v_tar = self.calc_v(in_tar, emb_t, pool_t, t, current_cfg)
            
            # E. Update (Delta)
            delta = v_tar - v_src
            zt_edit = zt_edit + dt * delta

        # 5. Decode
        zt_edit = (zt_edit / self.pipe.vae.config.scaling_factor).to(dtype=torch.float32)
        image = self.pipe.vae.decode(zt_edit, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

# ==================== MAIN EXPERIMENT ====================
def main():
    editor = FlowEditor()

    # --- SUJET : VOITURE ---
    print("\n>>> LANCEMENT MATRICE : VOITURE")
    img = load_img("example_images/red_car.png")
    src_p = "A photo of a red sedan car on a road"
    chain = [src_p, "A photo of a purple sedan car on a road", "A photo of a metallic blue sedan car on a road"]
    
    results = []
    
    # 1. Source
    results.append({'img': img, 'label': "SOURCE"})
    
    # 2. Baseline (FlowEdit Original - Direct Jump)
    res_base = editor.run_edit(img, src_p, chain, mode="baseline", base_cfg=7.5)
    results.append({'img': res_base, 'label': "BASELINE (Direct)"})
    
    # 3. ContinueEdit (Linear + Constant) - Approche naïve
    res_lin = editor.run_edit(img, src_p, chain, mode="chain", interp_type="linear", cfg_type="constant")
    results.append({'img': res_lin, 'label': "Lin + ConstCFG"})
    
    # 4. ContinueEdit (Slerp + Constant) - Mieux géométriquement
    res_slerp = editor.run_edit(img, src_p, chain, mode="chain", interp_type="slerp", cfg_type="constant")
    results.append({'img': res_slerp, 'label': "Slerp + ConstCFG"})
    
    # 5. ContinueEdit (Slerp + Increasing CFG) - Force la fin
    res_inc = editor.run_edit(img, src_p, chain, mode="chain", interp_type="slerp", cfg_type="increasing")
    results.append({'img': res_inc, 'label': "Slerp + IncCFG"})
    
    # 6. ContinueEdit (Slerp + Decreasing CFG) - Force le début
    res_dec = editor.run_edit(img, src_p, chain, mode="chain", interp_type="slerp", cfg_type="decreasing")
    results.append({'img': res_dec, 'label': "Slerp + DecCFG"})

    save_matrix(results, "MATRIX_CAR.png")

    # --- SUJET : BIG BEN ---
    print("\n>>> LANCEMENT MATRICE : BIG BEN")
    img2 = load_img("example_images/lighthouse.png")
    src_p2 = "A tall white lighthouse on rocky shore"
    chain2 = [src_p2, "A tall square white tower on rocky shore", "A tall stone clock tower, Big Ben, London"]
    
    results2 = []
    results2.append({'img': img2, 'label': "SOURCE"})
    
    # Baseline
    res_base2 = editor.run_edit(img2, src_p2, chain2, mode="baseline", base_cfg=7.5)
    results2.append({'img': res_base2, 'label': "BASELINE"})
    
    # Slerp Constant
    res_sl_co = editor.run_edit(img2, src_p2, chain2, mode="chain", interp_type="slerp", cfg_type="constant", base_cfg=7.5)
    results2.append({'img': res_sl_co, 'label': "Slerp + Const"})
    
    # Slerp Decreasing (Pour casser la structure du phare tôt)
    res_sl_dec = editor.run_edit(img2, src_p2, chain2, mode="chain", interp_type="slerp", cfg_type="decreasing", base_cfg=7.5)
    results2.append({'img': res_sl_dec, 'label': "Slerp + DecCFG"})
    
    save_matrix(results2, "MATRIX_BIGBEN.png")

    print("\n[SUCCESS] Matrices générées.")

if __name__ == "__main__":
    main()