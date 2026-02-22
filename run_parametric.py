import os
import torch
import numpy as np
import csv
import pandas as pd
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import lpips
from torchvision import transforms
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve
from PIL import Image

# ==================== CONFIGURATION ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

INPUT_ROOT = "Data/Images"
OUTPUT_DIR = "outputs/PARAMETRIC_STUDY"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LES CONFIGURATIONS À TESTER ====================
CONFIGS = [
    {"name": "Baseline",      "kappa": 0.0, "m": 1.0}, # Les valeurs k/m n'importent pas pour la baseline
    {"name": "Exp_k3.0_m3.0", "kappa": 3.0, "m": 3.0}, # Ta configuration actuelle (Référence)
    {"name": "Exp_k4.0_m3.0", "kappa": 4.0, "m": 3.0}, # Plus de force (boost/dampen plus agressif)
    {"name": "Exp_k5.0_m3.0", "kappa": 5.0, "m": 3.0}, # Force extrême
    {"name": "Exp_k3.0_m1.5", "kappa": 3.0, "m": 1.5}, # Haute sensibilité au conflit (réagit plus vite)
    {"name": "Exp_k3.0_m5.0", "kappa": 3.0, "m": 5.0}, # Basse sensibilité (nécessite un gros conflit pour réagir)
]

# ==================== DATASET (15 images variées) ====================
DATASET = [
    {
        "filename": "bear.png",
        "source": "A large brown bear walking through a stream of water. The stream appears to be shallow, as the bear is able to walk through it without any difficulty.",
        "targets": [
            ("A large black bear walking through a stream of water. The stream appears to be shallow, as the black bear is able to walk through it without any difficulty.", "black_bear"),
            ("A large polar bear walking through a stream of water. The stream appears to be shallow, as the polar bear is able to walk through it without any difficulty.", "polar_bear")
        ]
    },
    {
        "filename": "bikes.png",
        "source": "A bicycle parked on the sidewalk in front of a red brick building. The bicycle is positioned close to the door of the building, making it easily accessible for the owner. The building has a red brick exterior, giving it a distinctive appearance.",
        "targets": [
            ("A motorcycle parked on the sidewalk in front of a red brick building.", "motorcycle"),
            ("A vespa scooter parked on the sidewalk in front of a red brick building.", "vespa")
        ]
    },
    {
        "filename": "boat_silhouette.png",
        "source": "A serene scene of a lake with a silhouette of a sailboat floating on the water. The silhouette of the boat is positioned in the middle of the lake, slightly to the left. The sun is setting in the background, casting a warm glow over the scene.",
        "targets": [
            ("A serene scene of a lake with a sailboat floating on the water. The sailboat has white sails and red hull.", "white_sails")
        ]
    },
    {
        "filename": "bus.png",
        "source": "A yellow van parked in front of a house, outside a garage. The van is of a classic model. The house has a brown exterior, and there is a tree nearby. There is a spare tire hanged between the headlights of the van.",
        "targets": [
            ("A SUV parked in front of a house, outside a garage. The SUV is of a modern model, possibly electric.", "SUV"),
            ("A pink van parked in front of a house, outside a garage.", "pink_van")
        ]
    },
    {
        "filename": "cake.png",
        "source": "A three-layer cake with white frosting, placed on a wooden table. The cake is adorned with a variety of fruits. The cake is presented on a white plate.",
        "targets": [
            ("A three-layer cake with chocolate frosting, placed on a wooden table. The cake is adorned with a variety of berries.", "chocolate_berries"),
            ("A three-layer wedding cake with white frosting, placed on a wooden table.", "wedding_cake")
        ]
    },
    {
        "filename": "cat_and_dog.png",
        "source": "A dog and a cat sitting together on a sidewalk.",
        "targets": [
            ("A dog and a cat made out of lego sitting together on a sidewalk.", "lego")
        ]
    },
    {
        "filename": "clown_fish.png",
        "source": "A vibrant underwater scene with a clownfish swimming among various coral reefs.",
        "targets": [
            ("A vibrant underwater scene with a small sea turtle swimming among various coral reefs.", "sea_turtle"),
            ("A vibrant underwater scene with a shark swimming among various coral reefs.", "shark")
        ]
    },
    {
        "filename": "corgi.png",
        "source": "A brown and white dog sitting on a dirt ground near a body of water.",
        "targets": [
            ("A wooden sculpture of a brown and white dog made sitting on a dirt ground.", "wooden_sculpture"),
            ("A red fox sitting on a dirt ground near a body of water.", "red_fox")
        ]
    },
    {
        "filename": "dog.png",
        "source": "A small black and brown dog sitting on a lush green field.",
        "targets": [
            ("A small black and brown dog made out of lego bricks sitting on a lush green field.", "lego_bricks"),
            ("A small black and brown poodle sitting on a lush green field.", "poodle")
        ]
    },
    {
        "filename": "flowers.png",
        "source": "A vase filled with a beautiful bouquet of pink, red and white flowers.",
        "targets": [
            ("A vase filled with a beautiful bouquet of orange, yellow and white flowers.", "orange_yellow_white")
        ]
    },
    {
        "filename": "gas_station.png",
        "source": "A gas station with a white and red sign that reads 'CAFE'.",
        "targets": [
            ("A gas station with a white and red sign that reads 'CVPR'.", "cvpr"),
            ("A gas station with a white and red sign that reads 'FOOD'.", "food")
        ]
    },
    {
        "filename": "horse.png",
        "source": "A white horse running through a grassy field.",
        "targets": [
            ("A white unicorn running through a grassy field.", "unicorn"),
            ("A sculpture bronze horse running through a grassy field.", "bronze_sculpture")
        ]
    },
    {
        "filename": "pizza.png",
        "source": "A large, cheesy pizza sitting on a wooden pizza board.",
        "targets": [
            ("A large, cheesy pizza, topped with pineapple and ham.", "pineapple_ham")
        ]
    },
    {
        "filename": "sign.png",
        "source": "A large white billboard with a bold message written in black letters. The message reads, 'LOVE IS ALL YOU NEED.'",
        "targets": [
            ("A large white billboard with a bold message written in black letters. The message reads, 'CVPR IS ALL YOU NEED.'", "cvpr")
        ]
    },
    {
        "filename": "tiger.png",
        "source": "A large tiger standing in a swamp.",
        "targets": [
            ("A large crochet tiger standing in a swamp.", "crochet_tiger"),
            ("A large wolf standing in a swamp.", "wolf")
        ]
    }
]

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

    def get_structural_metrics(self, img_source, img_generated):
        src = np.array(img_source.resize((512, 512))).astype(np.float32) / 255.0
        gen = np.array(img_generated.resize((512, 512))).astype(np.float32) / 255.0

        mse = float(np.mean((src - gen) ** 2)) * 1e4
        psnr = float(peak_signal_noise_ratio(src, gen, data_range=1.0))
        ssim = float(structural_similarity(src, gen, data_range=1.0, channel_axis=2, win_size=7)) * 100.0

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        lum_w = np.array([0.299, 0.587, 0.114])
        lum_src = np.dot(src, lum_w)
        lum_gen = np.dot(gen, lum_w)
        gx_s = convolve(lum_src, kx);  gy_s = convolve(lum_src, kx.T)
        gx_g = convolve(lum_gen, kx);  gy_g = convolve(lum_gen, kx.T)
        grad_src = np.sqrt(gx_s**2 + gy_s**2 + 1e-8)
        grad_gen = np.sqrt(gx_g**2 + gy_g**2 + 1e-8)
        grad_diff = np.abs(grad_src - grad_gen) / (grad_src + grad_gen + 1e-6)
        struct_dist = float(np.mean(grad_diff)) * 1e3

        return mse, psnr, ssim, struct_dist

# ==================== MOTEUR DE COMPARAISON ====================
class ComparisonEditor:
    def __init__(self):
        print(f"[INIT] Loading SD3 Pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=DTYPE
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)

    @torch.no_grad()
    def process(self, init_image, src_prompt, tgt_prompt, strategy_name, kappa, m):
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x0_src = (self.pipe.vae.encode(img_t).latent_dist.mode() - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        x0_src = x0_src.to(dtype=DTYPE)

        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(src_prompt, src_prompt, src_prompt, device=DEVICE)
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(tgt_prompt, tgt_prompt, tgt_prompt, device=DEVICE)

        t_start  = 0.85
        base_cfg = 7.5

        steps = 50
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps   = self.pipe.scheduler.timesteps
        start_index = int((1.0 - t_start) * steps)

        zt = x0_src.clone()

        for i, t_tensor in enumerate(timesteps):
            if i < start_index: continue

            t  = t_tensor.item() / 1000.0
            dt = (timesteps[i+1].item() / 1000.0 if i+1 < len(timesteps) else 0.0) - t
            noise  = torch.randn_like(x0_src)
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
            current_lambda = base_cfg  # Baseline

            if strategy_name != "Baseline":
                diff = vc_t - vc_s
                norm_diff = torch.norm(diff, p=2, dim=(1,2,3)).mean().item()
                norm_src = torch.norm(vc_s, p=2, dim=(1,2,3)).mean().item()
                relative_conflict = norm_diff / (norm_src + 1e-5)

                # Formule Exponential CAG
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

# ==================== STATISTIQUES ====================
def analyze_and_print_results(csv_path):
    print("\n" + "="*80)
    print(" 📊 STATISTICAL ANALYSIS: EXPONENTIAL CAG PARAMETRIC STUDY ")
    print("="*80)
    
    try:
        df = pd.read_csv(csv_path)
        metrics = ['CLIP', 'LPIPS', 'MSE_x1e4', 'PSNR_dB', 'SSIM_x100', 'StructDist_x1e3']
        
        # 1. Calcul des moyennes par configuration
        summary = df.groupby('Config_Name')[metrics].mean()
        
        print("\n--- MOYENNES GLOBALES (MEAN METRICS) ---")
        print(summary.round(4).to_string())
        
        # 2. Calcul des différences par rapport à la Baseline
        if 'Baseline' in summary.index:
            print("\n--- DIFFERENCE VS BASELINE (Δ) ---")
            print("Note: Négatif = Mieux pour LPIPS, MSE, StructDist | Positif = Mieux pour CLIP, PSNR, SSIM")
            baseline_vals = summary.loc['Baseline']
            delta = summary - baseline_vals
            
            # Formater avec le signe + ou - (correction du warning map)
            formatted_delta = delta.map(lambda x: f"{x:+.4f}" if pd.notnull(x) else "NaN")
            formatted_delta = formatted_delta.drop('Baseline') 
            print(formatted_delta.to_string())
            
            # Sauvegarde du résumé
            summary_path = csv_path.replace(".csv", "_summary.csv")
            summary.to_csv(summary_path)
            print(f"\n[SAVED] Résumé statistique sauvegardé dans : {summary_path}")
            
    except Exception as e:
        print(f"[ERREUR] Impossible d'analyser les résultats : {e}")

# ==================== MAIN ====================
if __name__ == "__main__":
    editor    = ComparisonEditor()
    evaluator = MetricEvaluator(DEVICE)

    MAX_IMAGES = 15

    print(f"[START] Parametric Study (Hyperparameter tuning on Exponential CAG)...")

    csv_path = os.path.join(OUTPUT_DIR, "parametric_study_results.csv")
    fieldnames = ['Image', 'Target', 'Config_Name', 'Kappa', 'M_Threshold', 'CLIP', 'LPIPS', 'MSE_x1e4', 'PSNR_dB', 'SSIM_x100', 'StructDist_x1e3', 'Filename']

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(DATASET):
            if idx >= MAX_IMAGES: 
                break

            filename_clean = item["filename"].replace("flowedit_data/", "")
            full_path = os.path.join(INPUT_ROOT, filename_clean)

            if not os.path.exists(full_path):
                full_path_alt = os.path.join(INPUT_ROOT, "flowedit_data", filename_clean)
                if os.path.exists(full_path_alt):
                    full_path = full_path_alt
                else:
                    print(f"[SKIP] Image introuvable : {item['filename']}")
                    continue

            print(f"\nProcessing [{idx+1}/{MAX_IMAGES}]: {filename_clean}...")
            init_img  = load_image(full_path).resize((1024, 1024))
            base_name = os.path.splitext(filename_clean)[0]

            for tgt_prompt, code in item["targets"]:
                print(f"   -> Target: {code}")
                for config in CONFIGS:
                    strat_name = config["name"]
                    kappa_val = config["kappa"]
                    m_val = config["m"]
                    
                    res_img = editor.process(init_img, item["source"], tgt_prompt, strat_name, kappa_val, m_val)

                    clip_s = evaluator.get_clip_score(res_img, tgt_prompt)
                    lpips_d = evaluator.get_lpips_distance(init_img, res_img)
                    mse, psnr, ssim, struct_dist = evaluator.get_structural_metrics(init_img, res_img)

                    print(f"      [{strat_name:15s}]  CLIP: {clip_s:.4f} | LPIPS: {lpips_d:.4f} | MSE: {mse:.2f}")

                    fname = f"{base_name}_{code}_{strat_name}.jpg"
                    res_img.save(os.path.join(OUTPUT_DIR, fname))

                    writer.writerow({
                        'Image':           base_name,
                        'Target':          code,
                        'Config_Name':     strat_name,
                        'Kappa':           kappa_val if kappa_val is not None else "N/A",
                        'M_Threshold':     m_val if m_val is not None else "N/A",
                        'CLIP':            f"{clip_s:.4f}",
                        'LPIPS':           f"{lpips_d:.4f}",
                        'MSE_x1e4':        f"{mse:.4f}",
                        'PSNR_dB':         f"{psnr:.4f}",
                        'SSIM_x100':       f"{ssim:.4f}",
                        'StructDist_x1e3': f"{struct_dist:.4f}",
                        'Filename':        fname,
                    })
                    csv_file.flush()
                    torch.cuda.empty_cache()

    print(f"\n[DONE] Étude paramétrique terminée.")
    
    # Appel de la fonction d'analyse à la toute fin
    analyze_and_print_results(csv_path)