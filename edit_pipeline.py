import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
import sys

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def load_model(model_type: str):
    """
    Charge le modèle approprié avec optimisation de la mémoire.
    """
    print(f"⏳ Chargement du modèle : {model_type}...")
    
    model_id = ""
    if model_type == "flux":
        # Flux.1-dev est excellent pour suivre les prompts
        model_id = "black-forest-labs/FLUX.1-dev" 
    elif model_type == "sdxl":
        # SDXL est souvent plus robuste que SD3 pour l'édition pure
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        # SD3 Medium
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, 
            torch_dtype=DTYPE,
            use_safetensors=True
        )
        
        # Optimisation mémoire cruciale pour les gros modèles
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload() 
        
        print("✅ Modèle chargé avec succès.")
        return pipe
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

def decompose_prompt(prompt):
    """
    Découpe le prompt de manière plus robuste.
    """
    parts = []
    # Découpage basique, pourrait être remplacé par un NLP (Spacy/NLTK) pour plus de précision
    delimiters = [',', ' and ', '.']
    
    current_parts = [prompt]
    for delimiter in delimiters:
        new_parts = []
        for p in current_parts:
            new_parts.extend(p.split(delimiter))
        current_parts = new_parts
        
    return [p.strip() for p in current_parts if p.strip()]

def resize_for_condition_image(image, resolution=1024):
    """Redimensionne l'image pour qu'elle soit compatible avec les modèles de diffusion (multiple de 8/16)."""
    w, h = image.size
    aspect_ratio = w / h
    if w > h:
        new_w = resolution
        new_h = int(resolution / aspect_ratio)
    else:
        new_h = resolution
        new_w = int(resolution * aspect_ratio)
    
    # Arrondir au multiple de 16 le plus proche pour éviter les erreurs de tenseurs
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    return image.resize((new_w, new_h), Image.LANCZOS)

def edit_image_global(pipe, image, prompt, strength=0.7, guidance_scale=7.5):
    """
    Stratégie 1 : Édition globale directe.
    """
    print(f"🎨 Stratégie Globale avec le prompt : '{prompt}'")
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength, # 0.0 = image originale, 1.0 = bruit total
        guidance_scale=guidance_scale,
        num_inference_steps=30
    ).images[0]
    return result

def edit_image_sequential(pipe, image, sub_prompts, strength=0.5, guidance_scale=7.5):
    """
    Stratégie 2 : Édition séquentielle (Multi-turn).
    Applique les changements les uns après les autres pour une meilleure cohérence.
    """
    print(f"🔄 Stratégie Séquentielle. Étapes : {len(sub_prompts)}")
    
    current_image = image
    
    # On réduit le strength à chaque étape pour ne pas détruire l'image précédente
    # ou on le garde constant mais faible.
    step_strength = strength 
    
    for i, sub_p in enumerate(sub_prompts):
        print(f"   👉 Étape {i+1}/{len(sub_prompts)} : '{sub_p}'")
        
        # Pour Flux, le guidance_scale est souvent ignoré ou bas (3.5), pour SDXL c'est plus haut (7.5)
        current_image = pipe(
            prompt=sub_p,
            image=current_image,
            strength=step_strength,
            guidance_scale=guidance_scale,
            num_inference_steps=25
        ).images[0]
        
    return current_image

def edit_process(pipe, image, tar_prompt, strategy="global"):
    """
    Fonction principale de dispatch.
    """
    # Prétraitement de l'image
    image = resize_for_condition_image(image)
    
    # Paramètres par défaut (à ajuster selon le modèle)
    # Flux préfère un guidance faible (3.5), SDXL/SD3 préfèrent (7.0 - 9.0)
    is_flux = "flux" in pipe.config._name_or_path.lower()
    cfg = 3.5 if is_flux else 7.5
    
    if strategy == "sequential":
        sub_prompts = decompose_prompt(tar_prompt)
        # Si la décomposition échoue ou ne donne qu'un élément, fallback sur global
        if len(sub_prompts) > 1:
            # On utilise un strength plus faible en séquentiel pour préserver la structure
            return edit_image_sequential(pipe, image, sub_prompts, strength=0.45, guidance_scale=cfg)
        else:
            print("⚠️ Un seul sous-prompt détecté, passage en mode global.")
            
    # Défaut : Global
    return edit_image_global(pipe, image, tar_prompt, strength=0.75, guidance_scale=cfg)

if __name__ == "__main__":
    # Gestion des arguments
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "sdxl" # Par défaut SDXL (plus stable pour l'edit que SD3)
    strategy_choice = sys.argv[2] if len(sys.argv) > 2 else "sequential"

    # Chargement
    pipe = load_model(model_choice)

    # Chargement de l'image (Création d'une image dummy si pas de fichier local pour tester)
    image_path = "test_image.jpg"
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print("⚠️ Image locale non trouvée, téléchargement d'une image de test...")
        from diffusers.utils import load_image as dl_load
        image = dl_load("https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg")

    # Prompts
    # Note: Il est important que le prompt cible décrive l'image entière, pas juste le changement
    tar_prompt = "A realistic photo of a snowy mountain at sunset, with a cabin in the foreground, cinematic lighting"

    print(f"🎯 Cible : {tar_prompt}")
    print(f"⚙️ Stratégie : {strategy_choice}")

    # Exécution
    final_image = edit_process(pipe, image, tar_prompt, strategy=strategy_choice)

    # Sauvegarde
    output_name = f"edited_{model_choice}_{strategy_choice}.jpg"
    final_image.save(output_name)
    print(f"✨ Image sauvegardée sous : {output_name}")