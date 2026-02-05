# utils.py
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torchvision.transforms as T

def load_models(model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    print(f"Chargement du pipeline '{model_id}' en FP16...")
    torch_dtype = torch.float16
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None).to(device)
        print("Pipeline Stable Diffusion chargé avec succès.")
        return pipe
    except Exception as e:
        print(f"Erreur lors du chargement du pipeline : {e}")
        return None

def preprocess_image(image_path, size=(512, 512)):
    transform = T.Compose([
        T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)