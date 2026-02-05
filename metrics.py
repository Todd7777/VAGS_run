import torch
import torch.nn.functional as F
from PIL import Image
import clip
import lpips


def load_image(path, device):
    """Charge une image depuis un chemin et la convertit en tensor normalisé pour le modèle CLIP."""
    image = Image.open(path).convert("RGB")
    preprocess = clip.load("ViT-B/32")[1]
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor


def compute_clip_similarity(image, text, model, preprocess):
    """Calcule la similarité cosinus entre l’image et le texte en utilisant CLIP."""
    device = next(model.parameters()).device

    if isinstance(image, str):
        image = load_image(image, device)
    else:
        image = image.to(device)

    text_tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

    return similarity


def compute_lpips_distance(img1, img2, lpips_model):
    """Calcule la distance perceptuelle LPIPS entre deux images."""
    device = next(lpips_model.parameters()).device

    if isinstance(img1, str):
        img1 = load_image(img1, device)
    if isinstance(img2, str):
        img2 = load_image(img2, device)

    # LPIPS attend des images dans [-1,1], 3xHxW
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    with torch.no_grad():
        dist = lpips_model(img1, img2).item()

    return dist


def compute_continuity(images_list):
    """Calcule la différence moyenne absolue normalisée entre images successives."""
    diffs = []
    for i in range(len(images_list) - 1):
        im1 = images_list[i].float()
        im2 = images_list[i + 1].float()
        diff = torch.abs(im2 - im1).mean().item()
        diffs.append(diff)

    if len(diffs) == 0:
        return 0.0

    mean_diff = sum(diffs) / len(diffs)
    # Normalisation entre 0 et 1 (suppose images dans [0,1])
    return min(max(mean_diff, 0.0), 1.0)


if __name__ == "__main__":
    import os

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger modèle CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Charger modèle LPIPS
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # Chemins d’images d’exemple (à adapter)
    img_path1 = os.path.join(os.path.dirname(__file__), "example1.jpg")
    img_path2 = os.path.join(os.path.dirname(__file__), "example2.jpg")

    # Charger images
    img1 = load_image(img_path1, device)
    img2 = load_image(img_path2, device)

    # Tester CLIP similarity
    text = "a photo of a cat"
    sim = compute_clip_similarity(img1, text, clip_model, clip_preprocess)
    print(f"CLIP similarity between image and text '{text}': {sim:.4f}")

    # Tester LPIPS distance
    dist = compute_lpips_distance(img1, img2, lpips_model)
    print(f"LPIPS distance between two images: {dist:.4f}")

    # Tester continuité sur une liste d’images
    continuity = compute_continuity([img1, img2])
    print(f"Continuity metric between images: {continuity:.4f}")
