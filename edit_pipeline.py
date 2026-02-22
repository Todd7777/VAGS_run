import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FluxPipeline

def load_model(model_type: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "flux":
        pipe = FluxPipeline.from_pretrained("facebook/flux-model")
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3")
    pipe = pipe.to(device)
    return pipe

def decompose_prompt(prompt):
    # Split the prompt by commas or 'and' to get sub-prompts
    parts = []
    for part in prompt.split(','):
        subparts = part.split(' and ')
        parts.extend([sp.strip() for sp in subparts if sp.strip()])
    return parts

def edit_image(pipe, image, src_prompt, tar_prompt, strategy, lambda_weight=0.5):
    sub_prompts = decompose_prompt(tar_prompt)
    print(f"Decomposed target prompt into sub-prompts: {sub_prompts}")
    # Generate images for each sub-prompt
    images = []
    for p in sub_prompts:
        img = pipe(prompt=p, image=image).images[0]
        images.append(img)
    # Also generate image for full tar_prompt (e(P))
    full_img = pipe(prompt=tar_prompt, image=image).images[0]

    # Combine images by weighted average of their pixel values
    # Convert images to tensors
    full_tensor = torch.tensor(full_img).float()
    sub_tensors = [torch.tensor(img).float() for img in images]
    # Average sub-prompts tensors
    sub_avg = torch.stack(sub_tensors).mean(dim=0)
    # Combine with lambda_weight
    combined_tensor = (1 - lambda_weight) * full_tensor + lambda_weight * sub_avg
    combined_tensor = combined_tensor.clamp(0, 255).byte()
    combined_img = Image.fromarray(combined_tensor.numpy())
    print(f"Combined full prompt image and sub-prompts images with lambda={lambda_weight}")
    return combined_img

if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "stable"
    pipe = load_model(model_type)

    # Load a local image for testing
    image_path = "test_image.jpg"
    image = Image.open(image_path).convert("RGB")

    src_prompt = "A photo of a cat sitting on a sofa"
    tar_prompt = "A photo of a dog sitting on a sofa, playing with a ball and smiling"

    strategy = "discrete"  # Can be "discrete", "smooth", or "multi_turn"
    edited_img = edit_image(pipe, image, src_prompt, tar_prompt, strategy)

    edited_img.save("edited_image.jpg")
    print("Edited image saved as edited_image.jpg")
