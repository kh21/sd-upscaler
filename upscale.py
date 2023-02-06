import hashlib
import os

import requests
import torch
from absl import app, flags
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF

import model_upscaler
import model_vae

flags.DEFINE_string("prompt", "Jane Random", "Prompt to be given to the SD model.")
flags.DEFINE_integer("seed", 1123, "Random seed.", lower_bound=0)
flags.DEFINE_string("output_dir", "output", "Output directory.")

FLAGS = flags.FLAGS

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
UPSCALER_CONFIG_URL = "https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json"
UPSCALER_MODEL_URL = "https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"

VAE_MODEL_NAME = "stabilityai/sd-vae-ft-mse-original"
VAE_MODEL_CKPT = "vae-ft-mse-840000-ema-pruned.ckpt"
VAE_MODEL_CONFIG_PATH = "latent-diffusion/models/first_stage_models/kl-f8/config.yaml"

# Model configuration values
SD_C = 4  # Latent dimension
SD_F = 8  # Latent patch size (pixels per latent)
SD_Q = 0.18215  # sd_model.scale_factor; scaling for latents in first stage models

NOISE_AUG_LEVEL = 0.0
GUIDANCE_SCALE = 7.5
BATCH_SIZE = 1


def fetch(url_or_path):
    _, ext = os.path.splitext(os.path.basename(url_or_path))
    cachekey = hashlib.md5(url_or_path.encode("utf-8")).hexdigest()
    cachename = f"{cachekey}{ext}"
    if not os.path.exists(f"cache/{cachename}"):
        os.makedirs("tmp", exist_ok=True)
        os.makedirs("cache", exist_ok=True)
        response = requests.get(url_or_path)
        with open(f"cache/{cachename}", "wb") as f:
            f.write(response.content)
    return f"cache/{cachename}"


@torch.no_grad()
def embed_prompts(prompts, tokenizer, embedder):
    return embedder(tokenizer(prompts))


def main(argv):
    """Main function"""

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print("")
    print("Step 1: Run SD model to generate an image...")
    prompt = FLAGS.prompt
    seed = FLAGS.seed
    seed_everything(seed)

    if device == "cpu":
        sd_pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID).to(device)
    else:
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID, torch_dtype=torch.float16
        ).to(device)

    image = sd_pipeline(prompt, guidance_scale=GUIDANCE_SCALE).images[0]

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    target_path = os.path.join(FLAGS.output_dir, "512x512.jpg")
    image.save(target_path)
    print(f"Generated image was saved in {target_path}")

    print("")
    print("Step 2.1: Encode the generated image to the latent...")
    vae_model_path = model_vae.download_from_huggingface(VAE_MODEL_NAME, VAE_MODEL_CKPT)
    vae_model = model_vae.load_model_from_config(
        VAE_MODEL_CONFIG_PATH,
        vae_model_path,
    )
    vae_model = vae_model.to(device)

    image = TF.to_tensor(image).to(device) * 2 - 1
    low_res_latent = vae_model.encode(image.unsqueeze(0)).sample() * SD_Q
    latent_noised = low_res_latent + NOISE_AUG_LEVEL * torch.randn_like(low_res_latent)

    print("")
    print("Step 2.2: Upscale the latent...")
    upscaler_model = model_upscaler.make_upscaler_model(
        fetch(UPSCALER_CONFIG_URL), fetch(UPSCALER_MODEL_URL), device=device
    )

    clip_tokenizer = model_upscaler.CLIPTokenizerTransform()
    clip_embedder = model_upscaler.CLIPEmbedder(device=device)
    empty_embedding = embed_prompts([""], clip_tokenizer, clip_embedder)
    prompt_embedding = embed_prompts([prompt], clip_tokenizer, clip_embedder)

    wrapped_model = model_upscaler.CFGUpscaler(
        upscaler_model, empty_embedding, cond_scale=GUIDANCE_SCALE
    )
    low_res_sigma = torch.full([BATCH_SIZE], NOISE_AUG_LEVEL, device=device)
    extra_args = {
        "low_res": latent_noised,
        "low_res_sigma": low_res_sigma,
        "c": prompt_embedding,
    }
    [_, c, h, w] = low_res_latent.shape
    x_shape = [BATCH_SIZE, c, 2 * h, 2 * w]
    noise = torch.randn(x_shape, device=device)

    upscaled_latents = model_upscaler.do_sample(
        wrapped_model, noise, device, extra_args
    )

    print("")
    print("Step 2.3: Decode the latent to get the upscaled image...")
    pixels = vae_model.decode(upscaled_latents / SD_Q)
    pixels = pixels.add(1).div(2).clamp(0, 1)

    print("")
    print("Step 3: Save the upscaled image...")
    upscaled_image = TF.to_pil_image(pixels[0])
    target_path = os.path.join(FLAGS.output_dir, "1024x1024.jpg")
    upscaled_image.save(target_path)
    print(f"Upscaled image was saved in {target_path}")


if __name__ == "__main__":
    app.run(main)
