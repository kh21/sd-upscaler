# sd-upscaler
Upscaling images generated by Stable Diffusion. Based on https://twitter.com/StabilityAI/status/1590531946026717186

## Setup

Install the dependencies

```bash
git clone https://github.com/CompVis/stable-diffusion
git clone https://github.com/CompVis/taming-transformers
git clone https://github.com/CompVis/latent-diffusion

poetry install
poetry shell
```

and run the following command to get an 1024x1024 image

```bash
python upscale.py --prompt "a prompt you want to use" --seed 3112 --output_dir output
```
