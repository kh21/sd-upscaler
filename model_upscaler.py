import k_diffusion as K
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TOL_SCALE = 0.25
ETA = 1.0


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1.0, embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode="nearest") * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            input,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs,
        )


class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
            # Shortcut for when we don't need to run both.
            if self.cond_scale == 0.0:
                c_in = self.uc
            elif self.cond_scale == 1.0:
                c_in = c
            return self.inner_model(
                x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in
            )

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(
            x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in
        ).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        from transformers import CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok_out["input_ids"][indexer]
        attention_mask = 1 - tok_out["attention_mask"][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPTextModel, logging

        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = self.transformer.eval().requires_grad_(False).to(device)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(
            input_ids=input_ids.to(self.device), output_hidden_states=True
        )
        return (
            clip_out.hidden_states[-1],
            cross_cond_padding.to(self.device),
            clip_out.pooler_output,
        )


def make_upscaler_model(
    config_path, model_path, pooler_dim=768, train=False, device="cpu"
):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config["model"]["sigma_data"],
        embed_dim=config["model"]["mapping_cond_dim"] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_ema"])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


def do_sample(model_wrap, noise, device, extra_args, tol_scale=TOL_SCALE, eta=ETA):
    # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
    # TODO Check if this is model dependent.
    sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512
    sampler_opts = dict(
        s_noise=1.0,
        rtol=tol_scale * 0.05,
        atol=tol_scale / 127.5,
        pcoeff=0.2,
        icoeff=0.4,
        dcoeff=0,
    )
    return K.sampling.sample_dpm_adaptive(
        model_wrap,
        noise * sigma_max,
        sigma_min,
        sigma_max,
        extra_args=extra_args,
        eta=eta,
        **sampler_opts,
    )
