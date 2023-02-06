import sys

sys.path.extend(["./taming-transformers", "./stable-diffusion", "./latent-diffusion"])

import huggingface_hub
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from requests.exceptions import HTTPError


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model = model.to("cpu").eval().requires_grad_(False)
    return model


def download_from_huggingface(repo, filename):
    while True:
        try:
            return huggingface_hub.hf_hub_download(repo, filename)
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(
                    f"Go here and agree to the click through license on your account: https://huggingface.co/{repo}"
                )
                input("Hit enter when ready:")
                continue
            else:
                raise e
