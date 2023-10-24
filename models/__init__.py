import torch
import yaml
from constants import Models
from models.network import ERFNet, AleatoricERFNet, UNet, AleatoricUNet
from pytorch_lightning.core.lightning import LightningModule


def get_model(cfg) -> LightningModule:
    name = cfg["model"]["name"]
    if isinstance(cfg, dict):
        if name == Models.ERFNET:
            return ERFNet(
                cfg,
            )
        elif name == Models.ERFNET_W_ALEATORIC:
            return AleatoricERFNet(
                cfg,
            )
        elif name == Models.UNET:
            return UNet(
                cfg,
            )
        elif name == Models.UNET_W_ALEATORIC:
            return AleatoricUNet(
                cfg,
            )
        else:
            RuntimeError(f"{name} model not implemented")
    else:
        raise RuntimeError(f"{type(cfg)} not a valid config")


def load_pretrained_model(config_path: str, checkpoint_path: str) -> LightningModule:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    return get_model(cfg).load_from_checkpoint(checkpoint_path, cfg=cfg).to(device)
