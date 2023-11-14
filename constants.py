class Models:
    ERFNET = "erfnet"
    ERFNET_W_ALEATORIC = "erfnet_w_aleatoric"
    UNET = "unet"
    UNET_W_ALEATORIC = "unet_w_aleatoric"


class Losses:
    CROSS_ENTROPY = "xentropy"
    MSE = "mse"
    NLL = "nll"
    ALEATORIC = "aleatoric"


IGNORE_INDEX = {"cityscapes": 19, "potsdam": 0, "flightmare": 9, "shapenet": None}
