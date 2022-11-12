import torch

from .model import *


def build_model(cfg):

    if cfg.model == 'PCReg_KPURes18_MSF':
        model = PCReg_KPURes18_MSF(cfg)
    elif cfg.model == "no_model":
        print("Warning: no model is being loaded; rarely correct thing to do")
        model = torch.nn.Identity()
    else:
        raise ValueError("Model {} is not recognized.".format(cfg.name))

    """
    We will release code for baselines soon. This has been delayed since some of the
    baselines have particular licenses that we want to make sure we're respecting.
    """

    return model