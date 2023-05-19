# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.


dependencies = ["torch"]

import torch

from vidar.core.evaluator import Evaluator
from vidar.utils.config import read_config
from vidar.utils.setup import setup_arch
from vidar.utils.config import cfg_has
from imageio import imread, imsave
import torch.nn.functional as F
import torch
import urllib


def PackRef(pretrained=True, **kwargs):
    """# This docstring shows up in hub.help()
    PackNet model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    #cfg = read_config("ModelRef_config.yaml")
    cfg = read_config("packnet_config.yaml")
    model = Evaluator(cfg)

    if pretrained:
        repo = "TRI-ML/vidar"
        packnet_model = torch.hub.load(repo, "PackNet", pretrained=True, trust_repo=True)

        torch.save(packnet_model.state_dict(), "packnet_weights.pth")
        
        model = setup_arch(cfg.arch, verbose=True)
        state_dict = torch.load("packnet_weights.pth", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model
    
