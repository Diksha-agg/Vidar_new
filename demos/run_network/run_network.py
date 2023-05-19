# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch



from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


from vidar.arch.networks.depth.MonoDepthResNet import MonoDepthResNet
from vidar.utils.config import read_config

### Create network

cfg = read_config('demos/run_network/config.yaml')
net = MonoDepthResNet(cfg)

### Create dummy input and run network
'''
rgb = torch.randn((2, 3, 128, 128))
depth = net(rgb=rgb)['depths']
print(depth)
'''

# Load and preprocess the image
image_path = "frame0.jpg"  # Replace with the path to your image file
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])
input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension to the input tensor
depth = net(rgb=input_tensor)['depths']
print(depth)

