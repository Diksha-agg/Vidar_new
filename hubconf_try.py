# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.



import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from load_pack import PackRef

import torch.nn as nn
import torch.optim as optim




# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

from vidar.arch.networks.BaseNet import BaseNet
from vidar.arch.networks.layers.packnet.packnet import \
    PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth
from vidar.utils.depth import inv2depth

from vidar.arch.losses import ReprojectionLoss


# Load PackNet model
model = PackRef(pretrained=True)
model.eval()
# Load and preprocess the image
'''
image_path = "0000000002.png"  # Replace with the path to your image file
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 1024)),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor

])

input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension to the input tensor
print("Input tensor shape: {}".format(input_tensor.shape))



# Perform inference
output = model(input_tensor)

'''
'''
output_arrays = [tensor.detach().cpu().numpy() for tensor in output]
# Save the list of arrays as a single NPZ file
np.savez('output.npz', *output_arrays)
'''
'''
# Print the shape of the output
#print("Output shape:", (output))
#print("weights:",model.state_dict().shape)

print("parameters:", model.parameters())

# Freeze the parameters of the PackNet model
for param in model.parameters():
    param.requires_grad = False
    
    '''
'''
# Append your own network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Define your own layers
        
        
        
        self.conv1 = nn.Conv2d(3, 512, kernel_size=1)  # Example convolutional layer
        self.conv2 = nn.Conv2d(512, 10, kernel_size=1)  # Example convolutional layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through your own layers
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

'''



class MyNetwork(BaseNet, ABC):
    """
    PackNet depth network (https://arxiv.org/abs/1905.02693)

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        # Configuration parameters
        self.min_depth = 0.5
        self.dropout = 0.0

        # Input/output channels
        in_channels = 3
        out_channels = 1

        # Hyper-parameters
        ni, no = 64, out_channels
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]

        # Initial convolutional layer
        #self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        # Support for different versions
        n1o, n1i = n1, n1 + ni + no
        n2o, n2i = n2, n2 + n1 + no
        n3o, n3i = n3, n3 + n2 + no
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4

        # Encoder

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0])
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1])

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=self.dropout)

        # Decoder


        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3])
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4])


        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp2_layer = InvDepth(n2, out_channels=out_channels, min_depth=self.min_depth)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels, min_depth=self.min_depth)

        self.init_weights()

    def init_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, intrinsics=None):
        """Network forward pass"""

        # Initial convolution

        #x = self.pre_calc(rgb)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)


        # Skips

        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        # Decoder
        
        unpack2 = self.unpack2(x2p)
        concat2 = torch.cat((unpack2, skip2), 1)
        iconv2 = self.iconv2(concat2)
        inv_depth2 = self.disp2_layer(iconv2)
        up_inv_depth2 = self.unpack_disp2(inv_depth2)

        unpack1 = self.unpack1(iconv2)
        concat1 = torch.cat((unpack1, skip1, up_inv_depth2), 1)
        iconv1 = self.iconv1(concat1)
        inv_depth1 = self.disp1_layer(iconv1)
        if self.training:
            inv_depths = [inv_depth1, inv_depth2]
        else:
            inv_depths = [inv_depth1]


        return {
            'depths': inv2depth(inv_depths),
        }
    


# Create an instance of your network
#my_network = MyNetwork()

# Move the model to the GPU
model = model.cuda()
#my_network = my_network.cuda()

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(my_network.parameters(), lr=0.001)

# Print the summary of your network
#summary(my_network, (3, 256, 256))  # Adjust the input shape as per your requirement

print("networks created")
image_path = "0000000002.png"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).cuda()
print("Image loaded")
# Pass the image through the PackNet feature extractor
features = model(input_tensor)
print("packnet done")
# Forward pass through your network
outputs = MyNetwork(features)
print("mynetworks done")
print('output:',outputs)





























