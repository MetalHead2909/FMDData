# CA Block

import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CABlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CABlock, self).__init__()

        # Point-wise convolutional layer to reduce the number of channels
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        
        # Batch normalization layer for the reduced channels
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        
        # Activation function using h-swish
        self.act = h_swish()

        # Point-wise convolutional layers for channel-wise attention
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Store the identity for later use
        identity = x

        # Get the batch size, number of channels, height, and width of the input tensor
        n, c, h, w = x.size()
        #print("shape:\n",n, c, h, w)

        # Spatial pooling along height and width separately
        x_h = F.avg_pool2d(x, kernel_size=(1, w))  # Average pooling along height
        #print("x_h before",x_h.shape)
        x_w = F.avg_pool2d(x, kernel_size=(h, 1))
        #print("x_h before",x_h.shape)
        #print("x_w before",x_w.shape)
        x_w = x_w.permute(0, 1, 3, 2)  # Average pooling along width
        #print("x_w after permute",x_w.shape)
        # Concatenate the pooled features along the channel dimension
        y = torch.cat([x_h, x_w], dim=2)
        #print("y before",y.shape)

        # Apply 1x1 convolution followed by batch normalization and activation function
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        #print("y after",y.shape)

        # Split the concatenated features back into separate height and width features
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Transpose the width features to match the original shape
        #print("x_h after split",x_h.shape)
        #print("x_w after split",x_w.shape)
        # Apply channel-wise attention using 1x1 convolutions and sigmoid activation
        a_h = self.conv_h(x_h).sigmoid()  # Attention weights for height
        a_w = self.conv_w(x_w).sigmoid()  # Attention weights for width
        #print("a_h weights",a_h.shape)
        #print("a_w weights",a_w.shape)
        # Apply the attention weights to the original input tensor
        weights = a_w * a_h
        #print("weights shape",weights.shape)
        #print("initial shape",identity.shape)
        out = identity * weights
        
        #print("final shape",out.shape)

        return out
 