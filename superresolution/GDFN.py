import torch
import torch.nn as nn


class DConv(nn.Module):  # depthwise separable convolution
    def __init__(self,inchannel,outchannel):
        super(DConv,self).__init__()
        self.dconv=nn.Sequential(
            nn.Conv2d(inchannel,inchannel,kernel_size=3,padding=1,groups=inchannel),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel,outchannel,kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out=self.dconv(x)
        return out
class GDFN(nn.Module):
    def __init__(self,in_dim,ratio):
        super(GDFN,self).__init__()
        self.upconv1=nn.Conv2d(in_dim,ratio*in_dim,kernel_size=1)
        self.lowconv1 = nn.Conv2d(in_dim, ratio*in_dim, kernel_size=1)
        self.uppath=DConv(ratio*in_dim,ratio*in_dim)
        self.lowpath=DConv(ratio*in_dim,ratio*in_dim)
        self.relu=nn.SiLU(inplace=True)
        self.finconv=nn.Conv2d(ratio*in_dim,in_dim,kernel_size=1)
    def forward(self,x):
        temp=x
        x=nn.LayerNorm([x.size(1),x.size(2),x.size(3)])(x)
        # upper branch
        up=self.upconv1(x)
        up=self.uppath(up)
        # lower branch
        low=self.lowconv1(x)
        low=self.lowpath(low)
        low=self.relu(low)
        out=low*up
        out=self.finconv(out)
        out=temp+out
        return out


# Example usage:
batch_size = 1
num_channels = 64
image_height = 32
image_width = 32

# Creating random input images (for demonstration purpose only)
dummy_images = torch.randn(batch_size, num_channels, image_height, image_width)
dummy_images = torch.clamp(dummy_images, 0, 1)

# Creating the residual attention block
residual_attention_block = GDFN(64,4)

# Forward pass
output_features = residual_attention_block(dummy_images)

# Printing the output features shape
print("Output features shape:", output_features.shape)
