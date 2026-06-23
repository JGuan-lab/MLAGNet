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
class MDTA(nn.Module):
    def __init__(self, in_dim,layers):
        super(MDTA, self).__init__()
        self.query_conv = DConv(in_dim,in_dim)
        self.key_conv=DConv(in_dim,in_dim)
        self.value_conv=DConv(in_dim,in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.layers=layers
    def forward(self, x):
        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        if self.layers<2:
            energy = torch.bmm(proj_key, proj_query)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        else:
            energy = torch.bmm(proj_query, proj_key)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(x.size(0), -1, x.size(2), x.size(3))
        out = self.gamma * out + x
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
residual_attention_block = MDTA(64)

# Forward pass
output_features = residual_attention_block(dummy_images)

# Printing the output features shape
print("Output features shape:", output_features.shape)
