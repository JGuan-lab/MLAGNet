import torch
import torch.nn as nn
import torch.nn.functional as F


class DConv(nn.Module):  # Depthwise separable convolution
    def __init__(self, inchannel, outchannel):
        super(DConv, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, groups=inchannel),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=False),
            nn.Conv2d(inchannel, outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.dconv(x)
        return out


class MDTA(nn.Module):
    def __init__(self, in_dim, layers):
        super(MDTA, self).__init__()
        self.query_conv = DConv(in_dim, in_dim)
        self.key_conv = DConv(in_dim, in_dim)
        self.value_conv = DConv(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.layers = layers

    def forward(self, x):
        x = F.layer_norm(x, [x.size(1), x.size(2), x.size(3)])

        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        if self.layers < 2:
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


class GDFN(nn.Module):
    def __init__(self, in_dim, ratio):
        super(GDFN, self).__init__()
        self.upconv1 = nn.Conv2d(in_dim, ratio * in_dim, kernel_size=1)
        self.lowconv1 = nn.Conv2d(in_dim, ratio * in_dim, kernel_size=1)

        self.uppath = DConv(ratio * in_dim, ratio * in_dim)
        self.lowpath = DConv(ratio * in_dim, ratio * in_dim)
        self.relu = nn.SiLU(inplace=False)
        self.finconv = nn.Conv2d(ratio * in_dim, in_dim, kernel_size=1)

    def forward(self, x):
        temp = x
        x = F.layer_norm(x, [x.size(1), x.size(2), x.size(3)])
        up = self.upconv1(x)
        up = self.uppath(up)
        low = self.lowconv1(x)
        low = self.lowpath(low)
        low = self.relu(low)
        out = low * up
        out = self.finconv(out)
        out = temp + out
        return out


class Transformerblock(nn.Module):
    def __init__(self, in_dim, layers):
        super(Transformerblock, self).__init__()
        self.first = MDTA(in_dim, layers)
        self.second = GDFN(in_dim, ratio=2)

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double depthwise separable conv"""

    def __init__(self, in_channels, out_channels, nums, layers):
        super().__init__()
        self.blocks = nn.Sequential(*[Transformerblock(in_channels, layers) for _ in range(nums)])
        self.maxpool_conv = nn.MaxPool2d(2)
        self.conv = DConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool_conv(x)
        x = self.blocks(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, Downnumblocks, n_channels=3, bilinear=True):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inc = DConv(n_channels, 64)
        self.down1 = Down(64, 128, Downnumblocks[0], 0)
        self.down2 = Down(128, 256, Downnumblocks[1], 1)
        self.down3 = Down(256, 512, Downnumblocks[2], 2)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024, Downnumblocks[3], 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5


class Classifier(nn.Module):
    def __init__(self, num_classes, downblocks=[2, 2, 2, 2]):
        super(Classifier, self).__init__()
        self.encoder = Encoder(Downnumblocks=downblocks)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        print(num_classes)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.encoder(x)  # Extract features via encoder
        x = self.global_avg_pool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten feature map into vector
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# # Example: input for image classification
# batch_size = 1
# num_channels = 3
# image_height = 256
# image_width = 448
# num_classes = 10  # Assume 10 classes
#
# # Create random input image (for example only)
# dummy_images = torch.randn(batch_size, num_channels, image_height, image_width)
#
# # Initialize classification network
# classification_model = Classifier(num_classes=num_classes)
#
# # Forward pass
# output_logits = classification_model(dummy_images)
#
# # Print classification result
# print(output_logits.shape)  # Output should be (batch_size, num_classes)
