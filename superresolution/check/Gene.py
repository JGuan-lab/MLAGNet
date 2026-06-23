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
        # self.ln = nn.LayerNorm()

    def forward(self, x):
        # x = x.to(next(self.parameters()).device)
        # x = nn.LayerNorm([x.size(1), x.size(2), x.size(3)])(x)
        x = F.layer_norm(x, [x.size(1), x.size(2), x.size(3)])
        # x = self.ln(x)  # Use the input tensor's dimensions to define the LayerNorm normalization shape

        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        if self.layers < 2:
            energy = torch.bmm(proj_key, proj_query)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)

        else:
            energy = torch.bmm(proj_query, proj_key)
            # print("layer index", self.layers)
            # print("energy shape", energy.shape)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
            # print("v shape", proj_value.shape)
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

        # x = nn.LayerNorm(x.size()[1:])(x)  # Use the input tensor's dimensions to define the LayerNorm normalization shape
        x = F.layer_norm(x, [x.size(1), x.size(2), x.size(3)])
        # x = nn.LayerNorm([x.size(1), x.size(2), x.size(3)])(x)
        # Upper branch
        up = self.upconv1(x)
        up = self.uppath(up)
        # Lower branch
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


class Up(nn.Module):
    """Upscaling then double depthwise separable conv"""

    def __init__(self, in_channels, out_channels, nums, layers, bilinear=True):
        super().__init__()
        self.layers = layers
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.blocks = nn.Sequential(*[Transformerblock(in_channels, self.layers) for _ in range(nums)])
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.blocks = nn.Sequential(*[Transformerblock(in_channels, self.layers) for _ in range(nums)])
            self.conv = DConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.blocks(x1)
        x1 = self.up(x1)
        x1 = self.conv(x1)
        # print("size after upsampling the bottom layer", x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        # print("size after concatenation", x.shape)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, Downnumblocks, n_channels=3, n_classes=64, bilinear=True):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        return x5, x4, x3, x2, x1


class WeightedFeatureAggregation(nn.Module):
    def __init__(self, num_features=4, initial_weights=[0.5, 0.3, 0.15, 0.05]):
        super(WeightedFeatureAggregation, self).__init__()
        self.num_features = num_features
        # Define weights as learnable parameters
        self.weights = nn.Parameter(torch.tensor(initial_weights), requires_grad=True)

    def forward(self, feature_maps):
        # Apply weights to each feature map
        weighted_feature_maps = [feature_maps[i] * self.weights[i] for i in range(self.num_features)]
        # Sum the weighted feature maps
        fused_feature_map = torch.stack(weighted_feature_maps, dim=0).sum(dim=0)
        return fused_feature_map


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(YOLOHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Output bbox and class scores for each anchor
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        # Output shape: [batch_size, num_anchors * (5 + num_classes), grid_size, grid_size]
        output = self.conv(x)
        return output


class Decoder(nn.Module):
    def __init__(self, downblocks=[1, 1, 1, 1], Upblocks=[1, 1, 1, 1], num_anchors=3, num_classes=2, bilinear=True):
        super(Decoder, self).__init__()
        self.downblocks = downblocks
        self.Upblocks = Upblocks
        self.branch1 = Encoder(Downnumblocks=self.downblocks)
        self.up1 = Up(1024, 512, self.Upblocks[0], 3, bilinear)
        self.up2 = Up(512, 256, self.Upblocks[1], 2, bilinear)
        self.up3 = Up(256, 128, self.Upblocks[2], 1, bilinear)
        self.up4 = Up(128, 64, self.Upblocks[3], 0, bilinear)

        # Use YOLOHead as the output layer
        self.yolo_head = YOLOHead(in_channels=1024, num_anchors=num_anchors, num_classes=num_classes)

    def forward(self, x):
        x, x4, x3, x2, x1 = self.branch1(x)
        print(x.shape)
        # up1 = self.up1(x, x4)
        # print(up1.shape)
        # up2 = self.up2(up1, x3)
        # up3 = self.up3(up2, x2)
        # up4 = self.up4(up3, x1)

        # Logits output from the YOLO head
        yolo_output = self.yolo_head(x)

        return yolo_output



batch_size = 1
num_channels = 3
image_height =  224
image_width = 224
num_classes = 2  # Assume 2 classes

downblocks=[1, 1, 1, 1]
Upblocks=[1, 1, 1, 1]
# Create a random input image (for demonstration only)
dummy_images = torch.randn(batch_size, num_channels, image_height, image_width)

# Initialize the classification network
classification_model = Decoder(downblocks,Upblocks)

# Forward pass
output_logits = classification_model(dummy_images)

# Print the classification output
print(output_logits.shape)  # Expected shape: (batch_size, num_classes)
