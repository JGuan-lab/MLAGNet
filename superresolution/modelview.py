# For the case where a network model exists but no trained .pth file has been saved yet.
import netron
import torch.onnx
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf
from torchvision.models import resnet18  # Using resnet18 as an example.
class ResBlock(nn.Module):
    """Residual block."""

    def __init__(self, inChannals, outChannals):
        """Initialize the residual block."""
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(outChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(outChannals, outChannals, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        """Forward pass."""
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)

        out += resudial
        out = self.relu(out)
        return out

class Generator(nn.Module):
    """Generator model (4x upscaling)."""

    def __init__(self):
        """Initialize the model configuration."""
        super(Generator, self).__init__()
        # Convolution block 1.
        self.conv0 = nn.Conv2d(3, 32, kernel_size=9, padding=2, padding_mode='reflect', stride=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=9, padding=2, padding_mode='reflect', stride=1)
        self.conv3 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode='reflect', stride=1)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode='reflect', stride=1)
        self.relu = nn.PReLU()
        # Residual block.
        self.resBlock = self._makeLayer_(ResBlock, 64, 64, 5)
        # Convolution block 2.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()

        # Sub-pixel convolution.
        self.convPos1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=2, padding_mode='reflect')
        self.pixelShuffler1 = nn.PixelShuffle(2)
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pixelShuffler2 = nn.PixelShuffle(2)
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv2d(64, 3, kernel_size=9, stride=1)



    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """Build residual layers."""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.relu(x)
        residual = x
        out = self.resBlock(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.convPos1(out)
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)
        out = self.convPos2(out)
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)
        out = self.finConv(out)

        return out

myNet = Generator()  # Instantiate the Generator model.
x = torch.randn(1,1,512,512)  # Generate a random input tensor.
modelData = "./demo.pth"  # Define the path where model data will be saved.
# modelData = "./demo.onnx"  # Some suggest using an onnx file, but pth works too.
torch.onnx.export(myNet, x, modelData)  # Export the PyTorch model in ONNX format and save it.
netron.start(modelData)  # Visualize the network structure.

# For the case where a trained .pth file already exists.
import netron

modelData = "./demo.pth"  # Define the path where model data is saved.
netron.start(modelData)  # Visualize the network structure.

