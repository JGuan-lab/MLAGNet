import torch
from thop import profile
from torchvision import datasets, transforms, models
import torch.nn as nn
# model = models.shufflenet_v2_x0_5(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 10)  # replace the fully-connected layer
model = models.vit_b_16(pretrained=True)  # use ViT-b-16 as backbone

# get the input feature size of the current classification head
in_features = model.heads.head.in_features  # access the linear layer inside the classification head
model.heads.head = nn.Linear(in_features, 10)
input_tensor = torch.randn(1,3,224,224)
flops,params = profile(model,inputs = (input_tensor,))
print(f"==========================vit")
print(f"FLops:{flops/1e9:.2f}G")
print(f"Params:{params/1e6:.2f}M")