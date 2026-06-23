# Import required modules.
import torch
from torchvision import models, transforms
from PIL import Image
import config

# Load the image.
image_path = config.DEBUG_INPUT_RESNET
image = Image.open(image_path)

# Define image transforms.
transforms = transforms.Compose([
    transforms.Resize((256,512)),  # Resize the image to 256x512.
    #transforms.CenterCrop(64),  # Center-crop the image to 224x224.
    transforms.ToTensor(),  # Convert the image to a Tensor.
 # Normalize the image.
])

# Apply the image transforms.
image = transforms(image)
image = image.unsqueeze(0)  # Add an extra batch dimension.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
print(device)
# Load the pretrained ResNet152 model, excluding the last fully-connected layer.
feature_extractor = models.resnet152(pretrained=True)
new_model = torch.nn.Sequential(*(list(feature_extractor.children())[:-3]))  # Remove the last fully-connected layer.
new_model = new_model.to('cuda')

# Feed the image into the model and extract features.
with torch.no_grad():  # Disable gradient computation to save memory.
    features = new_model(image)
    print(features.shape)
    # Print feature shape.


# Add a global average pooling layer.
'''global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
features = global_avg_pool(features)

# Map features to a 3-channel output via a fully-connected layer.
fc_layer = torch.nn.Linear(in_features=2048, out_features=3)
features = fc_layer(features)

# Print the feature shape.
print(features.shape)'''