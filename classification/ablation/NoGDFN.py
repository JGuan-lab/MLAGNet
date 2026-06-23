import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# ==========================================
# 1. Basic modules (DConv, GDFN, etc.) - unchanged
# ==========================================
class DConv(nn.Module):
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
        return self.dconv(x)

class MDTA(nn.Module):
    def __init__(self, in_dim, layers):
        super(MDTA, self).__init__()
        self.query_conv = DConv(in_dim, in_dim)
        self.key_conv = DConv(in_dim, in_dim)
        self.value_conv = DConv(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.layers = layers

    def forward(self, x):
        x_norm = F.layer_norm(x, [x.size(1), x.size(2), x.size(3)])
        proj_query = self.query_conv(x_norm).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x_norm).view(x.size(0), -1, x.size(2) * x.size(3))
        
        if self.layers < 2:
            energy = torch.bmm(proj_key, proj_query)
            proj_value = self.value_conv(x_norm).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        else:
            energy = torch.bmm(proj_query, proj_key)
            proj_value = self.value_conv(x_norm).view(x.size(0), -1, x.size(2) * x.size(3))
            
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(x.size(0), -1, x.size(2), x.size(3))
        out = self.gamma * out + x
        return out

class BaseConvBlock(nn.Module):
    def __init__(self, dim):
        super(BaseConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
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
        # based on the current code, classification temporarily uses BaseConvBlock
        # self.first = BaseConvBlock(in_dim)
        # to use LSA, uncomment the line below and comment out the line above
        self.first = MDTA(in_dim, layers)
        # self.second = GDFN(in_dim, ratio=2)
        self.second = BaseConvBlock(in_dim)

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

class Down(nn.Module):
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

# ==========================================
# 2. Encoder (downsampling only)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, Downnumblocks, n_channels=3):
        super(Encoder, self).__init__()
        self.inc = DConv(n_channels, 64)
        self.down1 = Down(64, 128, Downnumblocks[0], 0)
        self.down2 = Down(128, 256, Downnumblocks[1], 1)
        self.down3 = Down(256, 512, Downnumblocks[2], 2)
        self.down4 = Down(512, 1024, Downnumblocks[3], 3)

    def forward(self, x):
        x1 = self.inc(x)       # 64
        x2 = self.down1(x1)    # 128
        x3 = self.down2(x2)    # 256
        x4 = self.down3(x3)    # 512
        x5 = self.down4(x4)    # 1024
        # classification only needs the deepest feature x5
        return x5

# ==========================================
# 3. Classifier (classification head only)
# ==========================================
class Classifier(nn.Module):
    def __init__(self, num_classes, downblocks=[1, 1, 1, 1]):
        super(Classifier, self).__init__()
        
        # 1. backbone (Encoder Only)
        self.encoder = Encoder(Downnumblocks=downblocks, n_channels=3)

        # 2. global pooling (1024x7x7 -> 1024x1x1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 3. classification head (MLP Head)
        # input dimension is the channel count of the encoder's last layer (1024)
        self.fc_block = nn.Sequential(
            nn.Linear(1024, 256),  # dimensionality reduction
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)  # output class logits
        )

    def forward(self, x):
        # extract features
        x = self.encoder(x)  # [B, 1024, H/16, W/16]

        # pooling
        x = self.global_avg_pool(x)  # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)      # [B, 1024]

        # classify
        x = self.fc_block(x)  # [B, num_classes]
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((200, 200)),  # Resize image
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Custom Dataset class to handle image loading errors
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.transform = transform
        self.valid_samples = []  # List to store valid image paths
        self.invalid_samples = []  # List to store invalid image paths

        # Filter out invalid images
        for path, label in self.samples:
            try:
                # Attempt to load the image
                img = Image.open(path).convert('RGB')
                img.close()  # Close the image after testing
                self.valid_samples.append((path, label))
            except (IOError, OSError, Image.DecompressionBombError, Image.UnidentifiedImageError):
                self.invalid_samples.append(path)

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, index):
        path, label = self.valid_samples[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# Dataset paths
train_dir = config.ISIC_TRAIN  # Update via MLAG_DATA_ROOT env var or config.py
val_dir = config.ISIC_VAL  # Update via MLAG_DATA_ROOT env var or config.py

# Load the datasets using the custom ImageFolder
train_dataset = CustomImageFolder(root=train_dir, transform=transform)
val_dataset = CustomImageFolder(root=val_dir, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Save invalid image names to a text file
log_dir = config.LOG_DIR_AB
# invalid_image_log_path = os.path.join(log_dir, 'invalid_images.txt')
os.makedirs(log_dir, exist_ok=True)

# with open(invalid_image_log_path, 'w') as f:
#     for image_path in train_dataset.invalid_samples:
#         f.write(f"{image_path}\n")
#     for image_path in val_dataset.invalid_samples:
#         f.write(f"{image_path}\n")

# print(f"Invalid images logged to {invalid_image_log_path}")

# Check class names
num_classes = len(train_dataset.classes)
print(f'Classes: {train_dataset.classes}')

# Instantiate model
model = Classifier(num_classes).to(device)

# Define paths for saving logs and models
model_dir = config.MODEL_SAVE_AB
os.makedirs(model_dir, exist_ok=True)

# Define a path for the training log file
log_file_path = os.path.join(log_dir, 'training_logs_ISIC_NoGDFN.txt')

# Validate model function
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_loss = val_loss / len(val_loader)
    return val_loss, val_acc

# Save logs function
def save_logs_to_txt(epoch, train_loss, train_acc, val_loss, val_acc):
    with open(log_file_path, 'a') as f:
        f.write(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, '
                f'Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%\n')

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=10):
    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Clear previous log file contents if any
    open(log_file_path, 'w').close()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validate
        val_loss, val_acc = validate_model(model, val_loader)

        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save logs to text file
        save_logs_to_txt(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # Save the model with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

        scheduler.step()

    # Save the best model
    if best_model_wts is not None:
        print(f'Saving best model with Val Loss: {best_val_loss:.4f}')
        torch.save(best_model_wts, os.path.join(model_dir, 'NoGDFN_modelisc.pth'))
# Metrics evaluation + confusion matrix plotting
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import torch.nn.functional as F
img_save_dir = config.IMG_DIR_AB
# Metrics evaluation + confusion matrix plotting
def evaluate_and_plot(model_path, val_loader, class_names):
    print(f"Loading model from {model_path} for evaluation...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true = []
    y_score = []  # store probability values for ROC

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # 1. get probabilities (Softmax)
            probs = F.softmax(outputs, dim=1)

            y_score.extend(probs.cpu().numpy())
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 2. derive predicted class from probabilities (for confusion matrix)
    y_pred = np.argmax(y_score, axis=1)

    # --- plot confusion matrix ---
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # create a new figure to prevent overlap with subsequent plots
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(img_save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()  # close figure and release memory

    # --- print classification report ---
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- plot ROC curves ---
    print("Plotting ROC Curves...")

    # Binarize labels (One-hot encoding) for multi-class ROC
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    # fix: for binary classification, label_binarize returns one column; manually expand to two columns
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        # compute ROC for class i (class i vs. all others)
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # also compute micro-average ROC (optional)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # start plotting
    plt.figure(figsize=(10, 8))
    lw = 2

    # define color cycle
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # diagonal line (random chance)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # save ROC image
    roc_save_path = os.path.join(img_save_dir, "roc_curve.png")
    plt.savefig(roc_save_path, dpi=300)
    print(f"ROC curve saved to {roc_save_path}")
    plt.close()


# Define loss and optimizer with L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Start training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
evaluate_and_plot(
    model_path=os.path.join(model_dir, 'NoGDFN_modelisc.pth'),
    val_loader=val_loader,
    class_names=train_dataset.classes
)
