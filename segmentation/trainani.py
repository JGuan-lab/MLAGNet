import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import Classifier
from torchvision import transforms  # Import torchvision transforms module

# Add transforms to MedicalImageDataset
class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform

        self.img_list = os.listdir(img_dir)

        # Define default transform (if none provided)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()  # Convert image to tensor
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        # mask_path = os.path.join(self.mask_dir, self.img_list[idx])
         # Replace img_list[idx] extension with .png
        mask_name = os.path.splitext(self.img_list[idx])[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        #mask = Image.open(mask_path)

        # Apply transform
        image = self.transform(image)


         # Resize and convert mask only
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = transforms.ToTensor()(mask).long()  # Convert to tensor and ensure long type
        mask = mask.squeeze(0)  # Remove channel dimension, resulting shape: [H, W]

        return image, mask


# Training loss functions
# Implement Dice loss
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
#         inputs = torch.sigmoid(inputs)  # Use Sigmoid to convert output to probabilities
#         inputs = inputs.view(-1)
#         targets = targets.reshape(-1)  # Use reshape instead of view
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return 1 - dice
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply sigmoid to convert output to probabilities
        preds = torch.sigmoid(preds)

        # Flatten tensor using reshape instead of view
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        # Compute TP, FP, FN
        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()

        # Compute Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Return Focal Tversky Loss
        return (1 - tversky_index) ** self.gamma


# Define Hybrid Loss: cross-entropy + Focal Tversky
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss with logits
        self.floss = FocalTverskyLoss()

    def forward(self, inputs, targets):
        # inputs: predicted logits
        # targets: ground-truth labels

        # Convert targets to one-hot encoding, ensuring long type and correct dimensions
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()

        # Compute BCE loss
        bce_loss = self.bce(inputs, targets_one_hot)

        # Compute Focal Tversky loss
        focus_loss = self.floss(inputs, targets_one_hot)

        # Return hybrid loss with weighted BCE and Focal Tversky
        return 0.3 * bce_loss + 0.7 * focus_loss

# Training function
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)
# Dice coefficient computation function
def dice_coefficient(preds, targets, smooth=1e-6):
    # Apply softmax or sigmoid to prediction output to get probabilities
    preds = torch.softmax(preds, dim=1)  # Apply softmax to 2-channel output to get probability distribution

    # Print shapes of preds and targets
    # print(f"Preds shape (before argmax): {preds.shape}, Targets shape: {targets.shape}")

    # Use argmax to convert output to single channel representing per-pixel predicted class
    preds = torch.argmax(preds, dim=1)

    # Print converted preds shape
    # print(f"Preds shape (after argmax): {preds.shape}, Targets shape: {targets.shape}")

    # Flatten tensors to ensure shape match
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Compute intersection and Dice coefficient
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice


def train_model(model, train_loader, val_loader, device, cfg):
    criterion = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['TRAIN']['learning_rate'], weight_decay=1e-4)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_dices, val_dices = [], []  # Track Dice coefficient
    best_val_accuracy = 0
    best_val_iou = 0
    best_val_dice = 0  # Track best Dice coefficient
    with open(os.path.join(cfg['TRAIN']['log_dir'], 'train_log_ani02.txt'), 'w') as log_file:
        log_file.write('Epoch, Train Loss, Train Accuracy, Train IoU, Train Dice\n')  # Add Train Dice

        for epoch in range(cfg['TRAIN']['epochs']):
            model.train()
            running_loss = 0.0
            correct = 0
            total_pixels = 0
            epoch_dice = 0.0  # Accumulate Dice coefficient

            intersection = np.zeros(cfg['DATASET']['num_class'])
            union = np.zeros(cfg['DATASET']['num_class'])

            for i, (images, masks) in enumerate(tqdm(train_loader)):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                print(f"Inputs shape: {outputs.shape}, Targets shape: {masks.shape}")
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Compute per-pixel predictions
                preds = torch.argmax(outputs, dim=1)

                # Compute accuracy
                correct += (preds == masks).sum().item()
                total_pixels += masks.numel()

                # Compute IoU
                intersection_tmp, union_tmp = intersectionAndUnion(preds.cpu().numpy(), masks.cpu().numpy(), 2)
                intersection += intersection_tmp
                union += union_tmp

                # Compute and accumulate Dice coefficient per batch
                epoch_dice += dice_coefficient(outputs, masks)

                if i == 0:
                    visualize_and_save_result(images[0], masks[0], preds[0], epoch, cfg['TRAIN']['mid_dir'], stage='train')

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total_pixels
            epoch_dice = epoch_dice / len(train_loader)  # Average Dice coefficient
            iou = intersection / (union + 1e-10)  # Avoid division by zero

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            train_dices.append(epoch_dice)  # Record Dice coefficient for training

            log_file.write(f"{epoch + 1}, {epoch_loss:.4f}, {epoch_accuracy:.4f}, {iou.mean():.4f}, {epoch_dice:.4f}\n")

            print(len(val_loader))
            val_loss, val_accuracy, val_iou, val_dice = validate_and_save_result(model, val_loader, device, cfg, epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_dices.append(val_dice)  # Record Dice coefficient for val

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = os.path.join(cfg['TRAIN']['model_dir'], f'model02_bestacc_{best_val_accuracy}.pth')
                torch.save(model.state_dict(), model_save_path)

            if val_iou.mean() > best_val_iou:
                best_val_iou = val_iou.mean()
                model_save_path = os.path.join(cfg['TRAIN']['model_dir'], f'model40_bestiou_{best_val_iou}.pth')
                torch.save(model.state_dict(), model_save_path)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                model_save_path = os.path.join(cfg['TRAIN']['model_dir'], f'model40_bestdice_{best_val_dice:.4f}.pth')
                torch.save(model.state_dict(), model_save_path)

            if epoch == cfg['TRAIN']['epochs'] - 1:
                # Write validation results to log file
                with open(os.path.join(cfg['TRAIN']['log_dir'], 'ani_02_val_log.txt'), 'a') as val_log_file:
                    val_log_file.write(f"  Accuracy: {best_val_accuracy:.4f}, IoU: {best_val_iou:.4f}, Dice: {best_val_dice:.4f}\n")

        # Record best IoU at the end of the log
        log_file.write(f"Best IoU achieved during training: {best_val_iou:.4f}\n")



# Updated validation function
def validate_and_save_result(model, val_loader, device, cfg, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_pixels = 0
    epoch_dice = 0.0  # Dice coefficient

    intersection = np.zeros(cfg['DATASET']['num_class'])
    union = np.zeros(cfg['DATASET']['num_class'])

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Compute per-pixel predictions
            preds = torch.argmax(outputs, dim=1)

            # Compute accuracy
            correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # Compute IoU
            intersection_tmp, union_tmp = intersectionAndUnion(preds.cpu().numpy(), masks.cpu().numpy(), 2)
            intersection += intersection_tmp
            union += union_tmp

            # Compute and accumulate Dice coefficient per batch
            epoch_dice += dice_coefficient(outputs, masks)

            if i == 0:  # Save validation sample
                visualize_and_save_result(images[0], masks[0], preds[0], epoch, cfg['TRAIN']['result_dir'], stage='val')

            break

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total_pixels
    val_iou = intersection / (union + 1e-10)  # Avoid division by zero
    val_dice = epoch_dice / len(val_loader)  # Average Dice coefficient

    # Write validation results to log file
    with open(os.path.join(cfg['TRAIN']['log_dir'], 'ani_02_val_log.txt'), 'a') as val_log_file:
        val_log_file.write(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, IoU: {val_iou.mean():.4f}, Dice: {val_dice:.4f}\n")

    return val_loss, val_accuracy, val_iou, val_dice


# Visualization and result saving
def visualize_and_save_result(image, mask, pred, epoch, result_dir, stage='train'):
    image = image.cpu().numpy().transpose(1, 2, 0)
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Prediction')

    plt.suptitle(f'Epoch {epoch} - {stage}')
    plt.savefig(os.path.join(result_dir, f'result_epoch_{epoch}_{stage}.png'))
    plt.close()

# Configuration parameters
cfg = {
    'TRAIN': {
        'learning_rate': 0.0001,
        'epochs': 500,
        'model_dir': config.ANI_MODEL_DIR,
        'log_dir': config.ANI_LOG_DIR,
        'result_dir': config.RESULT_DIR,
        'mid_dir': config.MID_DIR

    },
    'DATASET':{
        'num_class': 2
    }
}

# Usage example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = MedicalImageDataset(img_dir=config.ANI_TRAIN_IMG, mask_dir=config.ANI_TRAIN_LABEL, img_size=512)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = MedicalImageDataset(img_dir=config.ANI_TEST_IMG, mask_dir=config.ANI_TEST_LABEL, img_size=512)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
upblocks=[0, 0, 0, 0]
downblocks=[1, 1, 0, 0]
model = Classifier(2,upblocks,downblocks)  # Assumed segmentation model
model.to(device)

# Start training
train_model(model, train_loader, val_loader, device, cfg)
