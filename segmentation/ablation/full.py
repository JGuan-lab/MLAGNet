import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms  # Import torchvision transforms module
import torch.nn.functional as F
from thop import profile

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
        # x = self.ln(x)  # Use input tensor dimensions to define LayerNorm normalization shape

        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        if self.layers < 2:
            energy = torch.bmm(proj_key, proj_query)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)

        else:
            energy = torch.bmm(proj_query, proj_key)
            # print("layer index", self.layers)
            # print("energy size", energy.shape)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
            # print("v size", proj_value.shape)
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

        # x = nn.LayerNorm(x.size()[1:])(x)  # Use input tensor dimensions to define LayerNorm normalization shape
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



class BaseConvBlock(nn.Module):
    def     __init__(self,dim):
        super(BaseConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(dim,dim,kernel_size=3,padding = 1)
        self.act = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(dim,dim,kernel_size = 3,padding = 1)
    def forward(self,x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out
class Transformerblock(nn.Module):
    def __init__(self,in_dim,layers):
        super(Transformerblock,self).__init__()
        self.first=MDTA(in_dim,layers)
        #self.first = BaseConvBlock(in_dim)
        self.second=GDFN(in_dim,ratio=2)
        #self.second = BaseConvBlock(in_dim)
    def forward(self,x):
        x=self.first(x)
        x=self.second(x)
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

        # Use the current upsampled size here
        x1 = self.conv(x1)

        # Resize to match x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        else:
            x1 = x1[:, :, :x2.size(2), :x2.size(3)]  # Crop if x1 is larger than x2

        x = torch.cat([x2, x1], dim=1)
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
        self.weights = nn.Parameter(torch.tensor(initial_weights), requires_grad=True)

    def forward(self, feature_maps):
        # Ensure all feature maps have the same size
        target_size = feature_maps[0].size()[2:]  # Use the size of the first feature map
        weighted_feature_maps = []

        for i in range(self.num_features):
            # Resize feature map
            if feature_maps[i].size()[2:] != target_size:
                weighted_feature_maps.append(F.interpolate(feature_maps[i], size=target_size, mode='bilinear', align_corners=True) * self.weights[i])
            else:
                weighted_feature_maps.append(feature_maps[i] * self.weights[i])

        fused_feature_map = torch.stack(weighted_feature_maps, dim=0).sum(dim=0)
        return fused_feature_map



class Decoder(nn.Module):
    def __init__(self, downblocks, Upblocks, n_classes=64, bilinear=True):
        super(Decoder, self).__init__()
        self.downblocks = downblocks
        self.Upblocks = Upblocks
        self.branch1 = Encoder(Downnumblocks=self.downblocks)
        self.up1 = Up(1024, 512, self.Upblocks[0], 3, bilinear)
        self.up2 = Up(512, 256, self.Upblocks[1], 2, bilinear)
        self.up3 = Up(256, 128, self.Upblocks[2], 1, bilinear)
        self.up4 = Up(128, 64, self.Upblocks[3], 0, bilinear)
        self.upconv1 = OutConv(512, n_classes)
        self.upconv2 = OutConv(256, n_classes)
        self.upconv3 = OutConv(128, n_classes)

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.PixelShuffle(upscale_factor=2))
        self.weight = WeightedFeatureAggregation()
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x, x4, x3, x2, x1 = self.branch1(x)
        up1 = self.up1(x, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)
        up1p = self.upconv1(up1)
        up1p = self.upsample(up1p)
        up1p = self.upsample(up1p)
        up1pp = self.upsample(up1p)
        up2p = self.upconv2(up2)
        up2p = self.upsample(up2p)
        up2pp = self.upsample(up2p)
        up3p = self.upconv3(up3)
        up3pp = self.upsample(up3p)
        feature_maps = [up1pp, up2pp, up3pp, up4]
        out = self.weight(feature_maps)

        logits = self.outc(up4)
        return logits

class Classifier(nn.Module):

    def __init__(self, num_classes,  upblocks, downblocks):
        super(Classifier, self).__init__()
        self.mod_pad_h = 0
        self.mod_pad_w = 0
        self.decoder = Decoder(downblocks=downblocks, Upblocks=upblocks, n_classes=64)
        self.fc1 = nn.Conv2d(64,num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        x = self.decoder(x)
        x = self.fc1(x)

        return x
# Add transform to MedicalImageDataset
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
# Implement Dice loss function
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
#         inputs = torch.sigmoid(inputs)  # Apply sigmoid to convert output to probabilities
#         inputs = inputs.view(-1)
#         targets = targets.reshape(-1)  # Flatten tensor using reshape instead of view
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

    # print preds and targets shapes
    # print(f"Preds shape (before argmax): {preds.shape}, Targets shape: {targets.shape}")

    # Use argmax to convert output to single channel representing per-pixel predicted class
    preds = torch.argmax(preds, dim=1)

    # print converted preds shape
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
    with open(os.path.join(cfg['TRAIN']['log_dir'], 'train_log_retinaabi02_full.txt'), 'w') as log_file:
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
            val_dices.append(val_dice)  # Record Dice coefficient for validation

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = os.path.join(cfg['TRAIN']['model_dir'], f'model02_bestacc_{best_val_accuracy}.pth')
                torch.save(model.state_dict(), model_save_path)

            if val_iou.mean() > best_val_iou:
                best_val_iou = val_iou.mean()
                model_save_path = os.path.join(cfg['TRAIN']['model_dir'], f'model02_bestiou_{best_val_iou}.pth')
                torch.save(model.state_dict(), model_save_path)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                model_save_path = os.path.join(cfg['TRAIN']['model_dir'], f'model02_bestdice_{best_val_dice:.4f}.pth')
                torch.save(model.state_dict(), model_save_path)

            if epoch == cfg['TRAIN']['epochs'] - 1:
                # Write validation results to log file
                with open(os.path.join(cfg['TRAIN']['log_dir'], 'retina_02_val_log.txt'), 'a') as val_log_file:
                    val_log_file.write(f"  Accuracy: {best_val_accuracy:.4f}, IoU: {best_val_iou:.4f}, Dice: {best_val_dice:.4f}\n")

        # Record best IoU at the end of the log
        log_file.write(f"Best IoU achieved during training: {best_val_iou:.4f}\n")



# Modified validation function
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
    with open(os.path.join(cfg['TRAIN']['log_dir'], 'retina_02_val_log.txt'), 'a') as val_log_file:
        val_log_file.write(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, IoU: {val_iou.mean():.4f}, Dice: {val_dice:.4f}\n")

    return val_loss, val_accuracy, val_iou, val_dice


# Visualization and save results
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
        'model_dir': config.XIAORONG_MODEL_FULL,
        'log_dir': config.XIAORONG_LOG_FULL,
        'result_dir': config.XIAORONG_RESULT_FULL,
        'mid_dir': config.MID_DIR

    },
    'DATASET':{
        'num_class': 2
    }
}

# Usage example
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
train_dataset = MedicalImageDataset(img_dir=config.RETINA_TRAIN_IMG, mask_dir=config.RETINA_TRAIN_LABEL, img_size=512)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = MedicalImageDataset(img_dir=config.RETINA_TEST_IMG, mask_dir=config.RETINA_TEST_LABEL, img_size=512)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
upblocks=[0, 0, 0, 0]
downblocks=[1, 1, 0, 0]
model = Classifier(2,upblocks,downblocks)  # Assumed segmentation model
model.to(device)

# Start training
train_model(model, train_loader, val_loader, device, cfg)
