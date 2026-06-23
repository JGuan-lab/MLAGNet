import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
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
        #self.second=GDFN(in_dim,ratio=2)
        self.second = BaseConvBlock(in_dim)
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

# Dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.img_list = os.listdir(img_dir)

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = transforms.ToTensor()(mask).long().squeeze(0)

        base_name = os.path.splitext(img_name)[0]
        return image, mask, base_name

# Multi-metric computation function
def compute_metrics(pred, target, num_classes):
    pred = pred.flatten()
    target = target.flatten()

    metrics = {}
    eps = 1e-6

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        TP = (pred_cls & target_cls).sum().item()
        FP = (pred_cls & ~target_cls).sum().item()
        FN = (~pred_cls & target_cls).sum().item()
        TN = (~pred_cls & ~target_cls).sum().item()

        iou = TP / (TP + FP + FN + eps)
        dice = 2 * TP / (2 * TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        specificity = TN / (TN + FP + eps)
        accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        metrics[f'class_{cls}'] = {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1': f1
        }

    return metrics

# Inference and logging
def infer_model(model, data_loader, device, cfg):
    model.eval()
    results = []
    with torch.no_grad():
        for i, (images, masks, names) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for b in range(images.size(0)):
                name = names[b]
                metrics = compute_metrics(preds[b].cpu(), masks[b].cpu(), cfg['DATASET']['num_class'])
                mean_metrics = {
                    key: np.mean([v[key] for v in metrics.values()]) for key in ['iou', 'dice', 'precision', 'recall', 'specificity', 'accuracy', 'f1']
                }
                # Compute composite score
                score = 0.5*mean_metrics['accuracy']+0.3 * mean_metrics['iou'] + 0.2 * mean_metrics['dice']
                mean_metrics['score'] = score

                results.append({
                    'image': images[b].cpu(),
                    'mask': masks[b].cpu(),
                    'pred': preds[b].cpu(),
                    'metrics': mean_metrics,
                    'name': name
                })

    # Select top-5 images
    top_results = sorted(results, key=lambda x: x['metrics']['score'], reverse=True)[:5]

    for result in top_results:
        visualize_and_save_result(
            result['image'], result['mask'], result['pred'],
            result['name'], result['metrics'],
            cfg['TRAIN']['result_dir']
        )
        with open(os.path.join(cfg['TRAIN']['log_dir'], 'test_log.txt'), 'a') as log_file:
            m = result['metrics']
            log_file.write(
                f"Image: {result['name']}, "
                f"Score: {m['score']:.4f}, "
                f"Accuracy: {m['accuracy']:.4f}, "
                f"IoU: {m['iou']:.4f}, "
                f"Dice: {m['dice']:.4f}, "
                f"Precision: {m['precision']:.4f}, "
                f"Recall: {m['recall']:.4f}, "
                f"F1: {m['f1']:.4f}, "
                f"Specificity: {m['specificity']:.4f}\n"
            )

# Visualization and save function
def visualize_and_save_result(image, mask, pred, name, metrics, result_dir):
    # Convert tensors to numpy arrays
    image_np = image.numpy().transpose(1, 2, 0)
    mask_np = mask.numpy()
    pred_np = pred.numpy()

    os.makedirs(result_dir, exist_ok=True)

    # ---- 1. Save composite image with 3 subplots ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image_np)
    ax[0].set_title('Image')
    ax[1].imshow(mask_np, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[2].imshow(pred_np, cmap='gray')
    ax[2].set_title('Prediction')

    # Add metrics title
    plt.suptitle(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, "
                 f"Acc: {metrics['accuracy']:.4f}, Score: {metrics['score']:.4f}")

    # Save the composite figure
    composite_path = os.path.join(result_dir, f'test_result_{name}.png')
    plt.savefig(composite_path)
    plt.close()

    # ---- 2. Save single prediction image ----
    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))  # assuming pred is in [0,1]
    pred_path = os.path.join(result_dir, f'pred_only_{name}.png')
    pred_img.save(pred_path)

# Usage example
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
upblocks=[0, 0, 0, 0]
downblocks=[1, 1, 0, 0]
model = Classifier(2,upblocks,downblocks)  # Assumed segmentation model
model.load_state_dict(torch.load(config.XIAORONG_NOGDFN_BEST_MODEL))  # Replace with your model path
model.to(device)

# Load test dataset
test_dataset = MedicalImageDataset(img_dir=config.COMMODEL_RETINAL_IMG,
                                    mask_dir=config.COMMODEL_RETINAL_LABEL,
                                    img_size=512)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

cfg = {
    'TRAIN': {
        'result_dir': config.XIAORONG_RESULT_NOGDFN,
        'log_dir': config.XIAORONG_LOG_NOGDFN
    },
    'DATASET':{
        'num_class': 2
    }
}

# Run inference
infer_model(model, test_loader, device, cfg)
