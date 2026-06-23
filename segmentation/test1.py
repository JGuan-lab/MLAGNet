import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from model import Classifier

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
                score = 0.4 * mean_metrics['iou'] + 0.3 * mean_metrics['dice'] + 0.3 * mean_metrics['f1']
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
    image = image.numpy().transpose(1, 2, 0)
    mask = mask.numpy()
    pred = pred.numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Prediction')

    plt.suptitle(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, Acc: {metrics['accuracy']:.4f}")
    plt.savefig(os.path.join(result_dir, f'inference_result_{name}.png'))
    plt.close()

# Initialize and run inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier(2)
model.load_state_dict(torch.load(config.CRACK_BEST_MODEL))
model.to(device)

test_dataset = MedicalImageDataset(
    img_dir=config.CRACK_TEST_IMG,
    mask_dir=config.CRACK_TEST_LABEL,
    img_size=512
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

cfg = {
    'TRAIN': {
        'result_dir': config.COMMODEL_CRACK_OURS,
        'log_dir': config.COMMODEL_CRACK_LOGS
    },
    'DATASET': {
        'num_class': 2
    }
}

infer_model(model, test_loader, device, cfg)
