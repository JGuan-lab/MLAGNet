
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models
from torchvision.models import ViT_B_16_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import config

TASK_CONFIG = {
    "hist": {
        "val_dir": config.HIST_VAL,
        "model_path": config.VIT_HIST_MODEL,
        "out_dir": config.VIT_OUT_HIST,
        "class_name_map": {"0": "Benign", "1": "Malignant"},
    },
    "mhist": {
        "val_dir": config.MHIST_VAL,
        "model_path": config.VIT_MHIST_MODEL,
        "out_dir": config.VIT_OUT_MHIST,
        "class_name_map": {"0": "HP", "1": "SSA"},
    },
    "xray": {
        "val_dir": config.XRAY_VAL,
        "model_path": config.VIT_XRAY_MODEL,
        "out_dir": config.VIT_OUT_XRAY,
        "class_name_map": {"0": "Bacterial", "1": "Normal", "2": "Viral"},
    },
    "isic": {
        "val_dir": config.ISIC_VAL,
        "model_path": config.VIT_ISIC_MODEL,
        "out_dir": config.VIT_OUT_ISIC,
        "class_name_map": None,
    },
}

def resolve_display_names(raw_classes, class_name_map=None):
    if class_name_map is None:
        return raw_classes
    return [class_name_map.get(c, c) for c in raw_classes]

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_score)

def main(task, device_str):
    cfg = TASK_CONFIG[task]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    val_dataset = datasets.ImageFolder(root=cfg["val_dir"], transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    raw_class_names = val_dataset.classes
    display_class_names = resolve_display_names(raw_class_names, cfg["class_name_map"])

    print("raw_class_names:", raw_class_names)
    print("display_class_names:", display_class_names)
    print("class_to_idx:", val_dataset.class_to_idx)

    num_classes = len(raw_class_names)
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, num_classes)
    model = model.to(device)

    state_dict = torch.load(cfg["model_path"], map_location=device)
    model.load_state_dict(state_dict)

    y_true, y_pred, y_score = evaluate(model, val_loader, device)

    report = classification_report(y_true, y_pred, target_names=display_class_names, digits=4, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = (y_true == y_pred).mean()

    report_path = os.path.join(cfg["out_dir"], "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
        f.write(report)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format="d", ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(cfg["out_dir"], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    if num_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown", "pink"])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(display_class_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(cfg["out_dir"], "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved ROC curve to: {roc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()
    main(args.task, args.device)
