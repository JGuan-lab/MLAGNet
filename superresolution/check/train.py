import os
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score
from Gene import Decoder  # Assumes you have a Decoder model defined
import torch.nn as nn
import numpy as np

# Configuration parameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = config.CHECK_OUTPUT_DIR

# Training dataset paths
TRAIN_IMAGE_DIR = config.CHECK_TRAIN_IMG_DIR
TRAIN_LABEL_DIR = config.CHECK_TRAIN_LABEL_DIR
ANNOTATED_IMAGE_DIR = config.CHECK_ANNOTATED_DIR

# Create directories for saving models and logs
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "train_log.txt")

# Data preprocessing: basic image transforms
transform = T.Compose([
    T.ToPILImage(),  # Convert to PIL image format
    T.Resize((224, 224)),  # Resize image to 224x224 (adjust as needed)
    T.ToTensor(),  # Convert to Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using pretrained model mean and std
])

# Custom dataset: reads images and annotations
class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))  # Assumes images are .jpg and labels are .txt

        image = plt.imread(img_path)  # Read image
        with open(label_path, 'r') as f:
            boxes = []
            labels = []
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                boxes.append([x_center, y_center, width, height])
                labels.append(label)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        sample = {'image': image, 'boxes': boxes, 'labels': labels}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample

def collate_fn(batch):
    images = []
    targets = []
    for b in batch:
        images.append(b['image'])
        targets.append({'boxes': b['boxes'], 'labels': b['labels']})
    return torch.stack(images, 0), targets

train_dataset = YoloDataset(
    image_dir=TRAIN_IMAGE_DIR,
    label_dir=TRAIN_LABEL_DIR,
    transform=transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn  # Use the custom YOLO collate function
)

# Load the model
model = Decoder().to(DEVICE)

def decode_predictions(pred, num_classes, grid_size):
    """
    Decode the predictions to get bounding boxes.
    
    pred: Tensor of shape [batch_size, num_anchors * (5 + num_classes), grid_size, grid_size]
    num_classes: Number of object classes
    grid_size: Size of the grid (height and width)

    Returns:
    decoded_boxes: List of tensors, each containing the bounding boxes for each image in the batch
    """
    batch_size = pred.size(0)
    num_pred = 5 + num_classes  # 5 for x, y, w, h, confidence, and num_classes for class scores
    num_anchors = pred.size(1) // (num_pred)  # e.g., 3 anchors for each cell in YOLO

    decoded_boxes = []

    for b in range(batch_size):
        pred_boxes = pred[b]  # [num_anchors * (5 + num_classes), grid_size, grid_size]
        pred_boxes = pred_boxes.view(num_anchors, num_pred, grid_size, grid_size)  # [num_anchors, 5 + num_classes, grid_size, grid_size]

        # Extract box predictions (x, y, width, height)
        box_preds = pred_boxes[:, :4, :, :]  # x_center, y_center, width, height
        conf_preds = pred_boxes[:, 4, :, :]  # confidence score
        class_preds = pred_boxes[:, 5:, :, :]  # class probabilities

        # Apply sigmoid to x_center, y_center, and confidence
        x_center = torch.sigmoid(box_preds[:, 0])  # Sigmoid for x center
        y_center = torch.sigmoid(box_preds[:, 1])  # Sigmoid for y center
        width = torch.exp(box_preds[:, 2])  # Directly use the predicted width
        height = torch.exp(box_preds[:, 3])  # Directly use the predicted height
        confidence = torch.sigmoid(conf_preds)  # Confidence score for the box
        class_probs = torch.sigmoid(class_preds)  # Class probabilities for each anchor

        # Convert (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # Stack boxes into a single tensor for the decoded boxes
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        decoded_boxes.append(boxes)

    return decoded_boxes




# Assumes YoloLoss has already been implemented
class YoloLoss(nn.Module):
    def __init__(self, num_classes=2, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets, grid_size):
        """
        predictions: (batch_size, num_anchors * (5 + num_classes), grid_size, grid_size)
        targets: list of dictionaries with 'boxes' and 'labels'
        grid_size: size of the grid
        """
        print(predictions.shape)

        decoded_boxes = decode_predictions(predictions, self.num_classes, grid_size)
        loss = 0.0
        
        for b in range(predictions.size(0)):  # Iterate over the batch
            target = targets[b]
            true_boxes = target['boxes']
            true_labels = target['labels']
            
            # Calculate coordinate loss
            coord_loss = self.compute_coord_loss(decoded_boxes[b], true_boxes)
            
            # Calculate class loss
            class_loss = self.compute_class_loss(decoded_boxes[b], true_labels)
            
            # Calculate confidence loss
            conf_loss = self.compute_confidence_loss(decoded_boxes[b], true_boxes)
            
            loss += coord_loss + class_loss + conf_loss
        
        return loss


    def compute_coord_loss(self, decoded_boxes, true_boxes):
        # Compute coordinate loss (simple L2 loss)
        coord_loss = torch.sum((decoded_boxes - true_boxes) ** 2)
        return coord_loss

    def compute_class_loss(self, decoded_boxes, true_labels):
        # Compute class loss, usually using cross entropy loss
        # Here we assume each box has a label, and we're simplifying the calculation
        class_loss = torch.sum(torch.nn.functional.cross_entropy(decoded_boxes, true_labels))
        return class_loss

    def compute_confidence_loss(self, decoded_boxes, true_boxes):
        # Compute confidence loss (simple L2 loss)
        conf_loss = torch.sum((decoded_boxes[:, 4] - true_boxes[:, 4]) ** 2)
        return conf_loss


def train(model, criterion, train_dataloader, optimizer, DEVICE, epoch, grid_size):
    model.train()
    total_loss = 0.0
    all_pred_boxes = []
    all_true_boxes = []
    all_preds = []
    all_labels = []

    for images, targets in train_dataloader:
        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)  # Decoder outputs the predictions

        # Compute loss using YoloLoss, passing anchors and grid_size
        losses = criterion(predictions, targets, grid_size)

        # Backward pass
        losses.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += losses.item()

        # Collect predicted and ground-truth boxes
        for target, pred in zip(targets, predictions):
            all_true_boxes.append(target['boxes'])
            all_pred_boxes.append(pred)

            # Compute accuracy, recall, etc. here
            all_preds.append(pred.argmax(1))
            all_labels.append(target['labels'])

    # Compute and log metrics
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Write to log file
    with open(LOG_FILE, 'a') as f:
        f.write(f'Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_dataloader):.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n')

    return total_loss / len(train_dataloader)
# Set anchors and grid_size
grid_size = 14  # Fixed at 14x14

# Training loop
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = YoloLoss()

for epoch in range(EPOCHS):
    train_loss = train(model, criterion, train_dataloader, optimizer, DEVICE, epoch,  grid_size)
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}')
