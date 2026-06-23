import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import cv2
import config
class PreprocessDataset(Dataset):
    """Preprocessing dataset class."""

    def __init__(self, gtPath, lrPath, gt_patch_size, scale):
        """Initialize the preprocessing dataset class."""

        self.gt_imgs = []
        self.lr_imgs = []
        self.gt_patch_size = gt_patch_size
        self.scale = scale

        # Collect paths of all GT images.
        for root, _, files in os.walk(gtPath):
            for file in tqdm(files):
                self.gt_imgs.append(os.path.join(root, file))

        # Collect paths of all LR images.
        for root, _, files in os.walk(lrPath):
            for file in tqdm(files):
                self.lr_imgs.append(os.path.join(root, file))
    def paired_random_crop(self,img_gts, img_lqs, gt_patch_size, scale):
        """Randomly crop the original image pair."""

        h_lq,w_lq= img_lqs.size

        lq_patch_size = gt_patch_size // scale

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
        bottom = top+lq_patch_size
        right=  left+lq_patch_size
        img_lqs = img_lqs.crop((left, top, right, bottom))

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        right_gt=left_gt+gt_patch_size
        bottom_gt=top_gt+gt_patch_size
        img_gts=img_gts.crop((left_gt, top_gt, right_gt, bottom_gt))
        return img_gts, img_lqs
    def __len__(self):
        """Return the dataset length."""
        return min(len(self.gt_imgs), len(self.lr_imgs))  # Return the shorter length among GT and LR image lists.

    def __getitem__(self, index):
        """Retrieve a data sample."""
        gt_tempImg = self.gt_imgs[index]  # Get the GT image path at the given index.
        lr_tempImg = self.lr_imgs[index]  # Get the LR image path at the given index.

        try:
            gt_tempImg = Image.open(gt_tempImg)
            lr_tempImg = Image.open(lr_tempImg)
            img_gts, img_lqs = self.paired_random_crop(gt_tempImg, lr_tempImg, self.gt_patch_size, self.scale)  # Randomly crop the image pair.
            sourceImg = ToTensor()(img_gts)  # Convert the cropped GT image to a tensor.
            cropImg = ToTensor()(img_lqs)  # Convert the cropped LR image to a tensor.
            return cropImg, sourceImg  # Return the LR image and the GT image.
        except Exception as e:
            # Failed to load; skip this sample.
            print(f"Error loading image at index {index}: {str(e)}")
            return None

def MyDataloader(gtPath, lrPath, gt_patch_size, scale, BATCH):
    processDataset = PreprocessDataset(gtPath, lrPath, gt_patch_size, scale)
    cleaned_dataset = [sample for sample in processDataset if sample is not None]
    print("Total number of samples after cleaning:", len(cleaned_dataset))
    trainData = DataLoader(cleaned_dataset, batch_size=BATCH)
    dataiter = iter(trainData)
    testImgs, _ = next(dataiter)
    return trainData, testImgs

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    gt_path = config.SR_PROC_GT
    lq_path = config.SR_PROC_LQ
    MyDataloader(gt_path, lq_path, 256, 4, 4)