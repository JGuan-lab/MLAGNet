import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.optim as optim
from PIL import Image
from Gen import Generator
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from accelerate import Accelerator,DeepSpeedPlugin
import random
from torchvision.transforms import ToTensor
import config

# class PreprocessDataset(Dataset):
#     """Preprocessing dataset class"""

#     def __init__(self, gtPath, lrPath, gt_patch_size, scale):
#         """Initialize preprocessing dataset class"""

#         self.gt_imgs = []
#         self.lr_imgs = []
#         self.gt_patch_size = gt_patch_size
#         self.scale = scale

#         # Collect all GT image paths
#         for root, _, files in os.walk(gtPath):
#             for file in tqdm(files):
#                 self.gt_imgs.append(os.path.join(root, file))

#         # Collect all LR image paths
#         for root, _, files in os.walk(lrPath):
#             for file in tqdm(files):
#                 self.lr_imgs.append(os.path.join(root, file))
#     def paired_random_crop(self,img_gts, img_lqs, gt_patch_size, scale):
#         """Crop the original images"""

#         h_lq,w_lq= img_lqs.size

#         lq_patch_size = gt_patch_size // scale

#         # randomly choose top and left coordinates for lq patch
#         top = random.randint(0, h_lq - lq_patch_size)
#         left = random.randint(0, w_lq - lq_patch_size)
#         bottom = top+lq_patch_size
#         right=  left+lq_patch_size
#         img_lqs = img_lqs.crop((left, top, right, bottom))

#         # crop corresponding gt patch
#         top_gt, left_gt = int(top * scale), int(left * scale)
#         right_gt=left_gt+gt_patch_size
#         bottom_gt=top_gt+gt_patch_size
#         img_gts=img_gts.crop((left_gt, top_gt, right_gt, bottom_gt))
#         return img_gts, img_lqs
#     def __len__(self):
#         """Get dataset length"""
#         return min(len(self.gt_imgs), len(self.lr_imgs))  # Return the shorter length of GT and LR image path lists

#     def __getitem__(self, index):
#         """Get a data sample"""
#         gt_tempImg = self.gt_imgs[index]  # Get GT image path at the given index
#         lr_tempImg = self.lr_imgs[index]  # Get LR image path at the given index
#         gt=gt_tempImg.split('/')[-1]
#         lr=lr_tempImg.split('/')[-1]
#         try:
#             gt_tempImg = Image.open(gt_tempImg)
#             lr_tempImg = Image.open(lr_tempImg)
#               # Check whether loading failed
#             if gt != lr:
#                 raise Exception("Loaded images do not match")
#             img_gts, img_lqs = self.paired_random_crop(gt_tempImg, lr_tempImg, self.gt_patch_size, self.scale)  # Crop the original images
#             sourceImg = ToTensor()(img_gts)  # Process the cropped GT image
#             cropImg = ToTensor()(img_lqs)  # Process the cropped LR image
#             return cropImg, sourceImg  # Return LR image and GT image
#         except Exception as e:
#             # Loading failed; skip this sample
#             print(f"Error loading image at index {index}: {str(e)}")
#             return None
class PreprocessDataset(Dataset):
    """Preprocessing dataset class"""

    def __init__(self, gtPath, lrPath, gt_patch_size, scale):
        """Initialize preprocessing dataset class"""

        self.gt_imgs = []
        self.lr_imgs = []
        self.gt_patch_size = gt_patch_size
        self.scale = scale

        # Collect all GT image paths
        for root, _, files in os.walk(gtPath):
            for file in files:
                self.gt_imgs.append(os.path.join(root, file))

        # Collect all LR image paths
        for root, _, files in os.walk(lrPath):
            for file in files:
                self.lr_imgs.append(os.path.join(root, file))

        # Sort GT and LR image file paths
        self.gt_imgs.sort()
        self.lr_imgs.sort()

        # Check whether GT and LR image filenames match
        for gt_img, lr_img in zip(self.gt_imgs, self.lr_imgs):
            gt_filename = os.path.basename(gt_img)
            lr_filename = os.path.basename(lr_img)

            print(f"GT: {gt_filename}, LR: {lr_filename}")  # Print filenames

            if gt_filename != lr_filename:
                print(f"Warning: Filename mismatch -> GT: {gt_filename}, LR: {lr_filename}")

    def paired_random_crop(self, img_gts, img_lqs, gt_patch_size, scale):
        """Crop the original images"""
        h_lq, w_lq = img_lqs.size
        lq_patch_size = gt_patch_size // scale
        # Randomly choose crop position
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
        bottom = top + lq_patch_size
        right = left + lq_patch_size
        img_lqs = img_lqs.crop((left, top, right, bottom))
        # Crop the corresponding GT image
        top_gt, left_gt = int(top * scale), int(left * scale)
        right_gt = left_gt + gt_patch_size
        bottom_gt = top_gt + gt_patch_size
        img_gts = img_gts.crop((left_gt, top_gt, right_gt, bottom_gt))
        return img_gts, img_lqs

    def __len__(self):
        """Get dataset length"""
        return min(len(self.gt_imgs), len(self.lr_imgs))  # Return the shorter length of GT and LR image path lists

    def __getitem__(self, index):
        """Get a data sample"""
        gt_tempImg = self.gt_imgs[index]  # Get GT image path at the given index
        lr_tempImg = self.lr_imgs[index]  # Get LR image path at the given index
        gt_filename = os.path.basename(gt_tempImg)
        lr_filename = os.path.basename(lr_tempImg)

        try:
            if gt_filename != lr_filename:
                raise Exception(f"Filename mismatch -> GT: {gt_filename}, LR: {lr_filename}")

            gt_tempImg = Image.open(gt_tempImg)
            lr_tempImg = Image.open(lr_tempImg)
            img_gts, img_lqs = self.paired_random_crop(gt_tempImg, lr_tempImg, self.gt_patch_size, self.scale)  # Crop the original images
            sourceImg = ToTensor()(img_gts)  # Process the cropped GT image
            cropImg = ToTensor()(img_lqs)  # Process the cropped LR image
            return cropImg, sourceImg  # Return LR image and GT image

        except Exception as e:
            # Loading failed or image mismatch; skip this sample
            print(f"Error loading image at index {index}: {str(e)}")
            return None  # Return None so this sample is skipped in subsequent processing


def MyDataloader(gtPath, lrPath, gt_patch_size, scale, BATCH):
    processDataset = PreprocessDataset(gtPath, lrPath, gt_patch_size, scale)
    cleaned_dataset = [sample for sample in processDataset if sample is not None]
    print("Total number of samples after cleaning:", len(cleaned_dataset))
    trainData = DataLoader(cleaned_dataset, batch_size=BATCH)
    dataiter = iter(trainData)
    testImgs, _ = next(dataiter)
    return trainData, testImgs
lossF=nn.L1Loss()
def MyTrainer(trainData,testImgs,optimizerG,netG,EPOCHS,device,pathG):
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCHS):
        netG.to(device)
        netG.train()
        processBar = tqdm(enumerate(trainData, 1))

        for i, (cropImg, sourceImg) in processBar:
            # Iterate network
            cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)
            fakeImg=netG(cropImg).to(device)

            netG.zero_grad()
            gloss=lossF(fakeImg,sourceImg)
            accelerator.backward(gloss)
            #gloss.backward()
            optimizerG.step()
            # Visualize progress
            processBar.set_description(desc='[%d/%d] Loss_G: %.4f ' % (
                epoch, EPOCHS, gloss.item()))

        # Output files to directory
        with torch.no_grad():
            fig = plt.figure(figsize=(10, 10))
            plt.axis("off")
            fakeImgs = netG(testImgs).detach().cpu()
            plt.imshow(np.transpose(vutils.make_grid(fakeImgs, padding=2, normalize=True), (1, 2, 0)), animated=True)
            plt.savefig(os.path.join(config.SR_VIS_OUT_TG, 'Result_epoch % 05d.jpg' % epoch),
                        bbox_inches='tight', pad_inches=0)
            print('[INFO] Image saved successfully!')
            plt.close()
        if (epoch % 5 == 0):
            # Save model checkpoint
            #torch.save(netG.state_dict(), pathG +'netG_epoch_%d_%d.pth' % (4, epoch))
            #accelerator.wait_for_everyone()
            if accelerator.is_main_process: 
                unwrapped_modelG = accelerator.unwrap_model(netG)
                torch.save(unwrapped_modelG.state_dict(), os.path.join(config.SR_MODEL_G_L1_8, 'netGl1loss_epoch_%d_%d.pth' % (8, epoch)))
            print("save model successfully")
if(__name__ =='__main__'):

    transform = transforms.Compose(transforms.ToTensor())
    gt_path = config.SR_TRAIN_GT_TG
    lq_path = config.SR_TRAIN_LR_LUNG  # Input dataset path
#     lq_path = r"/home/yons/data/Weiyubing/lungmid" # Input dataset path
    # lq_path = '/home/yons/data/Weiyubing/qingtcrop2'  # Low-resolution image folder
    # gt_path = '/home/yons/data/Weiyubing/qingtcroptrue'
    # lq_path = '/home/yons/data/Weiyubing/qingtcroplow8'
    accelerator = Accelerator()
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1)
    device= accelerator.device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    BATCH = 32
    EPOCHS = 2501
    # Load dataset
    trainData, testImgs = MyDataloader(gt_path, lq_path, 128, 8, BATCH)
    # Build model
    netG = Generator(3,64,3) # 1 means 2x upscaling
#    # Load pretrained model
#     pretrained_dict = torch.load("/home/Yb/uureal/model/l1model/netGl1loss_epoch_2_1880.pth")
#     # Get current state dict of generator network
#     model_dict = netG.state_dict()
#     # Remove mismatched keys from current state dict (e.g., output layer)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # Update current state dict
#     model_dict.update(pretrained_dict)
#     # Load updated state dict into generator network
#     netG.load_state_dict(model_dict)
    # Print model parameters:
    total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f"Number of parameters in netG: {total_params}")
    # Build optimizer
    optimizerG = optim.Adam(netG.parameters())

    # Model save path
    pathG=config.SR_MODEL_G_TG
    netG, optimizer, trainDatadata = accelerator.prepare(netG, optimizerG, trainData)
    # Training function
    MyTrainer(trainData,testImgs,optimizerG,netG,EPOCHS,device,pathG)