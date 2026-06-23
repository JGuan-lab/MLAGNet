import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.optim as optim
from PIL import Image
from UnetDiscriminator import UNetDiscriminatorSN
from Gen import Generator
from tqdm import tqdm
from torchvision.transforms import ToTensor
from waveloss import Gloss,DLoss
import random
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import config

class PreprocessDataset(Dataset):
    """Preprocessing dataset class."""

    def __init__(self, gtPath, lrPath, gt_patch_size, scale):
        """Initialize the preprocessing dataset."""

        self.gt_imgs = []
        self.lr_imgs = []
        self.gt_patch_size = gt_patch_size
        self.scale = scale

        # collect all GT image paths
        for root, _, files in os.walk(gtPath):
            for file in tqdm(files):
                self.gt_imgs.append(os.path.join(root, file))

        # collect all LR image paths
        for root, _, files in os.walk(lrPath):
            for file in tqdm(files):
                self.lr_imgs.append(os.path.join(root, file))

    def __len__(self):
        """Return dataset length."""
        return min(len(self.gt_imgs), len(self.lr_imgs))  # use the shorter of the GT and LR lists
    def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
        """Randomly crop a paired LQ/GT patch."""

        h_lq,w_lq= img_lqs.size

        lq_patch_size = gt_patch_size // scale

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
        bottom = top+lq_patch_size
        right=left+lq_patch_size
        img_lqs = img_lqs.crop((left, top, right, bottom))

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        right_gt=left_gt+gt_patch_size
        bottom_gt=top_gt+gt_patch_size
        img_gts=img_gts.crop((left_gt, top_gt, right_gt, bottom_gt))
        return img_gts, img_lqs
    def __getitem__(self, index):
        """Return a (LR, GT) image pair."""
        gt_tempImg = self.gt_imgs[index]  # GT image path for this index
        lr_tempImg = self.lr_imgs[index]  # LR image path for this index
        paired_random_crop=self.paired_random_crop
        try:
            gt_tempImg = Image.open(gt_tempImg)
            lr_tempImg = Image.open(lr_tempImg)
            img_gts, img_lqs = paired_random_crop(gt_tempImg, lr_tempImg, self.gt_patch_size, self.scale)  # crop the pair
            sourceImg = ToTensor()(img_gts)  # convert cropped GT to tensor
            cropImg = ToTensor()(img_lqs)  # convert cropped LR to tensor
            return cropImg, sourceImg  # return (LR, GT)
        except Exception as e:
            # skip this sample on load failure
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
# class PreprocessDataset(Dataset):
#     """Preprocessing dataset class."""

#     def __init__(self, imgPath, transforms, ex=1):
#         """Initialize the preprocessing dataset."""
#         self.transforms = transforms
#         self.imgs = []

#         for idx, (_, _, files) in enumerate(os.walk(imgPath)):
#             self.imgs = []
#             for file in tqdm(files):
#                 self.imgs.append(imgPath + file)

#         np.random.shuffle(self.imgs)  # shuffle randomly

#     def __len__(self):
#         """Return dataset length."""
#         print(len(self.imgs))
#         return len(self.imgs)

#     def __getitem__(self, index):
#         """Return a (LR, GT) pair."""
#         tempImg = self.imgs[index]
#         try:
#             tempImg = Image.open(tempImg)
#             sourceImg = self.transforms(tempImg)  # apply transforms to the original image
#             cropImg = torch.nn.MaxPool2d(4, stride=4)(sourceImg)  # change to (2, 2) for 2x downscale
#             return cropImg, sourceImg
#         except Exception as e:
#             # skip this sample on load failure
#             print(f"Error loading image at index {index}: {str(e)}")
#             print(tempImg)
#             return None

# def MyDataloader(path,transform,BATCH):
#     processDataset = PreprocessDataset(imgPath=path,transforms=transform)
#     cleaned_dataset = [sample for sample in processDataset if sample is not None]
#     print(len(cleaned_dataset))
#     trainData = DataLoader(cleaned_dataset, batch_size=BATCH)
#     dataiter = iter(trainData)
#     testImgs, _ = next(dataiter)
#     testImgs = testImgs.to(device)
#     return trainData,testImgs

# def MyDataloader(path, transform, BATCH):
#     processDataset = PreprocessDataset(imgPath=path, transforms=transform)
#     cleaned_dataset = [sample for sample in processDataset if sample is not None]
#     print(len(cleaned_dataset))
#     trainData = DataLoader(cleaned_dataset, batch_size=BATCH, drop_last=True)  # drop_last=True discards the last incomplete batch
#     dataiter = iter(trainData)
#     testImgs, _ = next(dataiter)
#     testImgs = testImgs.to(device)
#     return trainData, testImgs
def MyTrainer(trainData,testImgs,optimizerG,optimizerD,netG,netD,EPOCHS,device,pathG,pathD):
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCHS):
        netG.to(device)
        netD.to(device)
        netG.train()
        netD.train()
        processBar = tqdm(enumerate(trainData, 1))

        for i, (cropImg, sourceImg) in processBar:
            # network iteration
            cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)
            fakeImg=netG(cropImg).to(device)
            netD.zero_grad()
            realOut=netD(sourceImg).mean
            fakeOut=netD(fakeImg).mean
            dloss=DLoss(fakeImg,sourceImg,device)
            dloss.backward(retain_graph=True)
            optimizerD.step()
            netG.zero_grad()
            gloss=Gloss(fakeImg,sourceImg,fakeOut,realOut,device)
            gloss.backward()
            optimizerG.step()
            # progress bar update
            processBar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f ' % (
                epoch, EPOCHS, dloss.item(), gloss.item()))

        # save visualization to output directory
        with torch.no_grad():
            fig = plt.figure(figsize=(10, 10))
            plt.axis("off")
            fakeImgs = netG(testImgs).detach().cpu()
            plt.imshow(np.transpose(vutils.make_grid(fakeImgs, padding=2, normalize=True), (1, 2, 0)), animated=True)
            plt.savefig(os.path.join(config.SR_VIS_OUT_TRAIN, 'Result_epoch % 05d.jpg' % epoch),
                        bbox_inches='tight', pad_inches=0)
            print('[INFO] Image saved successfully!')
            plt.close()
        if (epoch % 100 == 0):
            # save model checkpoint
            torch.save(netG.state_dict(), pathG +'netG_epoch_%d_%d.pth' % (4, epoch))
            torch.save(netD.state_dict(), pathD + 'netG_epoch_%d_%d.pth' % (4, epoch))
            print("save model successfully")
if(__name__ =='__main__'):

    transform = transforms.Compose([transforms.RandomCrop(128),
                                transforms.ToTensor()])
    Datapath = config.SR_TRAIN_DIV2K  # input dataset path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    BATCH = 32
    EPOCHS = 501
    # load dataset
    trainData, testImgs = MyDataloader(Datapath, transform,BATCH)
    # build models
    netG = Generator(3,64,2)
    netD = UNetDiscriminatorSN()
    # print model parameter counts
    total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f"Number of parameters in netG: {total_params}")
    total_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    print(f"Number of parameters in netD: {total_params}")
    # build optimizers
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    # model save paths
    pathG=config.SR_MODEL_G_OLD
    pathD=config.SR_MODEL_D_OLD
    # start training
    MyTrainer(trainData,testImgs,optimizerG,optimizerD,netG,netD,EPOCHS,device,pathG,pathD)