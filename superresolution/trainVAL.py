import cv2
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
from waveloss import Gloss,DLoss
from metric import calculate
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
from torchvision.transforms import ToTensor
from accelerate import Accelerator
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
    def paired_random_crop(self,img_gts, img_lqs, gt_patch_size, scale):
        """Randomly crop a paired LQ/GT patch."""

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
        """Return dataset length."""
        return min(len(self.gt_imgs), len(self.lr_imgs))  # use the shorter of GT and LR lists

    def __getitem__(self, index):
        """Return a (LR, GT) image pair."""
        gt_tempImg = self.gt_imgs[index]  # GT image path for this index
        lr_tempImg = self.lr_imgs[index]  # LR image path for this index
        gt=gt_tempImg.split('/')[-1]
        lr=lr_tempImg.split('/')[-1]
        try:
            gt_tempImg = Image.open(gt_tempImg)
            lr_tempImg = Image.open(lr_tempImg)
              # check that both images loaded correctly
            if gt != lr:
                raise Exception("Loaded images do not match")
            img_gts, img_lqs = self.paired_random_crop(gt_tempImg, lr_tempImg, self.gt_patch_size, self.scale)  # crop the pair
            sourceImg = ToTensor()(img_gts)  # convert cropped GT to tensor
            cropImg = ToTensor()(img_lqs)  # convert cropped LR to tensor
            return cropImg, sourceImg  # return (LR, GT)
        except Exception as e:
            # skip this sample on load failure
            print(f"Error loading image at index {index}: {str(e)}")
            return None

def MyDataloader(gtPath, lrPath, gt_patch_size, scale, BATCH,device):
    processDataset = PreprocessDataset(gtPath, lrPath, gt_patch_size, scale)
    cleaned_dataset = [sample for sample in processDataset if sample is not None]
    print("Total number of samples after cleaning:", len(cleaned_dataset))
    trainData = DataLoader(cleaned_dataset, batch_size=BATCH)
    dataiter = iter(trainData)
    testImgs, _ = next(dataiter)
    testImgs = testImgs.to(device)
    return trainData, testImgs

# def calculate_val(model,lr_path,hr_path):
#     preTransform = transforms.Compose([transforms.ToTensor()])
#     pilImg = Image.open(lr_path)
#     img = preTransform(pilImg).unsqueeze(0).to(device)
#     source = model(img)[0, :, :, :]
#     source = source.cpu().detach().numpy()  # convert to numpy
#     source = source.transpose((1, 2, 0))  # reshape
#     source = np.clip(source, 0, 1)  # clamp pixel values

#     fakeimg = Image.fromarray(np.uint8(source * 255))
#     #fakeimg.save("/home/Yb/uureal/temp")
#     fakeimg.save("/home/Yb/uureal/temp/fake_img.png")
#     fake_img_png = plt.imread("/home/Yb/uureal/temp/fake_img.png").astype(np.float32) / 255.0
#     real_img_png = cv2.imread(hr_path)
#     psnr,ssim=calculate(fake_img_png,real_img_png)
#     return psnr,ssim
import os
import re

def extract_index(file_path):
    # extract filename from path
    filename = os.path.basename(file_path)
    # use regex to extract numeric part as index
    index_match = re.search(r'\d+', filename)
    if index_match:
        index = int(index_match.group())
        return index
    else:
        # return None if no number found
        return None

def calculate_val(model, lr_path, hr_path,epoch):
    preTransform = transforms.Compose([transforms.ToTensor()])
    pilImg = Image.open(lr_path).convert('RGB')  # ensure RGB format
    img = preTransform(pilImg).unsqueeze(0).to(device)
    source = model(img)[0, :, :, :]
    source = source.cpu().detach().numpy()  # convert to numpy
    source = source.transpose((1, 2, 0))  # reshape
    source = np.clip(source, 0, 1)  # clamp pixel values

    fakeimg = Image.fromarray(np.uint8(source * 255))
    
    # extract index from the file path
    index = extract_index(lr_path)
    index2=extract_index(hr_path)
    print(index)
    print(index2)
    if index is None:
        print("Could not extract a valid index from the file path.")
        return None, None

    # build the output image path
    output_path = os.path.join(config.SR_VIS_TEMP_DIR, f"fake_{epoch}_{index}.png")

    # save the image with proper error handling
    try:
        fakeimg.save(output_path)
    except Exception as e:
        print(f"Error saving image: {e}")
        return None, None

    # read image for evaluation with error handling
    try:
        fake_img_png = plt.imread(output_path).astype(np.float32) / 255.0
    except Exception as e:
        print(output_path)
        print(f"Error reading image: {e}")
        return None, None

    real_img_png = cv2.imread(hr_path).astype(np.float32) / 255.0

    if fake_img_png is not None and real_img_png is not None:

        psnr, ssim = calculate(fake_img_png, real_img_png)
        print(hr_path)
        print(psnr)
        return psnr, ssim
    else:
        print("Failed to read images for evaluation.")
        return None, None

def MyTrainer(trainData, val_hr_path,val_lr_path, testImgs, optimizerG, optimizerD, netG, netD, EPOCHS, device, pathG, pathD):
    netG, netD, optimizerG, optimizerD, trainData = accelerator.prepare(netG, netD, optimizerG, optimizerD, trainData)
    torch.autograd.set_detect_anomaly(True)
    best_psnr = 0
    

    for epoch in range(EPOCHS):
        netG.to(device)
        netD.to(device)
        netG.train()
        netD.train()
        processBar = tqdm(enumerate(trainData, 1))

        for i, (cropImg, sourceImg) in processBar:
            cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)
            fakeImg = netG(cropImg).to(device)
            netD.zero_grad()
        
            dloss = DLoss(fakeImg, sourceImg, device)
            accelerator.backward(dloss, retain_graph=True)
            optimizerD.step()
            netG.zero_grad()
            gloss = Gloss(fakeImg, sourceImg, device)
            accelerator.backward(gloss)
           # gloss.backward()
            optimizerG.step()
            processBar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f ' % (
                epoch, EPOCHS, dloss.item(), gloss.item()))
            # save visualization to output directory
        with torch.no_grad():
            fig = plt.figure(figsize=(10, 10))
            plt.axis("off")
            fakeImgs = netG(testImgs).detach().cpu()
            plt.imshow(np.transpose(vutils.make_grid(fakeImgs, padding=2, normalize=True), (1, 2, 0)),
                       animated=True)
            plt.savefig(os.path.join(config.SR_VIS_OUT_TRAINVAL, 'Result_epoch_% 05d.jpg' % epoch),
                        bbox_inches='tight', pad_inches=0)
            print('[INFO] Image saved successfully!')
            plt.close()

        # validation
        netG.eval()
        with torch.no_grad():
            # initialize PSNR and SSIM accumulators
            val_psnr = 0
            val_ssim = 0
            num_val_images = 0

            # list LR image files
            lr_image_files = os.listdir(val_lr_path)
            # list HR image files
            hr_image_files = os.listdir(val_hr_path)

            # iterate image pairs and compute PSNR/SSIM
            for lr_file, hr_file in zip(lr_image_files, hr_image_files):
                lr_img_path = os.path.join(val_lr_path, lr_file)
                hr_img_path = os.path.join(val_hr_path, hr_file)
                psnr, ssim = calculate_val(netG, lr_img_path, hr_img_path,epoch)
                if psnr is not None:
                    val_psnr += psnr
                    val_ssim += ssim
                    num_val_images += 1

            # compute average PSNR and SSIM
            if num_val_images > 0:
                val_psnr /= num_val_images
                val_ssim /= num_val_images

            # print results
            print(f"Epoch {epoch}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}, num_val_images: {num_val_images}")
                # val_psnr = 0

            # save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                if accelerator.is_main_process: 
                    unwrapped_modelG = accelerator.unwrap_model(netG)
                    model_path = os.path.join(pathbest, f'best_{best_psnr}_netG.pth')
                    torch.save(unwrapped_modelG.state_dict(), model_path)
                    #torch.save(netG.state_dict(), pathG + 'best_psnr_netG.pth')
                    print(f"Saved best PSNR model with PSNR: {best_psnr:.4f}")

        if (epoch % 5 == 0):
            # save current model
            # torch.save(netG.state_dict(), pathG + f'netG_epoch_{epoch}.pth')
            # torch.save(netD.state_dict(), pathD + f'netD_epoch_{epoch}.pth')
            accelerator.wait_for_everyone()
            if accelerator.is_main_process: 
                unwrapped_modelG = accelerator.unwrap_model(netG)
                torch.save(unwrapped_modelG.state_dict(), pathG + f'netG_epoch_{epoch}.pth')
                print( pathG + f'netG_epoch_{epoch}.pth')
                unwrapped_modelD = accelerator.unwrap_model(netD)
                torch.save(unwrapped_modelD.state_dict(),pathD + f'netD_epoch_{epoch}.pth')
                print(f"Saved model at epoch {epoch}")

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    gt_path = config.SR_TRAIN_GT_4X
    lq_path = config.SR_TRAIN_LR_4X  # input dataset path
    # gt_path = r"/home/yons/data/Weiyubing/gtsmall"
    # lq_path = r"/home/yons/data/Weiyubing/lqsmall"
    val_lr_path = config.SR_VAL_LR
    val_hr_path = config.SR_VAL_HR
    accelerator = Accelerator()
    device = accelerator.device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    BATCH = 32
    EPOCHS =2501 
    # load dataset
    trainData, testImgs = MyDataloader(gt_path, lq_path, 128, 4, BATCH,device)
    #valData = load_validation_data(val_lr_path, val_hr_path, transform, BATCH)
    # build generator
    netG = Generator(3, 64, 2)
    # load pretrained model
    # pretrained_dict = torch.load("/home/Yb/uureal/Ui/model/netGl1loss_epoch_4_20.pth")
    # # get current state dict
    # model_dict = netG.state_dict()
    # # filter out mismatched keys (e.g., output layer)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # update current state dict
    # model_dict.update(pretrained_dict)
    # # load updated state dict into generator
    # netG.load_state_dict(model_dict)

    # build discriminator
    netD = UNetDiscriminatorSN()
    # # load pretrained model
    # pretrained_dict = torch.load("/home/yons/data/Weiyubing/model/2full/netD_epoch_688.pth")
    # # get current state dict
    # model_dict = netD.state_dict()
    # # filter out mismatched keys (e.g., output layer)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # update current state dict
    # model_dict.update(pretrained_dict)
    # # load updated state dict into discriminator
    # netD.load_state_dict(model_dict)

    # print model parameter counts
    total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f"Number of parameters in netG: {total_params}")
    total_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    print(f"Number of parameters in netD: {total_params}")
    # build optimizers
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    # model save paths
    pathbest = config.SR_MODEL_BEST_4NOLSA
    pathG = config.SR_MODEL_G_4NOLSA
    pathD = config.SR_MODEL_D_4NOLSA
    # start training
    MyTrainer(trainData, val_hr_path,val_lr_path, testImgs, optimizerG, optimizerD, netG, netD, EPOCHS, device, pathG, pathD)
