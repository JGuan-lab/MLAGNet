# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os
# import torch.optim as optim
# from PIL import Image
# from tqdm import tqdm
# import torchvision.utils as vutils
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# import torchvision.models as models
# import random
# from torchvision.transforms import ToTensor
# from accelerate import Accelerator
# import re

# # assume these files are in the current directory, import directly
# try:
#     from UnetDiscriminator import UNetDiscriminatorSN
#     from Gen import Generator
#     from metric import calculate
# except ImportError:
#     print("Please ensure UnetDiscriminator.py, Gen.py, and metric.py are in the current directory")

# # ==============================================================================
# # 1. Perceptual Loss (VGG Perceptual Loss)
# # ==============================================================================
# class VGGPerceptualLoss(nn.Module):
#     def __init__(self, layer_ids=[34]): # conv5_4
#         super(VGGPerceptualLoss, self).__init__()
#         # use the weights parameter instead of pretrained to suppress warnings (for newer torchvision)
#         # if older version errors, fall back to pretrained=True
#         try:
#             from torchvision.models import VGG19_Weights
#             self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
#         except:
#             self.vgg = models.vgg19(pretrained=True).features
            
#         self.layer_ids = layer_ids
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     def forward(self, x, y):
#         x = (x - self.mean) / self.std
#         y = (y - self.mean) / self.std
#         loss = 0
#         for i, layer in enumerate(self.vgg):
#             x = layer(x)
#             y = layer(y)
#             if i in self.layer_ids:
#                 loss += F.l1_loss(x, y)
#         return loss

# # ==============================================================================
# # 2. Dataset Processing
# # ==============================================================================
# class PreprocessDataset(Dataset):
#     def __init__(self, gtPath, lrPath, gt_patch_size, scale):
#         self.gt_imgs = []
#         self.lr_imgs = []
#         self.gt_patch_size = gt_patch_size
#         self.scale = scale

#         for root, _, files in os.walk(gtPath):
#             for file in tqdm(files, desc="Loading GT"):
#                 self.gt_imgs.append(os.path.join(root, file))

#         for root, _, files in os.walk(lrPath):
#             for file in tqdm(files, desc="Loading LR"):
#                 self.lr_imgs.append(os.path.join(root, file))
        
#         self.gt_imgs.sort()
#         self.lr_imgs.sort()

#     def paired_random_crop(self, img_gts, img_lqs, gt_patch_size, scale):
#         h_lq, w_lq = img_lqs.size
#         lq_patch_size = gt_patch_size // scale

#         top = random.randint(0, h_lq - lq_patch_size)
#         left = random.randint(0, w_lq - lq_patch_size)
#         bottom = top + lq_patch_size
#         right = left + lq_patch_size
#         img_lqs = img_lqs.crop((left, top, right, bottom))

#         top_gt, left_gt = int(top * scale), int(left * scale)
#         right_gt = left_gt + gt_patch_size
#         bottom_gt = top_gt + gt_patch_size
#         img_gts = img_gts.crop((left_gt, top_gt, right_gt, bottom_gt))
#         return img_gts, img_lqs

#     def __len__(self):
#         return min(len(self.gt_imgs), len(self.lr_imgs))

#     def __getitem__(self, index):
#         gt_tempImg = self.gt_imgs[index]
#         lr_tempImg = self.lr_imgs[index]
#         try:
#             gt_img = Image.open(gt_tempImg).convert('RGB')
#             lr_img = Image.open(lr_tempImg).convert('RGB')
#             img_gts, img_lqs = self.paired_random_crop(gt_img, lr_img, self.gt_patch_size, self.scale)
#             sourceImg = ToTensor()(img_gts)
#             cropImg = ToTensor()(img_lqs)
#             return cropImg, sourceImg
#         except Exception as e:
#             print(f"Error loading image at index {index}: {str(e)}")
#             return None

# def collate_fn_skip_none(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

# def MyDataloader(gtPath, lrPath, gt_patch_size, scale, BATCH, device):
#     processDataset = PreprocessDataset(gtPath, lrPath, gt_patch_size, scale)
#     trainData = DataLoader(processDataset, batch_size=BATCH, shuffle=True, num_workers=2, collate_fn=collate_fn_skip_none,drop_last=True)
    
#     # take one batch for visualization
#     temp_loader = DataLoader(processDataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn_skip_none)
#     try:
#         testImgs, _ = next(iter(temp_loader))
#         testImgs = testImgs.to(device)
#     except:
#         testImgs = None
#     return trainData, testImgs

# # ==============================================================================
# # 3. Helper Functions
# # ==============================================================================
# def extract_index(file_path):
#     filename = os.path.basename(file_path)
#     index_match = re.search(r'\d+', filename)
#     if index_match:
#         return int(index_match.group())
#     return None

# def calculate_val(model, lr_path, hr_path, epoch, device):
#     preTransform = transforms.Compose([transforms.ToTensor()])
#     pilImg = Image.open(lr_path).convert('RGB')
#     img = preTransform(pilImg).unsqueeze(0).to(device)
    
#     # inference
#     model.eval()
#     with torch.no_grad():
#         source = model(img)[0, :, :, :]
    
#     source = source.cpu().detach().numpy()
#     source = source.transpose((1, 2, 0))
#     source = np.clip(source, 0, 1)

#     fakeimg = Image.fromarray(np.uint8(source * 255))
#     index = extract_index(lr_path)
#     if index is None: return None, None

#     os.makedirs("/home/Yb/uureal/temp/", exist_ok=True)
#     output_path = os.path.join("/home/Yb/uureal/temp/", f"fake_{epoch}_{index}.png")

#     try:
#         fakeimg.save(output_path)
#         fake_img_png = plt.imread(output_path).astype(np.float32) / 255.0
#         real_img_png = cv2.imread(hr_path).astype(np.float32) / 255.0
#         real_img_png = cv2.cvtColor(real_img_png, cv2.COLOR_BGR2RGB)
        
#         if fake_img_png.shape != real_img_png.shape:
#             real_img_png = cv2.resize(real_img_png, (fake_img_png.shape[1], fake_img_png.shape[0]))

#         psnr, ssim = calculate(fake_img_png, real_img_png)
#         return psnr, ssim
#     except Exception as e:
#         print(f"Eval Error: {e}")
#         return None, None

# # ==============================================================================
# # 4. Trainer (validation frequency updated)
# # ==============================================================================
# def MyTrainer(trainData, val_hr_path, val_lr_path, testImgs, optimizerG, optimizerD, netG, netD, EPOCHS, device, pathG, pathD, pathbest, accelerator):

#     # define losses
#     criterion_L1 = nn.L1Loss().to(device)
#     criterion_perceptual = VGGPerceptualLoss().to(device)
#     criterion_gan = nn.BCEWithLogitsLoss().to(device)

#     # loss weights
#     lambda_pixel = 1.0
#     lambda_perceptual = 1.0
#     lambda_gan = 0.1

#     # Prepare
#     netG, netD, optimizerG, optimizerD, trainData = accelerator.prepare(netG, netD, optimizerG, optimizerD, trainData)

#     best_psnr = 0
#     # set validation interval
#     VAL_INTERVAL = 500
    
#     for epoch in range(EPOCHS):
#         netG.train()
#         netD.train()
        
#         processBar = tqdm(enumerate(trainData, 1), total=len(trainData), disable=not accelerator.is_local_main_process)
        
#         for i, (cropImg, sourceImg) in processBar:
#             fakeImg = netG(cropImg)

#             # --- Update D ---
#             optimizerD.zero_grad()
#             pred_real = netD(sourceImg)
#             pred_fake = netD(fakeImg.detach())

#             loss_d_real = criterion_gan(pred_real - torch.mean(pred_fake), torch.ones_like(pred_real))
#             loss_d_fake = criterion_gan(pred_fake - torch.mean(pred_real), torch.zeros_like(pred_fake))
#             loss_d = (loss_d_real + loss_d_fake) / 2
            
#             accelerator.backward(loss_d)
#             optimizerD.step()

#             # --- Update G ---
#             optimizerG.zero_grad()
#             pred_real = netD(sourceImg).detach()
#             pred_fake = netD(fakeImg)

#             l_pixel = criterion_L1(fakeImg, sourceImg)
#             l_percep = criterion_perceptual(fakeImg, sourceImg)
            
#             loss_g_real = criterion_gan(pred_real - torch.mean(pred_fake), torch.zeros_like(pred_real))
#             loss_g_fake = criterion_gan(pred_fake - torch.mean(pred_real), torch.ones_like(pred_fake))
#             l_gan = (loss_g_real + loss_g_fake) / 2

#             loss_g_total = (lambda_pixel * l_pixel) + (lambda_perceptual * l_percep) + (lambda_gan * l_gan)
            
#             accelerator.backward(loss_g_total)
#             optimizerG.step()

#             if i % 10 == 0:
#                 processBar.set_description(
#                     desc=f'[{epoch}/{EPOCHS}] D:{loss_d.item():.3f} G_Pix:{l_pixel.item():.3f} G_Per:{l_percep.item():.3f} G_Adv:{l_gan.item():.3f}'
#                 )

#         # -----------------------------------------------------------
#         #  Validation & Save - every 500 epochs
#         # -----------------------------------------------------------
#         if (epoch % 50 == 0) or (epoch == EPOCHS - 1):

#             # 1. save current epoch model (main process only)
#             if accelerator.is_main_process:
#                 # must unwrap to save the original state_dict
#                 unwrapped_netG = accelerator.unwrap_model(netG)
#                 unwrapped_netD = accelerator.unwrap_model(netD)
#                 os.makedirs(pathG,exist_ok=True)
#                 os.makedirs(pathD,exist_ok=True)
#                 os.makedirs(pathbest,exist_ok=True)
#                 print(f"generator:{os.path.abspath(pathG)}")
#                 print(f"discriminator:{os.path.abspath(pathD)}")

                
#                 torch.save(unwrapped_netG.state_dict(), os.path.join(pathG, f'netG_epoch_{epoch}.pth'))
#                 torch.save(unwrapped_netD.state_dict(), os.path.join(pathD, f'netD_epoch_{epoch}.pth'))
#                 print(f"Saved Checkpoint at epoch {epoch}")

#                 # 2. validation logic
#                 print(f"Running Validation at epoch {epoch}...")
#                 netG_eval = unwrapped_netG  # use unwrapped model to avoid DDP broadcast deadlock
#                 netG_eval.eval()

#                 # save visualization
#                 with torch.no_grad():
#                     if testImgs is not None:
#                         fakeImgs_vis = netG_eval(testImgs[:4]).detach().cpu()
#                         os.makedirs('/home/Yb/uureal/out/', exist_ok=True)
#                         vutils.save_image(fakeImgs_vis, f'/home/Yb/uureal/out/Result_epoch_{epoch:05d}.jpg', normalize=True, padding=2)

#                 # compute PSNR/SSIM
#                 val_psnr = 0
#                 val_ssim = 0
#                 num_val_images = 0
                
#                 lr_files = sorted(os.listdir(val_lr_path))
#                 hr_files = sorted(os.listdir(val_hr_path))
                
#                 for lr_f, hr_f in zip(lr_files, hr_files):
#                     psnr, ssim = calculate_val(netG_eval, os.path.join(val_lr_path, lr_f), os.path.join(val_hr_path, hr_f), epoch, device)
#                     if psnr is not None:
#                         val_psnr += psnr
#                         val_ssim += ssim
#                         num_val_images += 1
                
#                 if num_val_images > 0:
#                     val_psnr /= num_val_images
#                     val_ssim /= num_val_images
#                     print(f"Epoch {epoch} Val PSNR: {val_psnr:.4f} SSIM: {val_ssim:.4f}")

#                     if val_psnr > best_psnr:
#                         best_psnr = val_psnr
#                         torch.save(netG_eval.state_dict(), os.path.join(pathbest, f'best_{best_psnr:.2f}_netG.pth'))
#                         print(f"New Best PSNR: {best_psnr:.4f}")

#             # [IMPORTANT: deadlock prevention]
#             # even if only the main process runs validation, all others must wait here.
#             # otherwise they advance to the next epoch and cause NCCL timeout.
#             accelerator.wait_for_everyone()

# # ==============================================================================
# # 5. Main Execution
# # ==============================================================================
# if __name__ == '__main__':
#     # path configuration
#     gt_path = r"/home/yons/data/Weiyubing/lungtotal"
#     lq_path = r"/home/yons/data/Weiyubing/lungresize4"
#     val_lr_path = r"/home/Yb/uureal/Ui/ctlow"
#     val_hr_path = r"/home/Yb/uureal/Ui/cttrue"

#     # output paths
#     pathbest = r"/home/yons/data/Weiyubing/model/4StandardGAN/"
#     pathG = r"/home/yons/data/Weiyubing/model/4StandardGAN/"
#     pathD = r"/home/yons/data/Weiyubing/model/4StandardGAN_D/"
    
#     os.makedirs(pathbest, exist_ok=True)
#     os.makedirs(pathG, exist_ok=True)
#     os.makedirs(pathD, exist_ok=True)

#     # initialize Accelerator
#     accelerator = Accelerator()
#     device = accelerator.device
#     if accelerator.is_main_process:
#         print(f"Training on Device: {device}")

#     BATCH = 32
#     EPOCHS = 4501

#     # load dataset
#     trainData, testImgs = MyDataloader(gt_path, lq_path, 128, 4, BATCH, device)

#     # initialize models
#     netG = Generator(3, 64, 2)
#     netD = UNetDiscriminatorSN()

#     # load pretrained weights (L1 model for warm start)
#     try:
#         pretrained_path = "/home/Yb/uureal/Ui/model/netGl1loss_epoch_4_20.pth"
#         if os.path.exists(pretrained_path):
#             pretrained_dict = torch.load(pretrained_path, map_location=device)
#             model_dict = netG.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             model_dict.update(pretrained_dict)
#             netG.load_state_dict(model_dict)
#             if accelerator.is_main_process:
#                 print("Loaded pretrained L1 model.")
#         else:
#             if accelerator.is_main_process:
#                 print("Pretrained model not found, training from scratch.")
#     except Exception as e:
#         print(f"Error loading pretrained: {e}")

#     # optimizers
#     optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.99))
#     optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9, 0.99))

#     # start training
#     MyTrainer(trainData, val_hr_path, val_lr_path, testImgs, optimizerG, optimizerD, netG, netD, EPOCHS, device, pathG, pathD, pathbest, accelerator)
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import random
from torchvision.transforms import ToTensor
from accelerate import Accelerator
import re
import config

# assume these files are in the current directory, import directly
try:
    from UnetDiscriminator import UNetDiscriminatorSN
    from Gen import Generator
    from metric import calculate
except ImportError:
    print("Please ensure UnetDiscriminator.py, Gen.py, and metric.py are in the current directory")

# ==============================================================================
# 1. Perceptual Loss (VGG Perceptual Loss)
# ==============================================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[34]): # conv5_4
        super(VGGPerceptualLoss, self).__init__()
        try:
            from torchvision.models import VGG19_Weights
            self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        except:
            self.vgg = models.vgg19(pretrained=True).features
            
        self.layer_ids = layer_ids
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.l1_loss(x, y)
        return loss

# ==============================================================================
# 2. Dataset Processing
# ==============================================================================
class PreprocessDataset(Dataset):
    def __init__(self, gtPath, lrPath, gt_patch_size, scale):
        self.gt_imgs = []
        self.lr_imgs = []
        self.gt_patch_size = gt_patch_size
        self.scale = scale

        for root, _, files in os.walk(gtPath):
            for file in files:  # drop tqdm to avoid console spam during multi-threaded loading
                self.gt_imgs.append(os.path.join(root, file))

        for root, _, files in os.walk(lrPath):
            for file in files:
                self.lr_imgs.append(os.path.join(root, file))
        
        self.gt_imgs.sort()
        self.lr_imgs.sort()

    def paired_random_crop(self, img_gts, img_lqs, gt_patch_size, scale):
        h_lq, w_lq = img_lqs.size
        lq_patch_size = gt_patch_size // scale

        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
        bottom = top + lq_patch_size
        right = left + lq_patch_size
        img_lqs = img_lqs.crop((left, top, right, bottom))

        top_gt, left_gt = int(top * scale), int(left * scale)
        right_gt = left_gt + gt_patch_size
        bottom_gt = top_gt + gt_patch_size
        img_gts = img_gts.crop((left_gt, top_gt, right_gt, bottom_gt))
        return img_gts, img_lqs

    def __len__(self):
        return min(len(self.gt_imgs), len(self.lr_imgs))

    def __getitem__(self, index):
        gt_tempImg = self.gt_imgs[index]
        lr_tempImg = self.lr_imgs[index]
        try:
            gt_img = Image.open(gt_tempImg).convert('RGB')
            lr_img = Image.open(lr_tempImg).convert('RGB')
            img_gts, img_lqs = self.paired_random_crop(gt_img, lr_img, self.gt_patch_size, self.scale)
            sourceImg = ToTensor()(img_gts)
            cropImg = ToTensor()(img_lqs)
            return cropImg, sourceImg
        except Exception as e:
            print(f"Error loading image at index {index}: {str(e)}")
            return None

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def MyDataloader(gtPath, lrPath, gt_patch_size, scale, BATCH, device):
    processDataset = PreprocessDataset(gtPath, lrPath, gt_patch_size, scale)
    # num_workers=0 to avoid deadlock risk; on Linux with enough RAM, set to 2 or 4
    trainData = DataLoader(processDataset, batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=collate_fn_skip_none, drop_last=True)

    # take one batch for visualization
    temp_loader = DataLoader(processDataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn_skip_none)
    try:
        testImgs, _ = next(iter(temp_loader))
        testImgs = testImgs.to(device)
    except:
        testImgs = None
    return trainData, testImgs

# ==============================================================================
# 3. Helper Functions
# ==============================================================================
def extract_index(file_path):
    filename = os.path.basename(file_path)
    index_match = re.search(r'\d+', filename)
    if index_match:
        return int(index_match.group())
    return None

def calculate_val(model, lr_path, hr_path, epoch, device):
    preTransform = transforms.Compose([transforms.ToTensor()])
    pilImg = Image.open(lr_path).convert('RGB')
    img = preTransform(pilImg).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        source = model(img)[0, :, :, :]
    
    source = source.cpu().detach().numpy()
    source = source.transpose((1, 2, 0))
    source = np.clip(source, 0, 1)

    fakeimg = Image.fromarray(np.uint8(source * 255))
    index = extract_index(lr_path)
    if index is None: return None, None

    os.makedirs(config.SR_VIS_TEMP_DIR, exist_ok=True)
    output_path = os.path.join(config.SR_VIS_TEMP_DIR, f"fake_{epoch}_{index}.png")

    try:
        fakeimg.save(output_path)
        fake_img_png = plt.imread(output_path).astype(np.float32) / 255.0
        real_img_png = cv2.imread(hr_path).astype(np.float32) / 255.0
        real_img_png = cv2.cvtColor(real_img_png, cv2.COLOR_BGR2RGB)

        if fake_img_png.shape != real_img_png.shape:
            real_img_png = cv2.resize(real_img_png, (fake_img_png.shape[1], fake_img_png.shape[0]))

        psnr, ssim = calculate(fake_img_png, real_img_png)
        return psnr, ssim
    except Exception as e:
        print(f"Eval Error: {e}")
        return None, None

# ==============================================================================
# 4. Trainer (supports resume from checkpoint)
# ==============================================================================
def MyTrainer(trainData, val_hr_path, val_lr_path, testImgs, optimizerG, optimizerD, netG, netD, start_epoch, final_epoch, device, pathG, pathD, pathbest, accelerator):

    criterion_L1 = nn.L1Loss().to(device)
    criterion_perceptual = VGGPerceptualLoss().to(device)
    criterion_gan = nn.BCEWithLogitsLoss().to(device)

    lambda_pixel = 1.0
    lambda_perceptual = 1.0
    lambda_gan = 0.1 

    # Prepare (after prepare, netG becomes DistributedDataParallel or similar wrapper)
    netG, netD, optimizerG, optimizerD, trainData = accelerator.prepare(netG, netD, optimizerG, optimizerD, trainData)

    best_psnr = 0

    # print training start info
    if accelerator.is_main_process:
        print(f"Start training: epoch {start_epoch + 1} -> {final_epoch}")

    # start from start_epoch
    for epoch in range(start_epoch + 1, final_epoch + 1):
        netG.train()
        netD.train()
        
        processBar = tqdm(enumerate(trainData, 1), total=len(trainData), disable=not accelerator.is_local_main_process)
        
        for i, (cropImg, sourceImg) in processBar:
            fakeImg = netG(cropImg)

            # --- Update D ---
            optimizerD.zero_grad()
            pred_real = netD(sourceImg)
            pred_fake = netD(fakeImg.detach())

            loss_d_real = criterion_gan(pred_real - torch.mean(pred_fake), torch.ones_like(pred_real))
            loss_d_fake = criterion_gan(pred_fake - torch.mean(pred_real), torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            accelerator.backward(loss_d)
            optimizerD.step()

            # --- Update G ---
            optimizerG.zero_grad()
            pred_real = netD(sourceImg).detach()
            pred_fake = netD(fakeImg)

            l_pixel = criterion_L1(fakeImg, sourceImg)
            l_percep = criterion_perceptual(fakeImg, sourceImg)
            
            loss_g_real = criterion_gan(pred_real - torch.mean(pred_fake), torch.zeros_like(pred_real))
            loss_g_fake = criterion_gan(pred_fake - torch.mean(pred_real), torch.ones_like(pred_fake))
            l_gan = (loss_g_real + loss_g_fake) / 2

            loss_g_total = (lambda_pixel * l_pixel) + (lambda_perceptual * l_percep) + (lambda_gan * l_gan)
            
            accelerator.backward(loss_g_total)
            optimizerG.step()

            if i % 10 == 0:
                processBar.set_description(
                    desc=f'[{epoch}/{final_epoch}] D:{loss_d.item():.3f} G_Pix:{l_pixel.item():.3f} G_Per:{l_percep.item():.3f} G_Adv:{l_gan.item():.3f}'
                )

        # -----------------------------------------------------------
        #  Validation & Save - every 50 epochs
        # -----------------------------------------------------------
        if (epoch % 50 == 0) or (epoch == final_epoch):

            if accelerator.is_main_process:
                unwrapped_netG = accelerator.unwrap_model(netG)
                unwrapped_netD = accelerator.unwrap_model(netD)

                # ensure output dirs exist
                os.makedirs(pathG, exist_ok=True)
                os.makedirs(pathD,exist_ok=True)
                os.makedirs(pathbest,exist_ok=True)

                # save model
                torch.save(unwrapped_netG.state_dict(), os.path.join(pathG, f'netG_epoch_{epoch}.pth'))
                torch.save(unwrapped_netD.state_dict(), os.path.join(pathD, f'netD_epoch_{epoch}.pth'))
                print(f"Saved Checkpoint at epoch {epoch}")

                # validation logic
                print(f"Running Validation at epoch {epoch}...")
                netG_eval = unwrapped_netG
                netG_eval.eval()

                with torch.no_grad():
                    if testImgs is not None:
                        fakeImgs_vis = netG_eval(testImgs[:4]).detach().cpu()
                        os.makedirs(config.SR_VIS_OUT_TRAINVAL, exist_ok=True)
                        vutils.save_image(fakeImgs_vis, os.path.join(config.SR_VIS_OUT_TRAINVAL, f'Result_epoch_{epoch:05d}.jpg'), normalize=True, padding=2)

                val_psnr = 0
                val_ssim = 0
                num_val_images = 0
                
                lr_files = sorted(os.listdir(val_lr_path))
                hr_files = sorted(os.listdir(val_hr_path))
                
                for lr_f, hr_f in zip(lr_files, hr_files):
                    psnr, ssim = calculate_val(netG_eval, os.path.join(val_lr_path, lr_f), os.path.join(val_hr_path, hr_f), epoch, device)
                    if psnr is not None:
                        val_psnr += psnr
                        val_ssim += ssim
                        num_val_images += 1
                
                if num_val_images > 0:
                    val_psnr /= num_val_images
                    val_ssim /= num_val_images
                    print(f"Epoch {epoch} Val PSNR: {val_psnr:.4f} SSIM: {val_ssim:.4f}")

                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        torch.save(netG_eval.state_dict(), os.path.join(pathbest, f'best_{best_psnr:.2f}_netG.pth'))
                        print(f"New Best PSNR: {best_psnr:.4f}")

            accelerator.wait_for_everyone()

# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == '__main__':
    # path configuration
    gt_path = config.SR_TRAIN_GT_4X
    lq_path = config.SR_TRAIN_LR_4X
    val_lr_path = config.SR_VAL_LR
    val_hr_path = config.SR_VAL_HR

    # output paths
    pathbest = config.SR_MODEL_BEST_4STDGAN
    pathG = config.SR_MODEL_G_4STDGAN
    pathD = config.SR_MODEL_D_4STDGAN

    os.makedirs(pathbest, exist_ok=True)
    os.makedirs(pathG, exist_ok=True)
    os.makedirs(pathD, exist_ok=True)

    # initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Training on Device: {device}")

    BATCH = 32

    # ================= resume configuration =================
    resume_training = True  # [toggle] whether to resume from a saved checkpoint
    start_epoch = 2000      # [start] epoch number of last checkpoint
    final_epoch = 4501      # [end] target total epochs
    # ========================================================

    # load dataset
    trainData, testImgs = MyDataloader(gt_path, lq_path, 128, 4, BATCH, device)

    # initialize models
    netG = Generator(3, 64, 2)
    netD = UNetDiscriminatorSN()

    # 1. try resume from checkpoint first (epoch 2000)
    if resume_training and start_epoch > 0:
        resume_G_path = os.path.join(pathG, f'netG_epoch_{start_epoch}.pth')
        resume_D_path = os.path.join(pathD, f'netD_epoch_{start_epoch}.pth')

        if os.path.exists(resume_G_path) and os.path.exists(resume_D_path):
            if accelerator.is_main_process:
                print(f"Loading weights from epoch {start_epoch} to resume training...")

            netG.load_state_dict(torch.load(resume_G_path, map_location=device))
            netD.load_state_dict(torch.load(resume_D_path, map_location=device))

            if accelerator.is_main_process:
                print("Successfully loaded epoch 2000 weights.")
        else:
            if accelerator.is_main_process:
                print(f"Warning: weight file for epoch {start_epoch} not found: {resume_G_path}")
                print("Will try the L1 pretrained model or start from scratch...")
            start_epoch = 0  # reset if checkpoint not found

            # fallback: try loading L1 pretrained model
            try:
                pretrained_path = config.UI_MODEL_DIR + "/netGl1loss_epoch_4_20.pth"
                if os.path.exists(pretrained_path):
                    pretrained_dict = torch.load(pretrained_path, map_location=device)
                    model_dict = netG.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    netG.load_state_dict(model_dict)
                    if accelerator.is_main_process:
                        print("Loaded pretrained L1 model.")
            except Exception as e:
                print(f"Error loading pretrained: {e}")

    # 2. if not resuming, use normal cold-start flow
    elif start_epoch == 0:
        try:
            pretrained_path = "/home/Yb/uureal/Ui/model/netGl1loss_epoch_4_20.pth"
            if os.path.exists(pretrained_path):
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                model_dict = netG.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                netG.load_state_dict(model_dict)
                if accelerator.is_main_process:
                    print("Loaded pretrained L1 model.")
        except Exception as e:
            print(f"Error loading pretrained: {e}")

    # optimizers
    # note: lr=1e-4 is fine if loss was stable; use 5e-5 for fine-tuning
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # start training (pass start_epoch and final_epoch)
    MyTrainer(trainData, val_hr_path, val_lr_path, testImgs, optimizerG, optimizerD, netG, netD, start_epoch, final_epoch, device, pathG, pathD, pathbest, accelerator)