import cv2
import numpy as np
import pywt
from torch.autograd import Variable
import torch
import torch.nn as nn
from  torchvision.models.vgg import vgg19

lossGAN = nn.BCEWithLogitsLoss()
lossF=nn.L1Loss()

def swt_change(image1, image2, level=1, wavelet='haar'):
    image1_np = image1.permute(1, 2, 0).cpu().detach().numpy()  # Reorder dimensions and convert to NumPy array
    image2_np = image2.permute(1, 2, 0).cpu().detach().numpy()
    yuv_image1 = cv2.cvtColor(image1_np, cv2.COLOR_BGR2YUV)
    # Convert the NumPy array's color space
    yuv_image2 = cv2.cvtColor(image2_np, cv2.COLOR_BGR2YUV)
    # Convert image to YUV color space
    # yuv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
    # yuv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2YUV)
    # Extract the Y channel
    y1 = yuv_image1[:, :, 0].astype(np.float32)
    y2 = yuv_image2[:, :, 0].astype(np.float32)
    # Perform SWT decomposition
    coeffs1 = pywt.swt2(y1, wavelet, level=level)
    coeffs2 = pywt.swt2(y2, wavelet, level=level)
    return coeffs1, coeffs2
def absloss(fakimg,sourceimg,device):
    realout_tensor = torch.tensor(sourceimg, dtype=torch.float32,device=device,requires_grad=True)
    fakout_tensor = torch.tensor(fakimg, dtype=torch.float32,device=device,requires_grad=True)
    loss=lossF(fakout_tensor,realout_tensor)
    return loss

def LossSWT(fakimg,realimg,device,weights=[0.1,0.01,0.01,0.05]):  # Weights for each subband
    swtloss = 0
    for idx in range(fakimg.size(0)):
        coeffs1,coeffs2=swt_change(fakimg[idx],realimg[idx])

        # Compute the difference between LL, LH, HL, and HH coefficients
        ll_diff = absloss(coeffs1[0][0],coeffs2[0][0],device)
        lh_diff = absloss(coeffs1[0][1][0],coeffs2[0][1][0],device)
        hl_diff = absloss(coeffs1[0][1][1],coeffs2[0][1][1],device)
        hh_diff = absloss(coeffs1[0][1][2],coeffs2[0][1][2],device)
        swtloss=weights[0]*ll_diff+weights[1]*lh_diff+weights[2]*hl_diff+weights[3]*hh_diff
    swtloss=swtloss/fakimg.size(0)

    return swtloss

def dlbranch(fakout, realout, device):
    realout_tensor = torch.tensor(realout, dtype=torch.float32, device=device, requires_grad=True)
    fakout_tensor = torch.tensor(fakout, dtype=torch.float32, device=device, requires_grad=True)
    valid = torch.ones_like(realout_tensor).to(device)  # Move tensor to the specified device
    fake = torch.zeros_like(realout_tensor).to(device)  # Move tensor to the specified device

    with torch.no_grad():
        mean_fakout = torch.mean(fakout_tensor, dim=0, keepdim=True)
        mean_realout = torch.mean(realout_tensor, dim=0, keepdim=True)

    loss_real = lossGAN(realout_tensor - mean_fakout, valid)
    loss_fake = lossGAN(fakout_tensor - mean_realout, fake)
    return (loss_fake + loss_real) / 2
def DLoss(fakimg,sourceimg,device):
    total_loss=0
    for idx in range(fakimg.size(0)):
        coeffsf, coeffsr = swt_change(fakimg[idx], sourceimg[idx])
        lh=dlbranch(coeffsf[0][1][0],coeffsr[0][1][0],device)
        hl=dlbranch(coeffsf[0][1][1],coeffsf[0][1][1],device)
        hh=dlbranch(coeffsf[0][1][2],coeffsf[0][1][2],device)
        total_loss=total_loss+lh+hl+hh
    total_loss=total_loss/fakimg.size(0)
    return total_loss
def glbranch(fakout,realout,device):
    realout_tensor = torch.tensor(realout, dtype=torch.float32, device=device, requires_grad=True)
    fakout_tensor = torch.tensor(fakout, dtype=torch.float32, device=device, requires_grad=True)
    valid = torch.ones_like(realout_tensor).to(device)  # Move tensor to the specified device
    #fake = torch.zeros_like(realout_tensor).to(device)  # Move tensor to the specified device
    #valid = torch.ones_like(realout).to(device)  # Create a tensor of ones with the same shape as realout
    rel_avg_real = realout_tensor - torch.mean(fakout_tensor, dim=0, keepdim=True)  # Relative mean difference: realout vs. mean of fakout
    rel_avg_fake = fakout_tensor - torch.mean(realout_tensor, dim=0, keepdim=True)  # Relative mean difference: fakout vs. mean of realout
    gLossGAN = lossGAN(rel_avg_fake - rel_avg_real, valid)  # Compute the generator's GAN loss

    return gLossGAN

def gLossGAN(fakimg,soueceimg,device):
    total_loss = 0
    for idx in range(fakimg.size(0)):
        coeffsf, coeffsr = swt_change(fakimg[idx], soueceimg[idx])
        lh = glbranch(coeffsf[0][1][0], coeffsr[0][1][0],device)
        hl = glbranch(coeffsf[0][1][1], coeffsr[0][1][1],device)
        hh = glbranch(coeffsf[0][1][2], coeffsf[0][1][2],device)
        total_loss+=lh+hl+hh
    total_loss=total_loss/fakimg.size(0)

    return total_loss

def perpetualloss(fakimg,realimg,device):
    vgg = vgg19(weights="VGG19_Weights.IMAGENET1K_V1").to(device)
    lossNetwork = nn.Sequential(*list(vgg.features.children())[:35]).eval()
    for param in lossNetwork.parameters():
        param.requires_grad = False
    loss=lossF(lossNetwork(fakimg),lossNetwork(realimg)).to(device)
    return loss

def Gloss(fakimg,realimg,device):
     #0.005->0.003
    gloss=LossSWT(fakimg,realimg,device)+0.003*gLossGAN(fakimg,realimg,device)+perpetualloss(fakimg,realimg,device)
    return gloss


