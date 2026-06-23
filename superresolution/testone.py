import torch.nn as nn
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from Gen import Generator
import os
import cv2
from metric import calculate
import config

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model loading function
def load_model(model_path):
    model = Generator(3, 64, 2)  # ensure Generator args match your model
    pretrained_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model

# reconstruct image with model and save result
def test_model(model, lr_image_path, sr_result_path):
    pre_transform = transforms.Compose([transforms.ToTensor()])
    pil_img = Image.open(lr_image_path)
    img = pre_transform(pil_img).unsqueeze(0).to(device)

    # generate super-resolved image
    with torch.no_grad():
        sr_img = model(img)[0, :, :, :].cpu().numpy()
        sr_img = sr_img.transpose((1, 2, 0))
        sr_img = np.clip(sr_img, 0, 1)  # clamp pixel values to [0, 1]

    # save super-resolved image
    sr_img_pil = Image.fromarray((sr_img * 255).astype(np.uint8))
    sr_img_pil.save(sr_result_path)
    print(f"Super-resolved image saved to: {sr_result_path}")

# test a given model on a single image
def evaluate_single_image(model_path, lr_image_path, sr_result_path, hr_image_path=None):
    # load model
    model = load_model(model_path)

    # generate and save super-resolved image
    test_model(model, lr_image_path, sr_result_path)

    # compute PSNR if HR image path is provided
    if hr_image_path:
        sr_img = cv2.imread(sr_result_path)
        hr_img = cv2.imread(hr_image_path)

        if sr_img is not None and hr_img is not None:
            print(sr_img.shape)
            print(hr_img.shape)
            psnr, ssim = calculate(sr_img, hr_img)
            print(f"PSNR for the generated image: {psnr}")
            print(f"SSIM for the generated image: {ssim}")

def main():
    # set model, input, and output paths
    model_path = config.SR_TESTONE_MODEL  # replace with your model path
    lr_image_path = os.path.join(config.SR_TESTONE_RESIZE, '9999.png')  # replace with your LR image path
    sr_result_path = config.SR_TESTONE_SR_OUT  # replace with your desired output path
    hr_image_path = os.path.join(config.SR_TESTONE_GT_DIR, '9999.png')  # replace with your HR image path (optional)

    # run evaluation
    evaluate_single_image(model_path, lr_image_path, sr_result_path, hr_image_path)

if __name__ == "__main__":
    main()
