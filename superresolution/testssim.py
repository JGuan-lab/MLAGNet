import torch.nn as nn
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from Gen import Generator
import os
import cv2
from tqdm import tqdm
from metric import calculate
import re
import config
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load model
def load_model(model_path):
    model = Generator(3, 64, 2)
    pretrained_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model

# Reconstruct image using the model and save the result
def test_model(model, lr_image_path, sr_result_path):
    pre_transform = transforms.Compose([transforms.ToTensor()])
    pil_img = Image.open(lr_image_path)
    img = pre_transform(pil_img).unsqueeze(0).to(device)

    # Generate super-resolved image using the model
    with torch.no_grad():
        sr_img = model(img)[0, :, :, :].cpu().numpy()
        sr_img = sr_img.transpose((1, 2, 0))
        sr_img = np.clip(sr_img, 0, 1)  # Clip pixel values to [0, 1]

    # Save super-resolved image
    sr_img_pil = Image.fromarray((sr_img * 255).astype(np.uint8))
    sr_img_pil.save(sr_result_path)
    print(f"Super-resolved image saved to: {sr_result_path}")
def test_images_in_folder(model, lr_folder_path, sr_result_folder_path, hr_folder_path, results_file):
    os.makedirs(sr_result_folder_path, exist_ok=True)
    best_psnr = 0
    best_file = ''
    psnrs = []
    ssims = []
    file_nums = []

    with open(results_file, 'w') as f:
        f.write("Filename\tPSNR\tSSIM\n")  # Write header

        for filename in tqdm(os.listdir(lr_folder_path)):
            if filename.endswith((".png", ".jpg")):
                lr_image_path = os.path.join(lr_folder_path, filename)
                sr_result_path = os.path.join(sr_result_folder_path, filename)
                hr_image_path = os.path.join(hr_folder_path, filename)  # High-resolution image path

                # Generate super-resolved image
                test_model(model, lr_image_path, sr_result_path)

                # Read the generated SR image and the corresponding HR image
                sr_img = cv2.imread(sr_result_path)
                hr_img = cv2.imread(hr_image_path)

                if sr_img is not None and hr_img is not None:
                    # Compute PSNR and SSIM values
                    psnr, ssim = calculate(sr_img, hr_img)
                    file_num = int(re.findall(r'\d+', filename)[0])  # Assume filename contains a number
                    psnrs.append(psnr)
                    ssims.append(ssim)
                    file_nums.append(file_num)

                    # Update best PSNR
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_file = filename

                    print(f"PSNR for {filename}: {psnr}, SSIM: {ssim}")
                    f.write(f"{filename}\t{psnr:.4f}\t{ssim:.4f}\n")  # Write per-image result

        avg_psnr = np.mean(psnrs)  # Compute average PSNR
        avg_ssim = np.mean(ssims)  # Compute average SSIM

        # Print best result
        print(f"Best file: {best_file}")
        print(f"Best PSNR: {best_psnr}")
        print(f"Average PSNR: {avg_psnr}")
        print(f"Average SSIM: {avg_ssim}")

        return avg_psnr, best_psnr, avg_ssim, best_file

# Iterate over model folder and save results
def evaluate_models_in_folder(model_folder_path, lr_folder_path, hr_folder_path, results_file_path):
    with open(results_file_path, 'w') as f:
        f.write("Model Name\tAverage PSNR\tBest PSNR\tAverage SSIM\tBest File\n")  # Write header

        for model_filename in os.listdir(model_folder_path):
            if model_filename.endswith(".pth"):
                model_path = os.path.join(model_folder_path, model_filename)
                sr_result_folder_path = os.path.join(config.SR_TMP_RESULTS_SSIM, model_filename.split('.')[0])  # Temp result folder

                # Load model and compute PSNR and SSIM values
                model = load_model(model_path)
                avg_psnr, best_psnr, avg_ssim, best_file = test_images_in_folder(
                    model, lr_folder_path, sr_result_folder_path, hr_folder_path, results_file_path.replace(".txt", f"_{model_filename.split('.')[0]}.txt")
                )

                # Write model results to file
                f.write(f"{model_filename}\t{avg_psnr:.4f}\t{best_psnr:.4f}\t{avg_ssim:.4f}\t{best_file}\n")
                print(f"Results for {model_filename} saved to {results_file_path}")

def main():
    model_folder_path = config.SR_TEST_MODEL_BEST  # Model folder
    lr_folder_path = config.SR_TEST_LR_SSIM  # Low-resolution image folder
    hr_folder_path = config.SR_TEST_HR_SSIM  # High-resolution image folder
    results_file_path = config.SR_RESULT_TXT_BESTPHOTO  # Results file

    # Iterate over model folder and compute results
    evaluate_models_in_folder(model_folder_path, lr_folder_path, hr_folder_path, results_file_path)

if __name__ == "__main__":
    main()