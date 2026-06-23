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

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model loading function
def load_model(model_path):
    model = Generator(3, 64, 2)
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

# batch-process images in folder
def test_images_in_folder(model, lr_folder_path, sr_result_folder_path, hr_folder_path):
    os.makedirs(sr_result_folder_path, exist_ok=True)
    best_psnr = 0
    best_file = ''
    psnrs = []
    file_nums = []

    for filename in tqdm(os.listdir(lr_folder_path)):
        if filename.endswith((".png", ".jpg")):
            lr_image_path = os.path.join(lr_folder_path, filename)
            sr_result_path = os.path.join(sr_result_folder_path, filename)
            hr_image_path = os.path.join(hr_folder_path, filename)  # high-resolution image path

            # generate super-resolved image
            test_model(model, lr_image_path, sr_result_path)

            # read generated SR image and corresponding HR image
            sr_img = cv2.imread(sr_result_path)
            print(sr_img.shape)
            hr_img = cv2.imread(hr_image_path)
            print(hr_img.shape)
            if sr_img is not None and hr_img is not None:
                # compute PSNR
                psnr, _ = calculate(sr_img, hr_img)
                file_num = int(re.findall(r'\d+', filename)[0])  # assume filename contains a number
                psnrs.append(psnr)
                file_nums.append(file_num)

                # update best PSNR
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_file = filename

                print(f"PSNR for {filename}: {psnr}")

    avg_psnr = np.mean(psnrs)  # compute average PSNR

    # print best results
    print(f"Best file: {best_file}")
    print(f"Best PSNR: {best_psnr}")
    print(f"Average PSNR: {avg_psnr}")

    return avg_psnr, best_psnr, best_file

# iterate model folder and save results
def evaluate_models_in_folder(model_folder_path, lr_folder_path, hr_folder_path, results_file_path):
    error_file_path = config.SR_RESULT_ERROR_TXT  # error model log file
    with open(results_file_path, 'w') as results_file, open(error_file_path, 'w') as error_file:
        results_file.write("Model Name\tAverage PSNR\tBest PSNR\tBest File\n")  # write header
        for model_filename in os.listdir(model_folder_path):
            if model_filename.endswith(".pth") and (model_filename.startswith("netG") or model_filename.startswith("best")):
                model_path = os.path.join(model_folder_path, model_filename)
                # sr_result_folder_path = os.path.join(config.SR_TMP_RESULTS_4, model_filename.split('.')[0])  # temporary result folder
                sr_result_folder_path = config.SR_TMP_RESULTS
                try:
                    # load model and compute PSNR
                    model = load_model(model_path)
                    avg_psnr, best_psnr, best_file = test_images_in_folder(model, lr_folder_path, sr_result_folder_path, hr_folder_path)

                    # write results to file
                    results_file.write(f"{model_filename}\t{avg_psnr:.4f}\t{best_psnr:.4f}\t{best_file}\n")
                    print(f"Results for {model_filename} saved to {results_file_path}")

                except Exception as e:
                    # catch exception and log the failed model
                    error_file.write(f"Error loading model {model_filename}: {str(e)}\n")
                    print(f"Error with model {model_filename}: {e}")

def main():
    model_folder_path = config.SR_TEST_MODEL_4FULL  # model folder
    lr_folder_path = config.SR_QINGT_LR  # LR image folder
    hr_folder_path = config.SR_QINGT_HR  # HR image folder
    results_file_path = config.SR_RESULT_TXT_4TEST  # results file

    # iterate model folder and compute results
    evaluate_models_in_folder(model_folder_path, lr_folder_path, hr_folder_path, results_file_path)

if __name__ == "__main__":
    main()
