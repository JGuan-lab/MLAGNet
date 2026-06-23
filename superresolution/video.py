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

# Function to load the model (same as original code)
def load_model(model_path):
    model = Generator(3, 64, 2)
    pretrained_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model

# Reconstruct image using the model and save the result (same as original code)
def test_model(model, lr_image_path, sr_result_path):
    pre_transform = transforms.Compose([transforms.ToTensor()])
    pil_img = Image.open(lr_image_path)
    img = pre_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_img = model(img)[0, :, :, :].cpu().numpy()
        sr_img = sr_img.transpose((1, 2, 0))
        sr_img = np.clip(sr_img, 0, 1)  # Clamp pixel values to the [0, 1] range

    sr_img_pil = Image.fromarray((sr_img * 255).astype(np.uint8))
    sr_img_pil.save(sr_result_path)
    print(f"Super-resolved image saved to: {sr_result_path}")

# Batch-process images in a folder and generate a video
def test_images_and_generate_video(model, lr_folder_path, sr_result_folder_path, hr_folder_path):
    os.makedirs(sr_result_folder_path, exist_ok=True)
    best_psnr = 0
    best_file = ''
    psnrs = []
    file_nums = []

    # Prepare the video writer object
    frame_size = None
    out = None

    for filename in tqdm(os.listdir(lr_folder_path)):
        if filename.endswith((".png", ".jpg")):
            lr_image_path = os.path.join(lr_folder_path, filename)
            sr_result_path = os.path.join(sr_result_folder_path, filename)
            hr_image_path = os.path.join(hr_folder_path, filename)

            # Generate the super-resolved image
            test_model(model, lr_image_path, sr_result_path)

            # Read the generated super-resolved image and the high-resolution reference image
            sr_img = cv2.imread(sr_result_path)
            hr_img = cv2.imread(hr_image_path)


            if sr_img is not None and hr_img is not None:
                psnr, _ = calculate(sr_img, hr_img)
                file_num = int(re.findall(r'\d+', filename)[0])  # Assumes the filename contains a number
                psnrs.append(psnr)
                file_nums.append(file_num)

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_file = filename

                print(f"PSNR for {filename}: {psnr}")

    avg_psnr = np.mean(psnrs)  # Compute the average PSNR

    if out:
        out.release()  # Release the video writer object

    print(f"Best file: {best_file}")
    print(f"Best PSNR: {best_psnr}")
    print(f"Average PSNR: {avg_psnr}")

    return avg_psnr, best_psnr, best_file

def create_video_from_images(image_folder, video_output_path, fps=24):
    # Get all image filenames in the folder and sort them
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # Sort by the number embedded in the filename

    # Ensure there are image files in the folder
    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to determine the video frame size
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Create the video writer object
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # Iterate over all image files and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Skipping {image_path}, as it could not be read.")
            continue

        out.write(frame)

    # Release the video writer object
    out.release()
    print(f"Video saved to {video_output_path}")

# Main function (updated)
def main():

    model_path = config.SR_VIDEO_MODEL  # Path to the model file
    model_name = os.path.basename(model_path).split('.')[0]  # Get the model filename without extension
    sr_result_folder_path = os.path.join(config.SR_VIDEO_TMP_PREFIX, f'sr_results_{model_name}')  # Temporary results folder path
    lr_folder_path = config.SR_QINGT_LR  # Low-resolution image folder
    hr_folder_path = config.SR_QINGT_HR  # High-resolution image folder

    video_output_path = os.path.join(config.SR_VIDEO_OUT_PREFIX, f'{model_name}_video.mp4')  # Output video file path
    results_file_path = config.SR_RESULT_TXT_VIDEO  # Results file

    # Load the model
    model = load_model(model_path)

    # Generate images and compute PSNR
    avg_psnr, best_psnr, best_file = test_images_and_generate_video(model, lr_folder_path, sr_result_folder_path, hr_folder_path)

     # Generate video
    create_video_from_images(sr_result_folder_path,video_output_path,15)
    # Save results to file
    with open(results_file_path, 'w') as f:
        f.write(f"Model Name: {os.path.basename(model_path)}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Best PSNR: {best_psnr:.4f}\n")
        f.write(f"Best File: {best_file}\n")
        f.write(f"Video Output Path: {video_output_path}\n")
        print(f"Results saved to {results_file_path}")

if __name__ == "__main__":
    main()
