import os
from PIL import Image, ImageFile
import config

# Ignore truncated image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_images(input_folder, output_folder, scale_factor=0.125):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    # Calculate the new size
                    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                    
                    # Resize the image
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    # Save the resized image to the output folder
                    resized_img.save(output_path)
                    print(f"Resized and saved {filename}")
            except OSError as e:
                print(f"Cannot process file {filename}: {e}")


input_folder = config.SR_KAGGLE_TOTAL
output_folder = config.SR_KAGGLE_8X

resize_images(input_folder, output_folder)
