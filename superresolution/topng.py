from PIL import Image
import os
import config

def convert_jpg_to_png(img_dir, output_dir):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .jpg files in the directory
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg"):
            # Build full file path
            img_path = os.path.join(img_dir, filename)
            # Open JPG file
            with Image.open(img_path) as img:
                # Get filename without extension
                base_filename = os.path.splitext(filename)[0]
                # Build PNG output file path
                output_path = os.path.join(output_dir, base_filename + ".png")
                # Save as PNG format
                img.save(output_path, "PNG")
                print(f"Saved {output_path}")

# Example usage
img_dir = config.SR_CRACK_IMG_DIR  # Input JPG folder path
output_dir = config.SR_CRACK_PNG_DIR  # Output PNG folder path

convert_jpg_to_png(img_dir, output_dir)
