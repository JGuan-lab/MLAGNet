import cv2
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config

# image path
image_path = config.UI_RESIEZ_IN

# read image
img = cv2.imread(image_path)

# check if image was loaded correctly
if img is None:
    print(f"Image not found at {image_path}")
else:
    # get image width and height
    height, width = img.shape[:2]

    # compute new width and height
    new_width, new_height = width // 4, height // 4

    # resize image using new dimensions
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # save resized image
    cv2.imwrite(config.UI_RESIEZ_OUT, resized_img)
