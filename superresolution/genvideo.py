import os
import cv2
import re
import config
def create_video_from_images(image_folder, video_output_path, fps=15):
    # get all image filenames in the folder and sort them
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # sort by the number in the filename

    # ensure there are image files in the folder
    if not images:
        print("No images found in the folder.")
        return

    # read the first image to determine the video frame size
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # create a video writer object
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # iterate over all image files and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Skipping {image_path}, as it could not be read.")
            continue

        out.write(frame)

    # release the video writer object
    out.release()
    print(f"Video saved to {video_output_path}")

# main function
def main():
    image_folder = config.SR_GENVIDEO_GT  # replace with your image folder path
    video_output_path = config.SR_GENVIDEO_OUT  # replace with the path where you want to save the video
    fps = 15  # set video frame rate

    create_video_from_images(image_folder, video_output_path, fps)

if __name__ == "__main__":
    main()
