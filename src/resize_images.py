from dotenv import load_dotenv
import os
import cv2
import glob


def extract_data_id(path):
    return path.split("/")[-1].split(".")[0]


load_dotenv()
IMAGE_DATA = os.environ.get("IMAGE_DATA")

image_files = glob.glob(os.path.join(IMAGE_DATA, "*.jpg"))

new_width = 832
new_height = 480

if not os.path.exists(os.path.join(IMAGE_DATA, "resized")):
    os.makedirs(os.path.join(IMAGE_DATA, "resized"))

for image in image_files:
    img = cv2.imread(image)
    resized_img = cv2.resize(img, (new_width, new_height))
    cv2.imwrite(
        os.path.join(IMAGE_DATA, "resized", extract_data_id(image) + ".jpg"),
        resized_img,
    )
