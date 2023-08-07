from dotenv import load_dotenv
import os
import cv2
import glob


def extract_data_id(path):
    return path.split("/")[-1].split(".")[0]


load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")

scale_percent = 60
image_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))

if not os.path.exists(os.path.join(DATA_DIR, "resized")):
    os.makedirs(os.path.join(DATA_DIR, "resized"))

for image in image_files:
    img = cv2.imread(image)
    (h, w, _) = img.shape
    new_width = int(w * scale_percent / 100)
    new_height = int(h * scale_percent / 100)
    new_dim = (new_width, new_height)
    resized_img = cv2.resize(img, new_dim)
    cv2.imwrite(
        os.path.join(DATA_DIR, "resized", extract_data_id(image) + ".jpg"),
        resized_img,
    )
print(new_dim)
