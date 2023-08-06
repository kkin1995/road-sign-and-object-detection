import pandas as pd
from dotenv import load_dotenv
import os
import glob
from sklearn.model_selection import train_test_split

load_dotenv()
image_data_dir = os.environ.get("IMAGE_DATA")
data_dir = os.environ.get("DATA_DIR")

image_file_paths = glob.glob(os.path.join(image_data_dir, "resized", "*.jpg"))

image_ids = [image_id.split("/")[-1].split(".")[0] for image_id in image_file_paths]

with open(os.path.join(image_data_dir, "resized", "classes.names")) as f:
    data = f.read()
classes = data.split("\n")[0:4]

dataset = []
n_data = len(image_ids)
for i in range(n_data):
    dataset_dict = {}
    dataset_dict["image_id"] = image_ids[i]
    df = pd.read_csv(image_data_dir + image_ids[i] + ".txt", header=None, sep=" ")
    dataset_dict["class_id"] = df.iloc[0, 0]
    dataset_dict["class"] = classes[df.iloc[0, 0]]
    dataset.append(dataset_dict)

df = pd.DataFrame(dataset)

# Split the data into training, validation and testing sets
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42
)  # 80% training, 20% testing
train_df, val_df = train_test_split(
    train_df, test_size=0.25, random_state=42
)  # Further split training into 60% training, 20% validation

df.to_csv(os.path.join(data_dir, "complete_dataset.csv"), index=None)
train_df.to_csv(os.path.join(data_dir, "train_dataset.csv"), index=None)
val_df.to_csv(os.path.join(data_dir, "val_dataset.csv"), index=None)
test_df.to_csv(os.path.join(data_dir, "test_dataset.csv"), index=None)
