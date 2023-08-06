import glob
from sklearn.model_selection import train_test_split
import os

# Define the new base path
base_path_resized = "/Users/karankinariwala/Library/CloudStorage/Dropbox/KARAN/5-Projects/RytleX-Computer-Vision-Project/data/ts/ts/resized"

# Get a list of all resized image files
image_files = glob.glob(os.path.join(base_path_resized, "*.jpg"))

# Split the data into training, validation and testing sets
train_files, test_files = train_test_split(
    image_files, test_size=0.2, random_state=42
)  # 80% training, 20% testing
train_files, val_files = train_test_split(
    train_files, test_size=0.25, random_state=42
)  # Further split training into 60% training, 20% validation

# Write the file paths to train.txt, val.txt and test.txt
with open("train_resized.txt", "w") as f:
    f.write("\n".join(train_files))

with open("val_resized.txt", "w") as f:
    f.write("\n".join(val_files))

with open("test_resized.txt", "w") as f:
    f.write("\n".join(test_files))
