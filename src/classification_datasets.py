import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import os


class TrafficSignDataset(Dataset):
    def __init__(
        self, data_csv_file, root_dir, transform=None, target_transform=None, **kwargs
    ):
        self.data_csv_file = data_csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.df = pd.read_csv(
            os.path.join(self.root_dir, self.data_csv_file), dtype={"image_id": str}
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        class_id = self.df.loc[idx, "class_id"]
        image_id += ".jpg"
        image = cv2.imread(os.path.join(self.root_dir, "resized", image_id))

        image = torch.Tensor(image).permute(2, 0, 1)

        return image, class_id


if __name__ == "__main__":
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt

    load_dotenv()
    data_dir = os.environ.get("DATA_DIR")
    training_data = TrafficSignDataset("train_dataset.csv", data_dir)

    fig = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        image, label = training_data[sample_idx]
        image = image.permute(1, 2, 0).numpy()
        classes = list(set(training_data.df.loc[:, "class"].values))
        fig.add_subplot(rows, cols, i)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
        plt.title(classes[label])
    plt.show()
