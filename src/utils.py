import glob
import pandas as pd
import numpy as np
import cv2

IMAGE = "image"
LABEL = "label"
LINE_THICKNESS_FACTOR = 0.002
COLOR_BBOX = (255, 0, 0)


def extract_data_id(path: str) -> str:
    """
    CUSTOM FUNCTION - NEEDS TO BE MODIFIED BASED ON FILENAME
    Extracts the data id from a given path.

    Parameters:
    ----
    path (str): The path from which to extract the data id.

    Returns:
    ----
    str: The extracted data id.
    """
    return path.split("/")[-1].split(".")[0]


def find_corresponding_image_or_label(
    data_id: str,
    find_image_or_label: str,
    image_paths: list[str],
    label_paths: list[str],
) -> list[str]:
    """
    Finds the corresponding image or label for a given data id.

    Parameters:
    ----
    data_id (str): The data id for which to find the corresponding image or label.
    find_image_or_label (str): Specifies whether to find an image or a label. Should be either "image" or "label".
    image_paths (list): A list of image paths.
    label_paths (list): A list of label paths.

    Returns:
    ----
    list: A list of image or label ids that correspond to the given data id.
    """
    if find_image_or_label not in [IMAGE, LABEL]:
        raise ValueError(f"find_image_or_label should be either '{IMAGE}' or '{LABEL}'")
    if find_image_or_label == IMAGE:
        return [image_id for image_id in image_paths if data_id in image_id]
    elif find_image_or_label == LABEL:
        return [label_id for label_id in label_paths if data_id in label_id]


def convert_bbox_to_opencv_format(
    label: pd.DataFrame, image_width: int, image_height: int
) -> tuple[int, int]:
    """
    Converts bounding box coordinates to OpenCV format.

    Parameters:
    ----
    label (DataFrame): A DataFrame containing class label and bounding box coordinates.
    image_width (int): The width of the image.
    image_height (int): The height of the image.

    Returns:
    ----
    tuple: A tuple containing the top left and bottom right coordinates of the bounding box.
    """
    label.loc[:, "x_center"] = label.loc[:, "x_center"] * image_width
    label.loc[:, "y_center"] = label.loc[:, "y_center"] * image_height
    label.loc[:, "w"] = label.loc[:, "w"] * image_width
    label.loc[:, "h"] = label.loc[:, "h"] * image_height
    label.loc[:, "x1"] = label.loc[:, "x_center"] - (label.loc[:, "w"] / 2)
    label.loc[:, "y1"] = label.loc[:, "y_center"] - (label.loc[:, "h"] / 2)
    label.loc[:, "x2"] = label.loc[:, "x_center"] + (label.loc[:, "w"] / 2)
    label.loc[:, "y2"] = label.loc[:, "y_center"] + (label.loc[:, "h"] / 2)
    c1, c2 = (int(label.loc[0, "x1"]), int(label.loc[0, "y1"])), (
        int(label.loc[0, "x2"]),
        int(label.loc[0, "y2"]),
    )
    return c1, c2


def compute_image_with_bbox(data_id: str, classes: list) -> np.ndarray:
    """
    Computes an image with a bounding box.

    Parameters:
    ----
    data_id (str): The data id for which to compute the image.
    classes (list): List of classes

    Returns:
    ----
    ndarray: An image with a bounding box.
    """
    image_path = find_corresponding_image_or_label(
        data_id, IMAGE, image_paths, label_paths
    )[0]
    label_path = find_corresponding_image_or_label(
        data_id, LABEL, image_paths, label_paths
    )[0]

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    label = pd.read_csv(label_path, sep=" ", header=None)
    label.columns = ["class", "x_center", "y_center", "w", "h"]

    c1, c2 = convert_bbox_to_opencv_format(label, width, height)

    line_thickness = round(
        LINE_THICKNESS_FACTOR * (image.shape[0] + image.shape[1]) / 2 + 1
    )

    image_with_bbox = cv2.rectangle(
        image, c1, c2, color=COLOR_BBOX, thickness=line_thickness
    )

    font_thickness = max(line_thickness - 1, 1)

    cv2.getTextSize(
        classes[label.loc[0, "class"]],
        0,
        fontScale=line_thickness / 3,
        thickness=font_thickness,
    )[0]

    image_with_bbox = cv2.putText(
        image_with_bbox,
        classes[label.loc[0, "class"]],
        (c1[0], c1[1] - 2),
        0,
        line_thickness / 3,
        [255, 0, 0],
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )

    return image_with_bbox


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt

    load_dotenv()
    IMAGE_DATA = os.environ.get("IMAGE_DATA")
    DATA_DIR = os.environ.get("DATA_DIR")

    image_paths = glob.glob(os.path.join(IMAGE_DATA, "*.jpg"))
    label_paths = glob.glob(os.path.join(IMAGE_DATA, "*.txt"))

    with open(os.path.join(DATA_DIR, "classes.names"), "r") as f:
        classes = f.read()
    classes = classes.split("\n")[0:4]

    data_id = extract_data_id(label_paths[0])

    image_with_bbox = compute_image_with_bbox(data_id, classes)

    plt.imshow(image_with_bbox)
    plt.show()
