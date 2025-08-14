import os
import numpy as np
from PIL import Image
import pandas as pd

# Path to training image folders
TRAIN_PATH = "data/GTSRB/Final_Training/Images/"

def load_training_data(image_size=(32, 32)):
    images = []
    labels = []

    # Loop through class folders
    for class_id in sorted(os.listdir(TRAIN_PATH)):
        class_dir = os.path.join(TRAIN_PATH, class_id)
        if not os.path.isdir(class_dir):
            continue

        # Loop through images in each class
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.ppm'):
                img_path = os.path.join(class_dir, img_name)

                # Load and resize image
                img = Image.open(img_path).resize(image_size)
                img = np.array(img)

                images.append(img)
                labels.append(int(class_id))  # Label is the folder name

    return np.array(images), np.array(labels)

def load_class_names(csv_path="data/GTSRB/signnames.csv"):
    df = pd.read_csv(csv_path)
    return dict(zip(df["ClassId"], df["SignName"]))
