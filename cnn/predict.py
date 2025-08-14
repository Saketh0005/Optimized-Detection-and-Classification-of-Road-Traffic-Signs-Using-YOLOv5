import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import sys
import os

# Class names – update if using all 43 later
class_names = {
    0: "Speed limit (30km/h)",
    1: "Stop",
    2: "No entry",
    3: "Keep right"
}

def load_and_prepare_image(img_path, target_size=(32, 32)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 32, 32, 3)
    return img


if __name__ == "__main__":
    # Allow image path from command line or hardcoded
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample.jpg"  # Change this to any image you want to test

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit()

    # Load and preprocess image
    image = load_and_prepare_image(image_path)

    # Load trained model
    model = load_model("models/best_cnn_model.h5")

    # Predict
    prediction = model.predict(image)
    predicted_class = prediction.argmax()

    # Show image and prediction
    img_rgb = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title(f"Prediction: {class_names[predicted_class]}")
    plt.axis("off")
    plt.show()
