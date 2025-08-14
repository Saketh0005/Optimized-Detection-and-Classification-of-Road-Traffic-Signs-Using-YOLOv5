import os

folders = [
    "data/GTSRB", "data/train", "data/test", "data/processed",
    "models", "notebooks", "src", "yolo/cropped", "app", "outputs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

with open("README.md", "w") as f:
    f.write("# Traffic Sign Recognition Using Deep Learning\n")

with open("requirements.txt", "w") as f:
    f.write("# Add package names here (e.g., tensorflow, opencv-python, etc.)\n")

print(" Project structure created.")
