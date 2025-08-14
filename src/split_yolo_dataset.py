import os
import random
import shutil

# Directories
img_dir = "data/yolo_dataset/images/train"
lbl_dir = "data/yolo_dataset/labels/train"
img_val_dir = "data/yolo_dataset/images/val"
lbl_val_dir = "data/yolo_dataset/labels/val"

# Create val directories if not exist
os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(lbl_val_dir, exist_ok=True)

# Parameters
val_ratio = 0.2
seed = 42

# Get all image files
all_images = [f for f in os.listdir(img_dir) if f.endswith(".ppm")]
random.seed(seed)
random.shuffle(all_images)

val_size = int(len(all_images) * val_ratio)
val_images = all_images[:val_size]

for fname in val_images:
    # Move image
    src_img = os.path.join(img_dir, fname)
    dst_img = os.path.join(img_val_dir, fname)
    shutil.move(src_img, dst_img)

    # Move label (if exists)
    label_name = fname.replace(".ppm", ".txt")
    src_lbl = os.path.join(lbl_dir, label_name)
    dst_lbl = os.path.join(lbl_val_dir, label_name)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print(f"Moved {val_size} images and labels to validation set.")
