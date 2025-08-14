import os
import cv2

# Paths
images_dir = "data/yolo_dataset/images/train"
labels_dir = "data/yolo_dataset/labels/train"
gt_file = "data/GTSDB/gt.txt"

os.makedirs(labels_dir, exist_ok=True)

# Target classes (we only care about these)
target_classes = {0, 14, 17, 33}

def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    # Convert absolute box coordinates to YOLO format (normalized)
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h

with open(gt_file, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split(";")
    filename = parts[0]
    x1, y1, x2, y2 = map(int, parts[1:5])
    class_id = int(parts[5])

    # Only process selected classes
    if class_id not in target_classes:
        continue

    img_path = os.path.join(images_dir, filename)
    if not os.path.exists(img_path):
        continue

    # Get image size
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Convert to YOLO format
    x_center, y_center, box_w, box_h = convert_to_yolo(x1, y1, x2, y2, w, h)

    # Save to .txt file
    label_file = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)

    with open(label_path, "a") as out:
        # Remap class ids to 0-3 (for 4 classes)
        new_class_id = {0:0, 14:1, 17:2, 33:3}[class_id]
        out.write(f"{new_class_id} {x_center} {y_center} {box_w} {box_h}\n")

print("YOLO label files created in:", labels_dir)
