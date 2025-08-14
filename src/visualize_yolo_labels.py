import os
import cv2
import matplotlib.pyplot as plt

# Paths
img_dir = r"data/yolo_dataset/images/train"
label_dir = r"data/yolo_dataset/labels/train"

# Updated class order from data.yaml
class_names = ['keep_right', 'stop', 'speed_limit', 'no_entry']

# Find label files
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

# Match labels to images (support .jpg, .png, .jpeg)
sample_files = []
for label_file in label_files:
    base_name = label_file.replace(".txt", "")
    for ext in [".jpg", ".png", ".jpeg"]:
        image_file = base_name + ext
        if os.path.exists(os.path.join(img_dir, image_file)):
            sample_files.append(image_file)
            break  # stop checking extensions if found

# Limit to first 10 samples
sample_files = sample_files[:10]

def draw_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x, y, bw, bh = map(float, parts)
            class_id = int(class_id)

            # Convert YOLO format to pixel coords
            x_center, y_center = x * w, y * h
            box_w, box_h = bw * w, bh * h
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # Draw rectangle + label
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_names[class_id], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Plot
plt.figure(figsize=(15, 10))
for i, fname in enumerate(sample_files):
    img_path = os.path.join(img_dir, fname)
    label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")

    img_with_boxes = draw_boxes(img_path, label_path)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img_with_boxes)
    plt.title(fname)
    plt.axis("off")

plt.tight_layout()
plt.show()
