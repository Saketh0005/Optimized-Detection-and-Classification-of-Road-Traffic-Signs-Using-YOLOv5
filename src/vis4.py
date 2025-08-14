import os
import cv2
import matplotlib.pyplot as plt

# Paths
img_dir = r"C:\Users\pavan\OneDrive\Documents\Desktop\saketh project\data\yolo_dataset\images\train"
label_dir = r"C:\Users\pavan\OneDrive\Documents\Desktop\saketh project\data\yolo_dataset\labels\train"


# Class names (order must match data.yaml)
class_names = ['keep_right', 'stop', 'speed_limit', 'no_entry']

# Collect matching image files for which a label exists
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
sample_files = []
for label_file in label_files:
    base = os.path.splitext(label_file)[0]
    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = base + ext
        if os.path.exists(os.path.join(img_dir, candidate)):
            sample_files.append(candidate)
            break

# Keep only the first 4
sample_files = sample_files[:4]

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

            # YOLO -> pixel coords
            x_c, y_c = x * w, y * h
            box_w, box_h = bw * w, bh * h
            x1, y1 = int(x_c - box_w / 2), int(y_c - box_h / 2)
            x2, y2 = int(x_c + box_w / 2), int(y_c + box_h / 2)

            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_names[class_id], (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Plot 2x2
cols, rows = 2, 2
plt.figure(figsize=(10, 10))
for i, fname in enumerate(sample_files):
    img_path = os.path.join(img_dir, fname)
    label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
    img_with_boxes = draw_boxes(img_path, label_path)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img_with_boxes)
    plt.title(fname)
    plt.axis("off")

# If fewer than 4 images, fill empty subplots with blank axes
for j in range(len(sample_files), rows * cols):
    plt.subplot(rows, cols, j + 1)
    plt.axis("off")

plt.tight_layout()
plt.show()
