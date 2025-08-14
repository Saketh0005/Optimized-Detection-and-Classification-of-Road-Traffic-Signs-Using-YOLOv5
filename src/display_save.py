import os
import cv2

# Paths to your dataset
images_path = r"data/yolo_dataset/images/train"
labels_path = r"data/yolo_dataset/labels/train"
output_path = r"yolo_labels_preview"

os.makedirs(output_path, exist_ok=True)

# Class names (same order as your data.yaml)
class_names = ['keep_right', 'stop', 'speed_limit', 'no_entry']

for img_file in os.listdir(images_path):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(images_path, img_file)
        label_file = os.path.join(labels_path, os.path.splitext(img_file)[0] + ".txt")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image {img_path}")
            continue
        h, w, _ = img.shape

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x_center, y_center, bw, bh = map(float, parts)

                    # Convert YOLO coords to pixel coords
                    x_center *= w
                    y_center *= h
                    bw *= w
                    bh *= h

                    x1 = int(x_center - bw / 2)
                    y1 = int(y_center - bh / 2)
                    x2 = int(x_center + bw / 2)
                    y2 = int(y_center + bh / 2)

                    # Draw rectangle and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{class_names[int(cls_id)]}"
                    cv2.putText(img, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save and show
        save_path = os.path.join(output_path, img_file)
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")

        # Uncomment if you want to see live preview
        # cv2.imshow("YOLO Labels", img)
        # if cv2.waitKey(0) & 0xFF == 27:  # ESC to break
        #     break

# cv2.destroyAllWindows()
