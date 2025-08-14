import os

def clean_unlabeled_images(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png", ".ppm"))]
    label_files = [f.replace(".txt", "") for f in os.listdir(label_dir) if f.endswith(".txt")]

    deleted = 0
    for image in image_files:
        name, _ = os.path.splitext(image)
        if name not in label_files:
            os.remove(os.path.join(image_dir, image))
            deleted += 1

    print(f"Deleted {deleted} unlabeled images from {image_dir}")

# Example usage
clean_unlabeled_images("data/yolo_dataset/images/train", "data/yolo_dataset/labels/train")
clean_unlabeled_images("data/yolo_dataset/images/val", "data/yolo_dataset/labels/val")
