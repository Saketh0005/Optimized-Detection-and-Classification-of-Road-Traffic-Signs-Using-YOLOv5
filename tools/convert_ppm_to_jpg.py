import os
from PIL import Image

SETS = [
    r"data/yolo_dataset/images/train",
    r"data/yolo_dataset/images/val",
]

def convert_dir(img_dir):
    if not os.path.isdir(img_dir):
        print("Missing:", img_dir); return
    count = 0
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(".ppm"):
            src = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            dst = os.path.join(img_dir, stem + ".jpg")
            try:
                im = Image.open(src).convert("RGB")
                im.save(dst, "JPEG", quality=95)
                os.remove(src)  # remove old .ppm
                count += 1
            except Exception as e:
                print("Failed:", src, "->", e)
    print(f"{img_dir}: converted {count} PPM to JPG")

if __name__ == "__main__":
    for d in SETS:
        convert_dir(d)
    print("Done.")
