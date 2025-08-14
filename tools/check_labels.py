import os, glob

IMG_DIR = r"C:\Users\pavan\OneDrive\Documents\Desktop\saketh project\data\yolo_dataset\images\train"
LBL_DIR = r"C:\Users\pavan\OneDrive\Documents\Desktop\saketh project\data\yolo_dataset\labels\train"

VAL_IMG_DIR = r"C:\Users\pavan\OneDrive\Documents\Desktop\saketh project\data\yolo_dataset\images\val"
VAL_LBL_DIR = r"C:\Users\pavan\OneDrive\Documents\Desktop\saketh project\data\yolo_dataset\labels\val"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm")
VALID_CLASSES = {0,1,2,3}

def collect_pairs(img_dir, lbl_dir):
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(VALID_EXTS)]
    missing, bad = [], []
    ok = 0
    for img in imgs:
        stem, _ = os.path.splitext(img)
        txt = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(txt):
            missing.append(img); continue
        with open(txt, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            bad.append((img, "empty .txt")); continue
        good_file = True
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                bad.append((img, f"line {i}: not 5 fields: {ln}")); good_file=False; continue
            try:
                cls = int(parts[0])
                x,y,w,h = map(float, parts[1:])
            except Exception as e:
                bad.append((img, f"line {i}: parse error: {ln}")); good_file=False; continue
            if cls not in VALID_CLASSES:
                bad.append((img, f"line {i}: invalid class {cls}")); good_file=False
            if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                bad.append((img, f"line {i}: coords not in [0,1]: {ln}")); good_file=False
        if good_file: ok += 1
    return imgs, ok, missing, bad

def report(split_name, img_dir, lbl_dir):
    imgs, ok, missing, bad = collect_pairs(img_dir, lbl_dir)
    print(f"\n=== {split_name} ===")
    print(f"Images found: {len(imgs)}")
    print(f"Good pairs  : {ok}")
    print(f"Missing .txt: {len(missing)}")
    print(f"Bad files   : {len(bad)}")
    if missing: print("  Missing (first 10):", missing[:10])
    if bad: 
        print("  Bad (first 10):")
        for r in bad[:10]: print("   ", r)

if __name__ == "__main__":
    report("TRAIN", IMG_DIR, LBL_DIR)
    if os.path.isdir(VAL_IMG_DIR):
        report("VAL", VAL_IMG_DIR, VAL_LBL_DIR)
