# Optimized-Detection-and-Classification-of-Road-Traffic-Signs-Using-YOLOv5
 This repository contains the code and configuration for a two-stage project:
1.	CNN Baseline for classification of cropped traffic signs.
2.	Optimized YOLOv5 pipeline for detection + classification in unconstrained real-world images.
Highlights
•	Custom dataset containing a large portion of self-captured traffic sign images under various lighting conditions, viewing angles, and backgrounds, plus supplemental weather-condition samples (e.g., snow, fog).
•	Manual annotations using LabelImg in YOLO format.
•	Fine-tuned YOLOv5 on 4 target classes:
•	keep_right, stop, speed_limit, no_entry
•	Clear commands for annotation, training, validation, and inference.
•	Fully reproducible training pipeline.

Repository Structure
project-root/
│
├── cnn_baseline/           # CNN classification scripts & notebooks
├── data/
│   ├── yolo_dataset/       # images/ and labels/ in YOLO format
│   ├── data.yaml           # YOLO dataset config
│
├── yolo/
│   └── yolov5/             # YOLOv5 cloned repo
│
├── tools/                  # helper scripts (visualization, plotting, etc.)
├── runs/                   # YOLOv5 training outputs (excluded in .gitignore)
├── requirements.txt        # root dependencies
└── README.md               # this file

Setup
# 1) Create environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2) Install root dependencies
pip install -r requirements.txt

# 3) Install YOLOv5 dependencies
pip install -r yolo/yolov5/requirements.txt

Dataset Preparation
The dataset follows the YOLO structure:
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
└── labels/
    ├── train/
    ├── val/
The data.yaml file should contain:
train: ../data/yolo_dataset/images/train
val: ../data/yolo_dataset/images/val

nc: 4
names: ['keep_right', 'stop', 'speed_limit', 'no_entry']

Annotation
Annotations were done using LabelImg.
# Install LabelImg
pip install labelImg

# Launch LabelImg
labelImg
•	Save in YOLO format.
•	Ensure class order in classes.txt matches data.yaml.

Training YOLOv5
cd yolo/yolov5

# Train from YOLOv5s pre-trained weights
python train.py --img 640 --batch 16 --epochs 80 \
  --data ../../data/data.yaml \
  --weights yolov5s.pt \
  --device cpu \
  --workers 0 \
  --name yolo_gtsdb_4class


Validation
python val.py \
  --weights runs/train/yolo_gtsdb_4class/weights/best.pt \
  --data ../../data/data.yaml \
  --imgsz 640 \
  --device cpu

Inference
python detect.py \
  --weights runs/train/yolo_gtsdb_4class/weights/best.pt \
  --source ../../data/yolo_dataset/images/val \
  --imgsz 640 \
  --conf-thres 0.25 \
  --device cpu
Predictions will be saved under yolo/yolov5/runs/detect/.

CNN Baseline (Initial Stage)
Before YOLOv5, a CNN model was trained to classify cropped traffic signs. While effective for centered images, it struggled with signs in cluttered or off-center real-world contexts, motivating the shift to YOLOv5.

Results
Class	Precision	Recall	mAP@0.5	mAP@0.5:0.95
keep_right	0.751	0.813	0.620	-
stop	0.685	0.714	0.534	-
speed_limit	0.698	0.724	0.565	-
no_entry	0.760	0.800	0.447	-

