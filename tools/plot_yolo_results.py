# tools/plot_yolo_results.py
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else r"yolo/yolov5/runs/train/yolo_gtsdb_4class7/results.csv"
out_dir  = os.path.dirname(csv_path)

df = pd.read_csv(csv_path)

# Helper to plot a list of (column, label)
def quick_plot(pairs, title, outname):
    plt.figure(figsize=(8,5))
    for col, lab in pairs:
        if col in df.columns:
            plt.plot(df['epoch'], df[col], label=lab)
    plt.xlabel("Epoch")
    plt.title(title)
    if any(col in df.columns for col,_ in pairs):
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, outname)
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)
    plt.close()

# Common YOLOv5 CSV column names (present if that metric was enabled)
quick_plot([
    ('train/box_loss', 'train box_loss'),
    ('train/obj_loss', 'train obj_loss'),
    ('train/cls_loss', 'train cls_loss'),
    ('val/box_loss',   'val box_loss'),
    ('val/obj_loss',   'val obj_loss'),
    ('val/cls_loss',   'val cls_loss'),
], "Loss curves", "results_losses.png")

quick_plot([
    ('metrics/precision',     'precision'),
    ('metrics/recall',        'recall'),
    ('metrics/mAP_0.5',       'mAP@0.5'),
    ('metrics/mAP_0.5:0.95',  'mAP@0.5:0.95'),
], "Validation metrics", "results_metrics.png")

# (Optional) learning rate curves if present
quick_plot([
    ('lr/pg0', 'lr pg0'),
    ('lr/pg1', 'lr pg1'),
    ('lr/pg2', 'lr pg2'),
], "Learning rates", "results_lrs.png")

print("Done. Look in:", out_dir)
