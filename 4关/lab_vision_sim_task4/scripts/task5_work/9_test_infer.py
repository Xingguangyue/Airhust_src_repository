# -*- coding: utf-8 -*-
"""离线推理：给一个目录里的图跑检测，画框输出到 infer_out/。

用法：
    python3 9_test_infer.py              # 默认跑 snap_raw
    python3 9_test_infer.py newtest      # 指定目录

imgsz 必须和训练时一致（416），否则精度会掉。
"""
import os
import sys
import glob

import cv2
from ultralytics import YOLO

WEIGHTS = "runs/detect/v1/weights/best.pt"
SRC = sys.argv[1] if len(sys.argv) > 1 else "snap_raw"
OUT = "infer_out"
CONF = 0.4
IMGSZ = 416

os.makedirs(OUT, exist_ok=True)
model = YOLO(WEIGHTS)

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
if not files:
    raise SystemExit("%s 里没有 jpg" % SRC)

hit = 0
for f in files:
    im = cv2.imread(f)
    r = model(im, imgsz=IMGSZ, conf=CONF, iou=0.45, verbose=False)
    boxes = r[0].boxes
    if len(boxes) > 0:
        hit += 1
        for xyxy, cf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, "%.2f" % cf, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT, os.path.basename(f)), im)

print("%d/%d 张检出  检出率 %.1f%%  结果在 %s/" %
      (hit, len(files), 100.0 * hit / len(files), OUT))
