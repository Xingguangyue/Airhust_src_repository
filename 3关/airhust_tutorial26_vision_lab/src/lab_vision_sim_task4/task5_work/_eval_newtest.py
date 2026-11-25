# -*- coding: utf-8 -*-
"""在 newtest 上算检出率、平均置信度、与原图橙色框的 IoU。"""
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

SRC = "newtest"
W = "runs/detect/v1/weights/best.pt"
IMGSZ = 416
CONF = 0.25


def orange_box(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 80, 60), (30, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if m.sum() == 0:
        return None
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    amax = int(st[1:, cv2.CC_STAT_AREA].max())
    thr = max(60, int(amax * 0.02))
    x1 = y1 = 10 ** 9
    x2 = y2 = -1
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) < thr:
            continue
        x = int(st[i, cv2.CC_STAT_LEFT]); y = int(st[i, cv2.CC_STAT_TOP])
        w = int(st[i, cv2.CC_STAT_WIDTH]); h = int(st[i, cv2.CC_STAT_HEIGHT])
        x1 = min(x1, x); y1 = min(y1, y); x2 = max(x2, x + w); y2 = max(y2, y + h)
    if x2 < 0:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


model = YOLO(W)
files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
hit = 0
confs, ious, multi = [], [], 0
for f in files:
    im = cv2.imread(f)
    gt = orange_box(im)
    r = model(im, imgsz=IMGSZ, conf=CONF, iou=0.45, verbose=False)
    b = r[0].boxes
    if len(b) == 0:
        print("%-28s 漏检" % os.path.basename(f))
        continue
    if len(b) > 1:
        multi += 1
    hit += 1
    xyxy = b.xyxy.cpu().numpy()
    cf = b.conf.cpu().numpy()
    k = int(cf.argmax())
    x1, y1, x2, y2 = xyxy[k]
    pb = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    confs.append(float(cf[k]))
    if gt:
        ious.append(iou(pb, gt))

print()
print("检出 %d/%d  多框 %d 张" % (hit, len(files), multi))
if confs:
    print("置信度  min %.2f  中位 %.2f  max %.2f" % (min(confs), float(np.median(confs)), max(confs)))
if ious:
    print("与真值框 IoU  min %.2f  中位 %.2f  mean %.2f" % (min(ious), float(np.median(ious)), float(np.mean(ious))))
