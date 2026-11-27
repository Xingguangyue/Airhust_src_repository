# -*- coding: utf-8 -*-
"""给 snap_raw 里的真实截图自动打标，输出到 snap_lab/。

用的是和合成图完全相同的 orange_box()，保证两边标签定义一致。
"""
import os
import glob
import shutil

import cv2
import numpy as np

SRC = "snap_raw"
DST = "snap_lab"
PREVIEW = "snap_preview"


def orange_box(bgr):
    """和 2_make_dataset.py 里的同名函数逐字一致。"""
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
        x = int(st[i, cv2.CC_STAT_LEFT])
        y = int(st[i, cv2.CC_STAT_TOP])
        w = int(st[i, cv2.CC_STAT_WIDTH])
        h = int(st[i, cv2.CC_STAT_HEIGHT])
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x + w)
        y2 = max(y2, y + h)
    if x2 < 0:
        return None
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 12:
        return None
    return x1, y1, w, h


os.makedirs(DST, exist_ok=True)
os.makedirs(PREVIEW, exist_ok=True)

ok, bad = 0, 0
for f in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
    im = cv2.imread(f)
    if im is None:
        bad += 1
        continue
    H, W = im.shape[:2]
    gb = orange_box(im)
    if gb is None:
        bad += 1
        continue
    x, y, w, h = gb
    area_ratio = (w * h) / float(W * H)
    if area_ratio < 0.002 or area_ratio > 0.5:
        bad += 1
        continue

    name = os.path.splitext(os.path.basename(f))[0]
    shutil.copy(f, os.path.join(DST, name + ".jpg"))
    cx, cy = (x + w / 2.0) / W, (y + h / 2.0) / H
    with open(os.path.join(DST, name + ".txt"), "w") as fp:
        fp.write("0 %.6f %.6f %.6f %.6f\n" % (cx, cy, w / float(W), h / float(H)))

    pv = im.copy()
    cv2.rectangle(pv, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(PREVIEW, name + ".jpg"), pv)
    ok += 1

print("打标成功 %d 张，丢弃 %d 张" % (ok, bad))
