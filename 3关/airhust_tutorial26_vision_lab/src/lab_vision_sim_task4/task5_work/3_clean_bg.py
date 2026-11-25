# -*- coding: utf-8 -*-
"""把 backgrounds/ 里的图片统一转成冷色单色调，输出到 backgrounds_clean/。

思路：照片的价值是纹理，污染源是颜色。保住纹理（灰度），把颜色换成
绿色~紫色区间（H=45~170），橙色区间 H=3~30 从此不可能出现。
"""
import cv2
import glob
import os
import random
import numpy as np

SRC = "backgrounds"
DST = "backgrounds_clean"
random.seed(7)

os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "*")))
done = 0
for f in files:
    im = cv2.imread(f)
    name = os.path.basename(f)
    if im is None:
        print("skip (unreadable):", name)
        continue
    if "solid_" in name:
        out = im
    else:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.convertScaleAbs(g, alpha=random.uniform(0.85, 1.15), beta=random.randint(-12, 12))
        hsv = np.zeros_like(im)
        hsv[:, :, 0] = random.randint(45, 170)
        hsv[:, :, 1] = random.randint(40, 80)
        hsv[:, :, 2] = g
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(DST, name), out)
    done += 1

bad = 0
for f in sorted(glob.glob(os.path.join(DST, "*"))):
    im = cv2.imread(f)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 80, 60), (30, 255, 255))
    if m.sum() > 0:
        bad += 1
        print("still orange:", os.path.basename(f))
print("输出 %d 张，含橙 %d 张" % (done, bad))
