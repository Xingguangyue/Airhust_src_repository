# -*- coding: utf-8 -*-
"""鼠标点图取色：点一下，终端打印那个点的颜色。

用法：
    python3 get_orange.py                 # 打开 logo.png
    python3 get_orange.py logo.png        # 指定图片
    python3 get_orange.py logo.png 300 400   # 不开窗，直接打印 (300,400) 的颜色

窗口里：左键取色  右键清除标记  q 或 ESC 退出
"""
import sys

import cv2
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "logo.png"

img = cv2.imread(SRC)
if img is None:
    raise SystemExit("读不到 %s" % SRC)


def report(bgr, x, y, tag=""):
    """打印一个像素的 BGR / HSV / HEX。"""
    px = np.uint8([[bgr]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    print("%s(%4d,%4d)  BGR(%3d,%3d,%3d)  HSV(%3d,%3d,%3d)  #%02X%02X%02X" %
          (tag, x, y, b, g, r, h, s, v, r, g, b))


if len(sys.argv) >= 4:
    cx, cy = int(sys.argv[2]), int(sys.argv[3])
    x, y = int(cx), int(cy)
    patch = img[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).astype(np.float32)
    report(img[y, x], x, y, "像素")
    report(patch.mean(axis=0), x, y, "5x5均值")
    raise SystemExit(0)

MAXW, MAXH = 1100, 760
scale = min(1.0, MAXW / float(img.shape[1]), MAXH / float(img.shape[0]))
disp = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()
marks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        marks[:] = []
        return
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    ox, oy = int(round(x / scale)), int(round(y / scale))
    if not (0 <= ox < img.shape[1] and 0 <= oy < img.shape[0]):
        return
    patch = img[max(0, oy - 2):oy + 3, max(0, ox - 2):ox + 3].reshape(-1, 3).astype(np.float32)
    report(img[oy, ox], ox, oy, "像素")
    report(patch.mean(axis=0), ox, oy, "5x5均值")
    marks.append((x, y))
    cv2.drawMarker(disp, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)


WIN = "click to read color"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WIN, on_mouse)
print("左键取色，右键清除标记，q 退出。%s  %dx%d" % (SRC, img.shape[1], img.shape[0]))

while True:
    cv2.imshow(WIN, disp)
    k = cv2.waitKey(20) & 0xFF
    if k in (ord("q"), 27):
        break
cv2.destroyAllWindows()
