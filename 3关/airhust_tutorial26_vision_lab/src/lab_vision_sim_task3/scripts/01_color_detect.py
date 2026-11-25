#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务一：颜色识别。"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
OUTPUT = os.path.join(ROOT, "output")
IMAGE_PATH = os.path.join(ASSETS, "color_targets.png")
SAVE_PATH = os.path.join(OUTPUT, "color_result.png")
MIN_AREA = 500
os.makedirs(OUTPUT, exist_ok=True)

COLOR_RANGES = {
    "red": ((0, 0, 150), (90, 90, 255)),
    "green": ((0, 150, 0), (90, 255, 90)),
    "blue": ((150, 0, 0), (255, 90, 90)),
}

def color_mask(image, lower, upper):
    mask = cv2.inRange(image, np.array(lower), np.array(upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_blobs(mask, min_area=MIN_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        blobs.append((int(cx), int(cy), int(radius), int(area)))
    return blobs

def annotate(image, results):
    canvas = image.copy()
    for item in results:
        cx, cy = item["center"]
        cv2.circle(canvas, (cx, cy), item["radius"], (0, 0, 0), 2)
        cv2.circle(canvas, (cx, cy), 4, (0, 0, 0), -1)
        label = "%s (%d,%d)" % (item["color"], cx, cy)
        cv2.putText(canvas, label, (cx - 55, cy - item["radius"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas

image = cv2.imread(IMAGE_PATH)
results = []
for name, (lower, upper) in COLOR_RANGES.items():
    mask = color_mask(image, lower, upper)
    blobs = find_blobs(mask, MIN_AREA)
    for (cx, cy, radius, area) in blobs:
        results.append({
            "center": (cx, cy),
            "radius": radius,
            "color": name,
        })

print("共检测到 %d 个色块" % len(results))
canvas = annotate(image, results)
cv2.imwrite(SAVE_PATH, canvas)
print("标注图已保存到", SAVE_PATH)
