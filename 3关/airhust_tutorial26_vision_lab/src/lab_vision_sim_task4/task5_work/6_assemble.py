# -*- coding: utf-8 -*-
"""把合成图 + 真实截图组装成 YOLO 数据集。验证集必须全部来自真实截图。"""
import os
import glob
import random
import shutil

random.seed(42)
ROOT = "dataset"
SNAP = "snap_lab"
BG_DIR = "backgrounds_clean"
N_NEG = 120

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(ROOT, sub), exist_ok=True)


def put(img_path, lbl_lines, split, name):
    shutil.copy(img_path, os.path.join(ROOT, "images", split, name + ".jpg"))
    with open(os.path.join(ROOT, "labels", split, name + ".txt"), "w") as f:
        f.writelines(lbl_lines)


snaps = sorted(glob.glob(os.path.join(SNAP, "*.jpg")))
random.shuffle(snaps)
n_val = max(1, int(len(snaps) * 0.3))
for i, f in enumerate(snaps):
    name = "snap_" + os.path.splitext(os.path.basename(f))[0]
    lbl = f.replace(".jpg", ".txt")
    lines = []
    if os.path.exists(lbl):
        lines = open(lbl).readlines()
    put(f, lines, "val" if i < n_val else "train", name)
print("真实截图：%d 张，其中验证集 %d 张" % (len(snaps), n_val))

bgs = [f for f in glob.glob(os.path.join(BG_DIR, "*")) if os.path.isfile(f)]
random.shuffle(bgs)
picked = bgs[:N_NEG]
for i, f in enumerate(picked):
    put(f, [], "train", "neg_%04d" % i)
print("负样本：%d 张" % len(picked))

for split in ("train", "val"):
    ni = len(glob.glob(os.path.join(ROOT, "images", split, "*.jpg")))
    nl = len(glob.glob(os.path.join(ROOT, "labels", split, "*.txt")))
    print("%s: 图 %d  标签 %d  %s" % (split, ni, nl, "OK" if ni == nl else "!!! 数量不匹配"))
