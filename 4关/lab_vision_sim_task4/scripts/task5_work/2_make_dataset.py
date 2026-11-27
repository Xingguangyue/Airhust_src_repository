# -*- coding: utf-8 -*-
"""合成训练集：logo 贴到背景上，自动算标签。
用法：
    python3 2_make_dataset.py 5      # 只造 5 张，用来检查标签
    python3 2_make_dataset.py 900    # 正式批量
"""

import os
import sys
import glob
import random

import cv2
import numpy as np

LOGO = "logo_cut_third.png"
BG_DIR = "backgrounds_clean"
OUT_IMG = "dataset/images/train"
OUT_LBL = "dataset/labels/train"
PREVIEW = "preview"

# 目标高度范围（像素）。实测：1.5m 高度下 logo 约 131px 高，所以这个区间要覆盖它
H_RANGE = (80, 200)
MAX_ROT = 30          # 旋转角度范围 ±30°（不要全 360°，见下）
IOU_MIN = 0.5         # 打标结果和粘贴框的 IoU 下限，低于此值丢弃该样本

random.seed(0)

def orange_box(bgr):
    """统一打标函数：返回橙色区域的外接框，零星小橙点直接丢弃。

    合成图和真实截图都用它，两边标签定义必须一致。
    """
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


def iou(a,b): #Intersection over Union
    ax,ay,aw,ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0 #发现0.0好可爱有没有 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / (aw * ah + bw * bh - inter)
        
def rotate_rgba(img,angle):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nW / 2 - w / 2
    M[1, 2] += nH / 2 - h / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
def trim(rgba):
    a = (rgba[:,:,3] > 10).astype(np.uint8) #根据第四个通道大小 将不透明的变成1 透明的变成0
    x,y,w,h = cv2.boundingRect(a)
    return rgba[y:y+h,x:x+w]

def paste(bg,rgba,x,y):
    h ,w = rgba.shape[:2]
    roi = bg[y:y + h, x:x + w]
    a = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    roi[:] = (rgba[:, :, :3].astype(np.float32) * a
              + roi.astype(np.float32) * (1 - a)).astype(np.uint8)

def photometric(img):
    """光度 + 噪声。只改像素值不改几何，所以标签不受影响。"""
    if random.random() < 0.6:
        img = np.clip(img.astype(np.float32) * random.uniform(0.6, 1.35)
                      + random.randint(-25, 25), 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.6, 1.4), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.5:
        sigma = random.uniform(1, 3)
        n = np.random.normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img
    
def main():
    n_imgs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LBL, exist_ok=True)
    os.makedirs(PREVIEW, exist_ok=True)

    logo = cv2.imread(LOGO, cv2.IMREAD_UNCHANGED)
    assert logo is not None and logo.shape[2] == 4, "logo_cut.png 不是 4 通道，回步骤 1"
    lh0, lw0 = logo.shape[:2]

    bgs = []
    for f in glob.glob(os.path.join(BG_DIR, "*")):
        im = cv2.imread(f)
        if im is not None:
            bgs.append(im)
    assert bgs, f"{BG_DIR} 里没有图片，先放几张背景（可以直接复制 logo.png 的灰度版占位）"

    ok, dropped = 0, 0
    while ok < n_imgs:
        bg = random.choice(bgs).copy()

        # 1) 缩放到目标像素尺寸
        th = random.randint(*H_RANGE)
        s = th / lh0
        logo_r = cv2.resize(logo, (max(1, int(lw0 * s)), th), interpolation=cv2.INTER_AREA)

        # 2) 旋转（±30°）。转完 trim，保证粘贴矩形就是内容矩形
        logo_r = trim(rotate_rgba(logo_r, random.uniform(-MAX_ROT, MAX_ROT)))

        # 3) 随机翻转
        if random.random() < 0.5:
            logo_r = cv2.flip(logo_r, 1)

        # 4) 贴上去（放不下就跳过这张）
        H, W = bg.shape[:2]
        th2, tw2 = logo_r.shape[:2]
        if th2 >= H or tw2 >= W:
            continue
        px = random.randint(0, W - tw2)
        py = random.randint(0, H - th2)
        paste(bg, logo_r, px, py)

        # 5) 用统一函数打标，再用粘贴矩形做一致性校验
        gb = orange_box(bg)
        if gb is None or iou(gb, (px, py, tw2, th2)) < IOU_MIN:
            dropped += 1
            continue

        # 6) 之后才加光度/噪声（只改像素值，不动几何）
        bg = photometric(bg)

        gx, gy, gw, gh = gb
        cx = (gx + gw / 2) / W
        cy = (gy + gh / 2) / H
        label = f"0 {cx:.6f} {cy:.6f} {gw/W:.6f} {gh/H:.6f}\n"

        name = f"syn_{ok:05d}"
        cv2.imwrite(os.path.join(OUT_IMG, name + ".jpg"), bg)
        with open(os.path.join(OUT_LBL, name + ".txt"), "w") as f:
            f.write(label)

        # 每 100 张（以及前 5 张）画一次框，供人眼检查
        if ok < 5 or ok % 100 == 0:
            pv = bg.copy()
            cv2.rectangle(pv, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(PREVIEW, name + ".jpg"), pv)

        ok += 1
        if ok % 50 == 0:
            print(f"已生成{ok}/{n_imgs}  丢弃{dropped}")

    print(f"完成：{ok} 张，丢弃{dropped} 张（打标与粘贴框 IoU <{IOU_MIN}）")


if __name__ == "__main__":
    main()






















        

