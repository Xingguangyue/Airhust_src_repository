#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五关 · logo 检测全流程（单文件，7 个子命令）

标签从哪来（这一版的核心改动）：
  合成图   贴图矩形就是真值。data/logo_cut.png 的不透明外接框 == 整幅图，
           所以贴上去的 (px, py, tw, th) 原样就是 YOLO 要的框。
           不去画面里找颜色，误差 0，也不会因为背景有橙色而丢样本。
  真实截图 没有这个便利，两条路：
             label --manual   自己拖框，绝对真值，60 张约 3 分钟
             label --auto     用已训练权重出伪标签（先拿纯合成数据训 v0）

子命令：
  cut     白底 logo -> 透明 RGBA（按 min(R,G,B) 反算 alpha，不碰色相）
  bg      背景池检查。默认原色不动，--tint 才洗成冷色单色
  synth   合成训练集
  label   真实截图打标（--manual / --auto 二选一）
  pack    汇总 dataset（合成 + 真实 + 负样本），顺手写 logo.yaml
  train   CPU 训练
  infer   离线推理，画框落到 out/preview/

目录：
  data/logo.png          data/logo_cut.png
  data/bg/               背景原图（往这里扔照片）
  data/bg/clean/         去色版（只有 bg --tint 才生成）
  data/synth/            合成图 a.jpg + a.txt（自动）
  data/snap/             真实截图 a.jpg + a.txt（自动拍 / 手工标）
  data/snap_test/        分布外测试截图（自动拍）
  data/dataset/          images/ labels/ logo.yaml（全自动）
  out/preview/           抽检图、推理图、伪标签预览
  out/runs/              训练产物（不进 git）
  weights/best.pt        仓库自带的权重，clone 下来就能直接跑
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
PREVIEW = OUT / "preview"
RUNS = OUT / "runs"

LOGO_RAW = DATA / "logo.png"
LOGO_CUT = DATA / "logo_cut.png"
BG_DIR = DATA / "bg"
BG_CLEAN = BG_DIR / "clean"
SYNTH = DATA / "synth"
SNAP = DATA / "snap"
SNAP_TEST = DATA / "snap_test"
DS = DATA / "dataset"

IMGSZ = 416
CAM_W, CAM_H = 640, 480  # 仿真相机分辨率。合成图统一成这个尺寸，跟真实截图一致
HR_RANGE = (0.18, 0.45)  # logo 高度占画面高度的比例。实测 z=1.0~2.1 对应 19.6%~41%
MAX_ROT = 30             # 旋转 ±30°。全 360° 会让模型学不会上下方向
PRETRAINED = HERE / "yolov8n.pt"


def _best():
    """权重。优先用自己训练出来的最新一版，没有就退回仓库自带的那份。

    out/runs/    训练产物，不进 git
    weights/     仓库自带，clone 下来就能直接跑飞行，不用先训一遍
    """
    if RUNS.exists():
        cands = list(RUNS.rglob("best.pt"))
        if cands:
            return str(max(cands, key=lambda p: p.stat().st_mtime))
    bundled = HERE / "weights" / "best.pt"
    if bundled.exists():
        return str(bundled)
    return str(RUNS / "v1" / "weights" / "best.pt")


# ---------------------------------------------------------------- 几何工具

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def rotate_rgba(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nW / 2 - w / 2
    M[1, 2] += nH / 2 - h / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def trim(rgba, thresh=10):
    """裁到不透明内容的外接框。贴图前必须做一次，这样粘贴矩形 == 内容矩形。"""
    a = (rgba[:, :, 3] > thresh).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(a)
    return rgba[y:y + h, x:x + w]


def paste(bg, rgba, x, y):
    h, w = rgba.shape[:2]
    roi = bg[y:y + h, x:x + w]
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (rgba[:, :, :3].astype(np.float32) * a
              + roi.astype(np.float32) * (1 - a)).astype(np.uint8)


def photometric(img):
    """只改像素值不改几何，所以标签不受影响。"""
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
        n = np.random.normal(0, random.uniform(1, 3), img.shape)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img


def draw_box(im, x, y, w, h, text=None, color=(0, 255, 0)):
    cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    if text:
        cv2.putText(im, text, (int(x), max(14, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def yolo_line(x, y, w, h, W, H):
    """YOLO 标签：类别 cx cy w h，后四个归一化到 0~1，空格分隔。"""
    return "0 %.6f %.6f %.6f %.6f\n" % ((x + w / 2.0) / W, (y + h / 2.0) / H,
                                        w / float(W), h / float(H))


def _need(path, hint):
    if not Path(path).exists():
        sys.exit("缺 %s\n  -> %s" % (path, hint))


# ---------------------------------------------------------------- 子命令

def cmd_cut(a):
    """白底 logo -> 透明 PNG。

    alpha = 255 - min(R,G,B)：纯白三通道都是 255 -> 全透明；纯色 min 低 -> 不透明。
    这一步只处理模板本身，跟检测无关；后面的流程里再也不出现色相判断。
    """
    _need(LOGO_RAW, "把原始的 logo.png 放到 data/ 下")
    im = cv2.imread(str(LOGO_RAW))
    b, g, r = cv2.split(im.astype(np.int16))
    alpha = (255 - np.minimum(np.minimum(b, g), r)).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    x, y, w, h = cv2.boundingRect(alpha)
    rgba = np.dstack([im[y:y + h, x:x + w], alpha[y:y + h, x:x + w]])
    LOGO_CUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(LOGO_CUT), rgba)
    print("抠图 -> %s  %dx%d（原图 %dx%d）"
          % (LOGO_CUT.relative_to(HERE), w, h, im.shape[1], im.shape[0]))


def cmd_bg(a):
    """背景池检查，可选去色。默认原样不动。

    背景保留原色是有意的：合成标签来自贴图矩形，不经过颜色判断，所以背景里
    有橙色也不会标错；反过来，背景有橙色能堵掉「见橙就框」这条捷径，逼模型看形状。
    加 --tint 才把背景洗成冷色单色（当初是为了让颜色打标不失败才加的）。
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [p for p in sorted(BG_DIR.glob("*"))
             if p.is_file() and p.suffix.lower() in exts]
    if not files:
        sys.exit("data/bg/ 里没有图片，扔几张照片进去")

    if not a.tint:
        from collections import Counter
        c = Counter()
        for p in files:
            im = cv2.imread(str(p))
            if im is not None:
                c[(im.shape[1], im.shape[0])] += 1
        print("背景池 %d 张，保持原色不做处理" % len(files))
        for (w, h), n in c.most_common():
            print("  %dx%d  %d 张" % (w, h, n))
        print("  合成时统一裁成 %dx%d，尺寸不齐没关系" % (CAM_W, CAM_H))
        return

    BG_CLEAN.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    n = 0
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            print("  跳过（读不出）:", p.name)
            continue
        if "solid_" not in p.name:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=random.uniform(0.85, 1.15),
                                       beta=random.randint(-12, 12))
            hsv = np.zeros_like(im)
            hsv[:, :, 0] = random.randint(45, 170)
            hsv[:, :, 1] = random.randint(40, 80)
            hsv[:, :, 2] = gray
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(str(BG_CLEAN / p.name), im)
        n += 1
    print("去色 -> %s  %d 张" % (BG_CLEAN.relative_to(HERE), n))


def _load_logo():
    logo = cv2.imread(str(LOGO_CUT), cv2.IMREAD_UNCHANGED)
    if logo is None or logo.ndim != 3 or logo.shape[2] != 4:
        sys.exit("data/logo_cut.png 不是 4 通道\n  -> python3 logo_pipeline.py cut")
    t = trim(logo)
    if t.shape != logo.shape:
        print("  模板 alpha 外接框不是整幅，已裁到 %dx%d" % (t.shape[1], t.shape[0]))
    return t


def _fit_bg(im):
    """随机裁成 4:3 再缩到相机分辨率。

    背景照片从 640x480 到 1920x1440 都有，不先统一的话同一个「80px 的 logo」
    在 1920 高的图上只占 4% 画面、在 640 图上占 42%，模型学到的尺度是乱的。
    随机裁剪还顺便把 30 张背景变成了几乎无限的背景。
    """
    h, w = im.shape[:2]
    tgt = CAM_W / float(CAM_H)
    if w / float(h) > tgt:                 # 太宽，裁左右
        nw = max(1, int(h * tgt))
        x = random.randint(0, w - nw)
        im = im[:, x:x + nw]
    else:                                  # 太高，裁上下
        nh = max(1, int(w / tgt))
        y = random.randint(0, h - nh)
        im = im[y:y + nh, :]
    return cv2.resize(im, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)


def _load_bgs():
    src = BG_CLEAN if any(BG_CLEAN.glob("*")) else BG_DIR
    bgs = []
    for p in sorted(src.glob("*")):
        if p.is_file():
            im = cv2.imread(str(p))
            if im is not None and im.shape[0] > 240:
                bgs.append(im)
    if not bgs:
        sys.exit("data/bg/ 里没有可用背景（至少 240 像素高）")
    return bgs


def cmd_synth(a):
    """合成训练集。标签 = 贴图矩形，不经过任何颜色识别。"""
    logo = _load_logo()
    bgs = _load_bgs()
    SYNTH.mkdir(parents=True, exist_ok=True)
    if a.check:
        PREVIEW.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)

    lh, lw = logo.shape[:2]
    ok, skip = 0, 0
    while ok < a.n:
        bg = _fit_bg(random.choice(bgs))
        H, W = bg.shape[:2]

        th = int(H * random.uniform(*HR_RANGE))
        s = th / float(lh)
        lr = cv2.resize(logo, (max(1, int(lw * s)), th), interpolation=cv2.INTER_AREA)
        lr = trim(rotate_rgba(lr, random.uniform(-MAX_ROT, MAX_ROT)))
        if random.random() < 0.5:
            lr = cv2.flip(lr, 1)

        th2, tw2 = lr.shape[:2]
        if th2 >= H or tw2 >= W:
            skip += 1
            continue
        px = random.randint(0, W - tw2)
        py = random.randint(0, H - th2)
        paste(bg, lr, px, py)
        bg = photometric(bg)

        name = "syn_%05d" % ok
        cv2.imwrite(str(SYNTH / (name + ".jpg")), bg)
        (SYNTH / (name + ".txt")).write_text(yolo_line(px, py, tw2, th2, W, H))

        if ok < a.check:
            pv = bg.copy()
            draw_box(pv, px, py, tw2, th2, "%dx%d" % (tw2, th2))
            cv2.imwrite(str(PREVIEW / (name + ".jpg")), pv)

        ok += 1
        if a.n >= 200 and ok % 100 == 0:
            print("  %d/%d" % (ok, a.n))

    print("合成 %d 张 -> %s（放不下跳过 %d 张）"
          % (ok, SYNTH.relative_to(HERE), skip))


def _mouse_label(im):
    """拖一个框。Enter 保存 / 空格跳过 / Esc 退出。返回 (x,y,w,h) 或 None 或 'quit'。"""
    snap = im.copy()
    st = {"p0": None, "p1": None, "drag": False}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["p0"], st["p1"], st["drag"] = (x, y), (x, y), True
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]:
            st["p1"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            st["p1"], st["drag"] = (x, y), False

    win = "label: drag box | Enter=save  Space=skip  Esc=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 720)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            show = snap.copy()
            if st["p0"] and st["p1"]:
                cv2.rectangle(show, st["p0"], st["p1"], (0, 255, 0), 2)
            cv2.imshow(win, show)
            k = cv2.waitKey(20) & 0xFF
            if k == 13:                       # Enter
                if st["p0"] and st["p1"]:
                    x1, x2 = sorted([st["p0"][0], st["p1"][0]])
                    y1, y2 = sorted([st["p0"][1], st["p1"][1]])
                    if x2 - x1 >= 8 and y2 - y1 >= 8:
                        return x1, y1, x2 - x1, y2 - y1
            elif k == 32:                     # 空格
                return None
            elif k == 27:                     # Esc
                return "quit"
    finally:
        cv2.destroyWindow(win)


def cmd_label(a):
    """真实截图打标。--manual 拖框（真值），--auto 用权重出伪标签。"""
    files = sorted(SNAP.glob("*.jpg"))
    if not files:
        sys.exit("data/snap/ 里没有 jpg\n  -> 先 roslaunch task5.launch，再 python3 snap_tour.py")
    print("data/snap/ 共 %d 张" % len(files))

    if a.manual:
        done, skipped = 0, 0
        for i, p in enumerate(files):
            im = cv2.imread(str(p))
            if im is None:
                continue
            print("[%d/%d] %s" % (i + 1, len(files), p.name))
            r = _mouse_label(im)
            if r == "quit":
                break
            if r is None:
                skipped += 1
                continue
            H, W = im.shape[:2]
            (p.with_suffix(".txt")).write_text(yolo_line(r[0], r[1], r[2], r[3], W, H))
            done += 1
        cv2.destroyAllWindows()
        print("手工标 %d 张，跳过 %d 张" % (done, skipped))
        return

    _need(a.weights, "先拿纯合成数据训一版：synth 900 && pack --neg 0 && train")
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    PREVIEW.mkdir(parents=True, exist_ok=True)
    ok, miss = 0, []
    for p in files:
        im = cv2.imread(str(p))
        r = model(im, imgsz=IMGSZ, conf=a.conf, iou=0.45, verbose=False)[0]
        H, W = im.shape[:2]
        if len(r.boxes) == 0:
            miss.append(p.name)
            continue
        j = int(r.boxes.conf.argmax())        # 单目标场景只取最自信那个
        x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
        cf = float(r.boxes.conf[j])
        p.with_suffix(".txt").write_text(
            yolo_line(x1, y1, x2 - x1, y2 - y1, W, H))
        pv = im.copy()
        draw_box(pv, x1, y1, x2 - x1, y2 - y1, "%.2f" % cf)
        cv2.imwrite(str(PREVIEW / ("lbl_" + p.name)), pv)
        ok += 1

    print("模型打标 %d 张，漏检 %d 张" % (ok, len(miss)))
    if miss:
        print("  漏检：%s" % ", ".join(miss[:10]))
    print("  !! 伪标签是模型的自己的看法，一定要抽检 %s/lbl_*.jpg"
          % PREVIEW.relative_to(HERE))


def cmd_pack(a):
    """汇总数据集。验证集只放真实截图 —— 合成图上分数虚高，没参考价值。"""
    assert DS.parent == DATA, "dataset 路径异常，拒绝删除"
    for d in (DS / "images", DS / "labels"):
        if d.exists():
            shutil.rmtree(d)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DS / sub).mkdir(parents=True, exist_ok=True)

    def put(img, text, split, name):
        shutil.copy(str(img), str(DS / "images" / split / (name + ".jpg")))
        (DS / "labels" / split / (name + ".txt")).write_text(text)

    random.seed(42)

    snaps = [p for p in sorted(SNAP.glob("*.jpg")) if p.with_suffix(".txt").exists()]
    random.shuffle(snaps)
    n_val = int(len(snaps) * 0.3)
    for i, p in enumerate(snaps):
        put(p, p.with_suffix(".txt").read_text(),
            "val" if i < n_val else "train", "snap_" + p.stem)
    print("真实截图 %d 张：val %d / train %d"
          % (len(snaps), n_val, len(snaps) - n_val))
    if n_val == 0:
        print("  !! 验证集为空，train 出来的指标不能信")

    syn = sorted(SYNTH.glob("*.jpg"))
    for p in syn:
        t = p.with_suffix(".txt")
        put(p, t.read_text() if t.exists() else "", "train", p.stem)
    print("合成图 %d 张 -> train" % len(syn))

    pool = [p for p in sorted(BG_CLEAN.glob("*")) if p.is_file()] or \
           [p for p in sorted(BG_DIR.glob("*")) if p.is_file()]
    random.shuffle(pool)
    n_neg = 0
    for p in pool:
        if n_neg >= a.neg:
            break
        im = cv2.imread(str(p))
        if im is None:
            continue
        # 负样本也裁成相机分辨率：仿真里「画面里没有 logo」就是这种下视截图
        cv2.imwrite(str(DS / "images" / "train" / ("neg_%04d.jpg" % n_neg)), _fit_bg(im))
        (DS / "labels" / "train" / ("neg_%04d.txt" % n_neg)).write_text("")
        n_neg += 1
    print("负样本 %d 张（空标签，已裁到 %dx%d）" % (n_neg, CAM_W, CAM_H))

    (DS / "logo.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: logo\n" % DS)

    print("-" * 44)
    for split in ("train", "val"):
        ni = len(list((DS / "images" / split).glob("*")))
        nl = len(list((DS / "labels" / split).glob("*")))
        print("%-5s 图 %4d  标签 %4d  %s"
              % (split, ni, nl, "OK" if ni == nl else "!! 数量不匹配"))


def cmd_train(a):
    _need(PRETRAINED, "把 yolov8n.pt 放到 task5_work/ 下（虚拟机连不上网，不会自动下）")
    _need(DS / "logo.yaml", "先跑 pack")
    from ultralytics import YOLO
    model = YOLO(str(PRETRAINED))
    model.train(data=str(DS / "logo.yaml"), imgsz=IMGSZ, epochs=a.epochs,
                batch=a.batch, patience=20, workers=a.workers,
                project=str(RUNS), name="v1", exist_ok=True, plots=True)
    print("权重 -> %s" % _best())


def cmd_infer(a):
    _need(a.weights, "先跑 train")
    src = Path(a.src)
    files = sorted(src.glob("*.jpg"))
    if not files:
        sys.exit("%s 里没有 jpg" % src)
    from ultralytics import YOLO
    PREVIEW.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(a.weights))

    hit, multi = 0, 0
    for p in files:
        im = cv2.imread(str(p))
        r = model(im, imgsz=IMGSZ, conf=a.conf, iou=0.45, verbose=False)[0]
        n = len(r.boxes)
        if n:
            hit += 1
            if n > 1:
                multi += 1
            for j in range(n):
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                draw_box(im, x1, y1, x2 - x1, y2 - y1,
                         "%.2f" % float(r.boxes.conf[j]))
        cv2.imwrite(str(PREVIEW / p.name), im)

    print("%d/%d 检出（%.1f%%）  多框 %d 张  imgsz=%d conf=%.2f  -> %s"
          % (hit, len(files), 100.0 * hit / len(files), multi, IMGSZ, a.conf,
             PREVIEW.relative_to(HERE)))


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(
        description="第五关 logo 检测全流程（合成标签不靠颜色）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="典型流程：\n"
               "  cut -> bg -> synth 900 --check 5 -> label --manual\n"
               "  -> pack -> train -> infer data/snap_test")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cut", help="白底 logo -> 透明 PNG").set_defaults(func=cmd_cut)

    p = sub.add_parser("bg", help="背景池检查（默认保持原色）")
    p.add_argument("--tint", action="store_true",
                   help="把背景洗成冷色单色，去掉橙色")
    p.set_defaults(func=cmd_bg, tint=False)

    p = sub.add_parser("synth", help="合成训练集")
    p.add_argument("n", type=int, nargs="?", default=5)
    p.add_argument("--check", type=int, default=0, help="顺便输出前 N 张带框图")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("label", help="真实截图打标")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manual", action="store_true", help="拖框手标（真值）")
    g.add_argument("--auto", action="store_true", help="用权重出伪标签")
    p.add_argument("--weights", default=_best())
    p.add_argument("--conf", type=float, default=0.5)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("pack", help="汇总数据集")
    p.add_argument("--neg", type=int, default=120, help="负样本张数")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("train", help="CPU 训练")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0,
                   help="CPU 训练保持 0，多进程 DataLoader 在 CPU 上容易死锁")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("infer", help="离线推理")
    p.add_argument("src", nargs="?", default=str(SNAP_TEST))
    p.add_argument("--weights", default=_best())
    p.add_argument("--conf", type=float, default=0.4)
    p.set_defaults(func=cmd_infer)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
