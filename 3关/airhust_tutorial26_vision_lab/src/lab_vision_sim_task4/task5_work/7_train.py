# -*- coding: utf-8 -*-
"""CPU 训练入口。imgsz=416 与后续推理保持一致。"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="logo.yaml",
    imgsz=416,
    epochs=60,
    batch=4,
    patience=20,
    workers=2,
    project=".",
    name="v1",
    exist_ok=True,
    plots=True,
)
