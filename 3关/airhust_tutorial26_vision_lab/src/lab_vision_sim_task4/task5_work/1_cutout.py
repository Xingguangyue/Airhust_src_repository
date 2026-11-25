# -*- coding: utf-8 -*-

#抠出 logo：白底 -> 透明，输出 logo_cut.png (RGBA)

import cv2
import numpy as np

SRC = "logo.png"
DST = "logo_cut_third.png"

im = cv2.imread(SRC)
assert im is not None,"logo.png 没读到，🍀"

# 1.转成HSV,BGR太拉了
hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv,(3,80,60),(30,255,255)) #用get_orange测出了来HSV是(13,255,255)

#2. 闭运算：把断开的笔画连起来
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

#3.开运算去掉噪点
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

#4.取全部橙色像素的外接框
x,y,w,h = cv2.boundingRect(mask)
print(f"橙色外接框: x={x} y={y} w={w} h={h}  占原图 {w/1024:.1%} x {h/1024:.1%}")
print(f"橙色像素占全图 {mask.sum()/255/(1024*1024):.2%}") #白色是255，所以要除255

#5.
cut = im[y:y+h,x:x+w].copy() #彩色图
alpha = mask[y:y+h,x:x+w].copy() #黑白图
rgba = np.dstack([cut, alpha]) #4通道 奥～原来这就是rgba图像

cv2.imwrite(DST, rgba)
print("已写出", DST, rgba.shape)
