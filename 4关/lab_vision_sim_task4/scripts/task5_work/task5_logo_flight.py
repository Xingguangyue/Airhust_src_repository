#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看见 logo -> 对准 -> 降落。

三个角色，频率不同，互不阻塞：
    image_cb     30Hz  只存最新一帧
    detect_timer 5Hz   独立线程推理，写 det
    主循环       20Hz  发 setpoint，不能断
"""
import glob
import os

import rospy
import cv2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from ultralytics import YOLO

# 权重路径自动挑最新的一版，别再写死
_TW = os.path.dirname(os.path.abspath(__file__))
_c = glob.glob(os.path.join(_TW, "out", "runs", "**", "best.pt"), recursive=True)
if _c:
    WEIGHTS = max(_c, key=os.path.getmtime)
else:                      # 没有训练产物就用仓库自带的
    _b = os.path.join(_TW, "weights", "best.pt")
    WEIGHTS = _b if os.path.exists(_b) else os.path.join(
        _TW, "out", "runs", "v1", "weights", "best.pt")
Z = 1.5
IMGSZ = 416          # 必须和训练时一致
CONF = 0.4
GAIN = 0.8           # 误差(米) -> 速度(m/s)
VMAX = 0.35          # 最大速度 m/s
LEAD = 0.25          # 预瞄时间 s：目标始终落在当前位置前方一小段
DEAD = 0.03          # 归一化偏差小于这个算「对准」
HOLD = 1.0           # 连续对准多久才降落（秒）
FRESH = 1.0          # 检出信息超过这么久算过期（秒）
LOST = 2.0           # 连续丢检这么久，回搜索点
SEARCH_X, SEARCH_Y = 1.0, 0.0

state = State()
pos = None
latest = None
det = None           # (conf, eu, ev)  归一化偏差
det_stamp = 0.0
bridge = CvBridge()
model = YOLO(WEIGHTS)


def state_cb(msg):
    global state
    state = msg


def pose_cb(msg):
    global pos
    pos = msg.pose.position


def image_cb(msg):
    """只做最轻的事，绝不在回调里推理。"""
    global latest
    latest = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def detect_timer(ev):
    """独立线程：推理一次，把最靠画面中心的框的偏差存下来。"""
    global det, det_stamp
    if latest is None:
        return
    r = model(latest, imgsz=IMGSZ, conf=CONF, iou=0.45, verbose=False)
    boxes = r[0].boxes
    if len(boxes) == 0:
        return
    xyxy = boxes.xyxy.cpu().numpy()
    cf = boxes.conf.cpu().numpy()
    h, w = latest.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0
    best = None
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
        if best is None or d < best[3]:
            best = (float(cf[i]), (cx - cx0) / w, (cy - cy0) / h, d)
    if best is not None:
        det = (best[0], best[1], best[2])
        det_stamp = rospy.get_time()


def setpoint(x, y, z):
    sp = PoseStamped()
    sp.header.stamp = rospy.Time.now()
    sp.pose.position.x = x
    sp.pose.position.y = y
    sp.pose.position.z = z
    sp.pose.orientation.w = 1.0
    return sp


def go_to(x, y, z, err_max=0.15, timeout=20.0):
    tgt = setpoint(x, y, z)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        pub.publish(tgt)
        rate.sleep()
        if pos is None:
            continue
        d = ((pos.x - x) ** 2 + (pos.y - y) ** 2 + (pos.z - z) ** 2) ** 0.5
        if d < err_max:
            rospy.loginfo("arrived (%.2f,%.2f,%.2f)  误差 %.3f m" % (x, y, z, d))
            return True
        if (rospy.Time.now() - t0).to_sec() > timeout:
            rospy.logwarn("timeout at (%.2f,%.2f,%.2f)" % (pos.x, pos.y, pos.z))
            return False
    return False


rospy.init_node("task5_logo")
rospy.Subscriber("/mavros/state", State, state_cb, queue_size=10)
rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb, queue_size=10)
rospy.Subscriber("/usb_cam/image_raw", Image, image_cb, queue_size=1)
pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

rospy.wait_for_service("/mavros/cmd/arming")
arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
rospy.wait_for_service("/mavros/set_mode")
mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

rate = rospy.Rate(20)
rospy.Timer(rospy.Duration(0.2), detect_timer)

home = setpoint(0.0, 0.0, Z)

while not rospy.is_shutdown() and not state.connected:
    rate.sleep()
rospy.loginfo("FCU connected")

for _ in range(100):
    pub.publish(home)
    rate.sleep()

while not rospy.is_shutdown() and state.mode != "OFFBOARD":
    mode_srv(custom_mode="OFFBOARD")
    pub.publish(home)
    rate.sleep()
rospy.loginfo("OFFBOARD ok")

arm_t0 = rospy.Time.now()
while not rospy.is_shutdown() and not state.armed:
    if (rospy.Time.now() - arm_t0).to_sec() > 15:
        rospy.logerr("15s 仍未解锁")
        raise SystemExit(1)
    arm_srv(True)
    pub.publish(home)
    rate.sleep()
rospy.loginfo("armed, climbing")

go_to(0.0, 0.0, Z)
go_to(1.0, 0.0, Z)

tx, ty = SEARCH_X, SEARCH_Y
last_stamp = -1.0
aligned_since = None
lost_since = None
n = 0
while not rospy.is_shutdown():
    if pos is None:
        pub.publish(setpoint(tx, ty, Z))
        rate.sleep()
        continue

    age = rospy.get_time() - det_stamp
    if det is not None and age < FRESH:
        lost_since = None
        c, eu, ev = det
        if det_stamp != last_stamp:
            last_stamp = det_stamp
            # 相似三角形：归一化像素偏差 x 高度 = 真实水平偏移（米）
            # 相机朝下装：图像右 = 世界 -y，图像下 = 世界 -x（实测确定，别照抄）
            vx = max(-VMAX, min(VMAX, -ev * Z * GAIN))
            vy = max(-VMAX, min(VMAX, -eu * Z * GAIN))
            # 比例式：目标 = 当前位置 + 前方一小段，不平滑累加
            tx = pos.x + vx * LEAD
            ty = pos.y + vy * LEAD
            if n % 6 == 0:
                rospy.loginfo("[LOGO] conf=%.2f  eu=%+.3f ev=%+.3f  v=(%+.2f,%+.2f)  目标=(%.2f,%.2f)"
                              % (c, eu, ev, vx, vy, tx, ty))
        if abs(eu) < DEAD and abs(ev) < DEAD:
            if aligned_since is None:
                aligned_since = rospy.get_time()
        else:
            aligned_since = None
    else:
        aligned_since = None
        if lost_since is None:
            lost_since = rospy.get_time()
        elif rospy.get_time() - lost_since > LOST:
            tx, ty = SEARCH_X, SEARCH_Y
            if n % 20 == 0:
                rospy.logwarn("丢检 %.1fs，回搜索点 (%.1f,%.1f)" % (LOST, SEARCH_X, SEARCH_Y))

    tx = max(SEARCH_X - 1.2, min(SEARCH_X + 1.2, tx))
    ty = max(SEARCH_Y - 1.2, min(SEARCH_Y + 1.2, ty))
    pub.publish(setpoint(tx, ty, Z))
    n += 1

    if aligned_since is not None and rospy.get_time() - aligned_since > HOLD:
        rospy.loginfo("已对准 %.1fs，位置 (%.2f,%.2f)，开始降落" % (HOLD, tx, ty))
        break
    rate.sleep()

res = mode_srv(custom_mode="AUTO.LAND")
if not res.mode_sent:
    rospy.logwarn("AUTO.LAND 没被飞控接受")

t0 = rospy.Time.now()
while not rospy.is_shutdown() and state.armed:
    if (rospy.Time.now() - t0).to_sec() > 30:
        rospy.logwarn("降落超时，仍 armed")
        break
    rate.sleep()
rospy.loginfo("结束 armed=%s  最终位置 (%.2f,%.2f)" % (state.armed, tx, ty))
