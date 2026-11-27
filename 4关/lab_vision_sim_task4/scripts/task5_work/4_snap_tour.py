#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动飞 9 个位置，每个位置截若干张下视画面。
跑之前：roslaunch lab_vision_sim_task4 task5.launch 已经起来
"""
import os
import rospy
import cv2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

OUT = "/home/airhust/airhust_tutorial2026_ws/3关/airhust_tutorial26_vision_lab/src/lab_vision_sim_task4/task5_work/snap_raw"
Z_LIST = [1.2, 1.5, 1.8]          # 三个高度 -> 三种目标像素尺寸
OFFS = [(-0.15, -0.15), (-0.15, 0.15), (0.15, -0.15), (0.15, 0.15)]   # 四个偏置

bridge = CvBridge()
latest = [None]
state = State()


def state_cb(msg):
    global state
    state = msg


def image_cb(msg):
    latest[0] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def setpoint(x, y, z):
    sp = PoseStamped()
    sp.header.stamp = rospy.Time.now()
    sp.pose.position.x = x
    sp.pose.position.y = y
    sp.pose.position.z = z
    sp.pose.orientation.w = 1.0
    return sp


def go_to(x, y, z, err_max=0.12, timeout=20.0):
    tgt = setpoint(x, y, z)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        pub.publish(tgt)
        rate.sleep()
        if pos is None:
            continue
        d = ((pos.x - x) ** 2 + (pos.y - y) ** 2 + (pos.z - z) ** 2) ** 0.5
        if d < err_max:
            return True
        if (rospy.Time.now() - t0).to_sec() > timeout:
            rospy.logwarn("timeout at (%.2f,%.2f,%.2f)" % (pos.x, pos.y, pos.z))
            return False
    return False


pos = None


def pose_cb(msg):
    global pos
    pos = msg.pose.position


def shoot(tag, n=6):
    """在一个位置连续存 n 张"""
    for i in range(n):
        pub.publish(setpoint(*(TGT[0], TGT[1], TGT[2])))
        rate.sleep()
        if latest[0] is not None:
            cv2.imwrite(os.path.join(OUT, f"{tag}_{i:02d}.jpg"), latest[0])


rospy.init_node("snap_tour")
os.makedirs(OUT, exist_ok=True)
rospy.Subscriber("/mavros/state", State, state_cb, queue_size=10)
rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb, queue_size=10)
rospy.Subscriber("/usb_cam/image_raw", Image, image_cb, queue_size=1)
pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
rospy.wait_for_service("/mavros/cmd/arming")
arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
rospy.wait_for_service("/mavros/set_mode")
mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
rate = rospy.Rate(20)

TGT = [0.0, 0.0, 1.5]

while not rospy.is_shutdown() and not state.connected:
    rate.sleep()
rospy.loginfo("FCU connected")

for _ in range(100):
    pub.publish(setpoint(0, 0, 1.5))
    rate.sleep()

while not rospy.is_shutdown() and state.mode != "OFFBOARD":
    mode_srv(custom_mode="OFFBOARD")
    pub.publish(setpoint(0, 0, 1.5))
    rate.sleep()
rospy.loginfo("OFFBOARD ok")

while not rospy.is_shutdown() and not state.armed:
    arm_srv(True)
    pub.publish(setpoint(0, 0, 1.5))
    rate.sleep()
rospy.loginfo("armed")

go_to(0.0, 0.0, 1.5)

for z in Z_LIST:
    for dx, dy in OFFS:
        TGT[0], TGT[1], TGT[2] = 1.0 + dx, 0.0 + dy, z
        if go_to(*TGT):
            shoot(f"z{int(z*100)}_x{int(dx*100):+d}_y{int(dy*100):+d}", n=5)
            rospy.loginfo("shot at z=%.2f dx=%+.2f dy=%+.2f" % (z, dx, dy))

rospy.loginfo("AUTO.LAND")
mode_srv(custom_mode="AUTO.LAND")
t0 = rospy.Time.now()
while not rospy.is_shutdown() and state.armed:
    if (rospy.Time.now() - t0).to_sec() > 30:
        break
    rate.sleep()
rospy.loginfo("done")
