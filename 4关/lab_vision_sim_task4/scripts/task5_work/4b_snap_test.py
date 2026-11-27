#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试集截图：故意飞到训练时没覆盖的高度和偏移，检验模型泛化。

和 4_snap_tour.py 的区别只有三处：输出目录 newtest/、高度 1.0~2.1（训练用 1.2~1.8）、
偏移 ±0.24（训练用 ±0.15）。

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

OUT = "/home/airhust/airhust_tutorial2026_ws/3关/airhust_tutorial26_vision_lab/src/lab_vision_sim_task4/task5_work/newtest"
Z_LIST = [1.0, 2.1]
OFFS = [(-0.24, -0.24), (-0.24, 0.24), (0.24, -0.24), (0.24, 0.24)]

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


pos = None


def pose_cb(msg):
    global pos
    pos = msg.pose.position


def go_to(x, y, z, err_max=0.12, timeout=25.0):
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


def shoot(tag, n=5):
    for i in range(n):
        pub.publish(setpoint(TGT[0], TGT[1], TGT[2]))
        rate.sleep()
        if latest[0] is not None:
            cv2.imwrite(os.path.join(OUT, "%s_%02d.jpg" % (tag, i)), latest[0])


rospy.init_node("snap_test")
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
            shoot("t_z%d_x%+d_y%+d" % (int(z * 100), int(dx * 100), int(dy * 100)), n=5)
            rospy.loginfo("shot at z=%.2f dx=%+.2f dy=%+.2f" % (z, dx, dy))

rospy.loginfo("AUTO.LAND")
mode_srv(custom_mode="AUTO.LAND")
t0 = rospy.Time.now()
while not rospy.is_shutdown() and state.armed:
    if (rospy.Time.now() - t0).to_sec() > 30:
        break
    rate.sleep()
rospy.loginfo("done")
