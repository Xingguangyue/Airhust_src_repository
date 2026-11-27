#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动截图：飞到一组已知位姿，每个位姿连拍 n 张下视画面。

  python3 snap_tour.py            训练用  z=1.2/1.5/1.8  偏移 ±0.15   -> data/snap/
  python3 snap_tour.py --test     测试用  z=1.0/2.1      偏移 ±0.24   -> data/snap_test/

跑之前先起仿真：roslaunch lab_vision_sim_task4 task5.launch
两个 profile 只差高度、偏移、输出目录 —— 测试集的位姿故意落在训练区间之外。
"""

import argparse
import os
import sys

import cv2
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))

PROFILES = {
    "train": dict(z=[1.2, 1.5, 1.8], off=0.15, out=os.path.join(HERE, "data", "snap")),
    "test":  dict(z=[1.0, 2.1],     off=0.24, out=os.path.join(HERE, "data", "snap_test")),
}
BASE_X, BASE_Y = 1.0, 0.0          # logo 大概在世界 (1.0, 0.0) 附近
SHOTS = 6                          # 每个位置连拍几张

state = State()
pos = None
latest = [None]
bridge = CvBridge()


def state_cb(msg):
    global state
    state = msg


def pose_cb(msg):
    global pos
    pos = msg.pose.position


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


def go_to(x, y, z, err_max=0.12, timeout=25.0):
    """发同一个目标点直到到位或超时。"""
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
            rospy.logwarn("超时：停在 (%.2f, %.2f, %.2f)" % (pos.x, pos.y, pos.z))
            return False
    return False


def shoot(tag, xyz, n):
    for i in range(n):
        pub.publish(setpoint(*xyz))
        rate.sleep()
        if latest[0] is not None:
            cv2.imwrite(os.path.join(OUT, "%s_%02d.jpg" % (tag, i)), latest[0])


def wait_until(pred, timeout, what, hint):
    t0 = rospy.Time.now()
    while not rospy.is_shutdown() and not pred():
        if (rospy.Time.now() - t0).to_sec() > timeout:
            rospy.logerr("%ds 还没 %s —— %s" % (timeout, what, hint))
            sys.exit(1)
        rate.sleep()


pub = None
rate = None
OUT = ""


def main():
    global pub, rate, OUT

    ap = argparse.ArgumentParser(description="自动截图")
    ap.add_argument("--test", action="store_true",
                    help="分布外测试集：z=1.0/2.1，偏移 ±0.24，输出 data/snap_test/")
    ap.add_argument("--n", type=int, default=SHOTS, help="每个位置几张")
    ap.add_argument("--start-z", type=float, default=1.5, help="起飞/复位高度")
    args = ap.parse_args()

    prof = PROFILES["test" if args.test else "train"]
    OUT = prof["out"]
    os.makedirs(OUT, exist_ok=True)
    zs, off = prof["z"], prof["off"]
    offs = [(-off, -off), (-off, off), (off, -off), (off, off)]

    rospy.init_node("snap_test" if args.test else "snap_tour")
    rospy.Subscriber("/mavros/state", State, state_cb, queue_size=10)
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb, queue_size=10)
    rospy.Subscriber("/usb_cam/image_raw", Image, image_cb, queue_size=1)
    pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
    rate = rospy.Rate(20)

    rospy.wait_for_service("/mavros/cmd/arming")
    arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
    rospy.wait_for_service("/mavros/set_mode")
    mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    z0 = args.start_z
    rospy.loginfo("等飞控连接……")
    wait_until(lambda: state.connected, 20, "连上飞控",
               "检查 task5.launch 里的 mavros 起没起")

    for _ in range(100):                     # OFFBOARD 要求先有一串 setpoint
        pub.publish(setpoint(0, 0, z0))
        rate.sleep()

    # 切模式和解锁都得边发指令边等：光读状态什么都不会发生
    rospy.loginfo("切 OFFBOARD……")
    t0 = rospy.Time.now()
    while not rospy.is_shutdown() and state.mode != "OFFBOARD":
        if (rospy.Time.now() - t0).to_sec() > 20:
            rospy.logerr("20s 没进 OFFBOARD —— PX4 要先收到 setpoint 才肯切")
            sys.exit(1)
        mode_srv(custom_mode="OFFBOARD")
        pub.publish(setpoint(0, 0, z0))
        rate.sleep()

    rospy.loginfo("解锁……")
    t0 = rospy.Time.now()
    while not rospy.is_shutdown() and not state.armed:
        if (rospy.Time.now() - t0).to_sec() > 15:
            rospy.logerr("15s 没解锁 —— 检查 COM_RC_IN_MODE 是不是 1")
            sys.exit(1)
        arm_srv(True)
        pub.publish(setpoint(0, 0, z0))
        rate.sleep()
    rospy.loginfo("已解锁")

    go_to(0.0, 0.0, z0)

    total, shot = len(zs) * len(offs), 0
    for z in zs:
        for dx, dy in offs:
            xyz = (BASE_X + dx, BASE_Y + dy, z)
            shot += 1
            if go_to(*xyz):
                tag = "z%d_x%+d_y%+d" % (int(z * 100), int(dx * 100), int(dy * 100))
                if args.test:
                    tag = "t_" + tag
                shoot(tag, xyz, args.n)
                rospy.loginfo("[%d/%d] z=%.2f dx=%+.2f dy=%+.2f"
                              % (shot, total, z, dx, dy))
            else:
                rospy.logwarn("[%d/%d] 没到位，跳过" % (shot, total))

    rospy.loginfo("AUTO.LAND")
    mode_srv(custom_mode="AUTO.LAND")
    t0 = rospy.Time.now()
    while not rospy.is_shutdown() and state.armed:
        if (rospy.Time.now() - t0).to_sec() > 30:
            break
        rate.sleep()
    rospy.loginfo("完成，图在 %s" % OUT)


if __name__ == "__main__":
    main()
