#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

TAKEOFF_Z = 1.5
QR_X,QR_Y = 1.0,0.0

state = State()
pos = None
qr_text = "" #解出来的内容
qr_stamp = 0.0
last_frame = None
bridge = CvBridge()
detector = cv2.QRCodeDetector()   # 全局建一次，别在回调里反复 new

def state_cb(msg):
    global state
    state = msg


def pose_cb(msg):
    global pos
    pos = msg.pose.position
    
def decode_qr(frame):
    """返回解出的字符串，解不出返回空串。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #先尝试同时解多个码
    try:
        res = detector.detectAndDecodeMulti(gray)
        if res[0]:
            for t in res[1]:
                if t:
                    return t
    except cv2.error:
        pass

    #不行在解一个码
    try:
        text, _, _ = detector.detectAndDecode(gray)
        if text:
            return text
    except cv2.error:
        pass

    return ""
    
    
def image_cb(msg):
    global qr_text, qr_stamp, last_frame
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    text = decode_qr(frame)
    if text:                       # 只在真解出来时才更新，失败不覆盖已有结果
        qr_text = text
    qr_stamp = rospy.get_time()
    last_frame = frame


def setpoint(x, y, z):
    sp = PoseStamped()
    sp.header.stamp = rospy.Time.now()
    sp.pose.position.x = x
    sp.pose.position.y = y
    sp.pose.position.z = z
    sp.pose.orientation.w = 1.0
    return sp

def go_to(x, y, z, err_max=0.15, timeout=20.0):
    target = setpoint(x, y, z)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        pub.publish(target)
        rate.sleep()
        if pos is None:
            continue
        d = ((pos.x - x) ** 2 + (pos.y - y) ** 2 + (pos.z - z) ** 2) ** 0.5
        if d < err_max:
            rospy.loginfo(f"arrived ({x:.2f},{y:.2f},{z:.2f})  误差 {d:.3f} m")
            return True
        if (rospy.Time.now() - t0).to_sec() > timeout:
            rospy.logwarn(f"timeout at ({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})")
            return False
    return False

rospy.init_node("task2_mission")

rospy.Subscriber("/mavros/state", State, state_cb, queue_size=10)
rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb, queue_size=10)
rospy.Subscriber("/usb_cam/image_raw", Image, image_cb, queue_size=1)

pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

rospy.wait_for_service("/mavros/cmd/arming")
arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
rospy.wait_for_service("/mavros/set_mode")
mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

rate = rospy.Rate(20)
target = setpoint(0.0, 0.0, TAKEOFF_Z)

while not rospy.is_shutdown() and not state.connected:
    rate.sleep()
rospy.loginfo("FCU connected")

for _ in range(100):
    pub.publish(target)
    rate.sleep()

while not rospy.is_shutdown() and state.mode != "OFFBOARD":
    mode_srv(custom_mode="OFFBOARD")
    pub.publish(target)
    rate.sleep()
rospy.loginfo("OFFBOARD ok")

while not rospy.is_shutdown() and not state.armed:
    arm_srv(True)
    pub.publish(target)
    rate.sleep()
rospy.loginfo("armed, climbing")

go_to(0.0, 0.0, TAKEOFF_Z)
go_to(QR_X, QR_Y, TAKEOFF_Z)

for _ in range(20):                        # 悬停 1 秒让画面稳定
    pub.publish(setpoint(QR_X, QR_Y, TAKEOFF_Z))
    rate.sleep()

if qr_text:
    rospy.loginfo(f"[QR]解出=  {qr_text}")
else:
    rospy.logwarn("[QR]未解出")

if last_frame is not None:
    cv2.imwrite("/tmp/shot_qr.jpg", last_frame)

rospy.loginfo("AUTO.LAND")
mode_srv(custom_mode="AUTO.LAND")

t0 = rospy.Time.now()
while not rospy.is_shutdown() and state.armed:
    if (rospy.Time.now() - t0).to_sec() > 30:
        rospy.logwarn("降落超时，仍 armed")
        break
    rate.sleep()
rospy.loginfo(f"结束 armed={state.armed}")



































