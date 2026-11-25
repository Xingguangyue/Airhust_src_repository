#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2                                              
import numpy as np                                    
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image                      
from cv_bridge import CvBridge                         
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

TAKEOFF_Z = 1.5 #目标高度
PATCH = 20 #取样半宽

COLOR_RANGES = {                                       # 用字典来存储不同颜色以及对应的BGR范围
# 下界 150 -> 100
    "red":   ((0, 0, 100),   (90, 90, 255)),
    "green": ((0, 100, 0),   (90, 255, 90)),
    "blue":  ((100, 0, 0),   (255, 90, 90)),
}

MIN_AREA  = 500
state = State()
pos = None #pos初始为None
color_stamp = 0.0   # 上次收到图像的秒数。
last_frame = None                           # 最新判色结果
bridge = CvBridge()                                    # 将ros格式的转成python格式


def state_cb(msg):
    global state
    state = msg


def pose_cb(msg):
    global pos
    pos = msg.pose.position

#！！！！判断颜色！！！！
color_now = ("none", 1e9, (0, 0), 0)     #

def judge_color(frame):
    h, w = frame.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0
    best = ("none", 1e9, (0, 0), 0)      # 名字, 偏移, 圆心, 面积
    for name, (lo, hi) in COLOR_RANGES.items():
        mask = cv2.inRange(frame, np.array(lo), np.array(hi))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue
            (cx, cy), _ = cv2.minEnclosingCircle(c)
            d = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
            if d < best[1]:              # 取离画面中心最近的
                best = (name, d, (int(cx), int(cy)), int(area))
    return best
def image_cb(msg):                                     
    global color_now,color_stamp,last_frame # 全局变量
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") #ROS图片消息转成CV图片
    color_now = judge_color(frame)
    color_stamp = rospy.get_time()
    last_frame = frame


def setpoint(x, y, z): #打点
    sp = PoseStamped()
    sp.header.stamp = rospy.Time.now()
    sp.pose.position.x = x
    sp.pose.position.y = y
    sp.pose.position.z = z
    sp.pose.orientation.w = 1.0
    return sp


def go_to(x, y, z, err_max=0.15, timeout=20.0): #运动
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


rospy.init_node("task1_mission")

rospy.Subscriber("/mavros/state", State, state_cb, queue_size=10)
rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb, queue_size=10)
rospy.Subscriber("/usb_cam/image_raw", Image, image_cb, queue_size=1)   

pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

rospy.wait_for_service("/mavros/cmd/arming")
arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
rospy.wait_for_service("/mavros/set_mode")
mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

# 原地起飞 套公式==================================================
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


#=================================套公式

    
WAYPOINTS = [
    ("red",   1.0, -1.0),
    ("green", 1.0,  0.0),
    ("blue",  1.0,  1.0),
]
go_to(0.0, 0.0, TAKEOFF_Z)                 # 先原地爬升

detected = {}
for expected,x,y in WAYPOINTS:
    if not go_to(x,y,TAKEOFF_Z):
        rospy.logwarn(f"未到达{expected},跳过")
        continue
        
    for _ in range(20): #悬停一秒
        pub.publish(setpoint(x,y,TAKEOFF_Z))
        rate.sleep()

    age = rospy.get_time() - color_stamp
    if age > 1.0:
    	rospy.logwarn(f"图像{age:.1f}s未更新，判定结果不可信")
    name, dist, center, area = color_now
    detected[expected] = name
    flag = "OK" if name == expected else "MISMATCH"
    h, w = last_frame.shape[:2]
    c = last_frame[h // 2, w // 2].tolist()
    rospy.loginfo(f"[{expected}] 看到{name} 面积={area} 偏移={dist:.0f}px 圆心={center} {flag}")
    cv2.imwrite(f"/tmp/shot_{expected}.jpg", last_frame)

    
rospy.loginfo(f"结果 = {detected}")


#降落
rospy.loginfo("AUTO.LAND")
mode_srv(custom_mode = "AUTO.LAND")

t0 = rospy.Time.now()
while not rospy.is_shutdown() and state.armed:
    if(rospy.Time.now() - t0).to_sec() > 30:
        rospy.logwarn("降落超时，仍armed")
        break
    rate.sleep()
    
rospy.loginfo(f"结束 armed={state.armed}")



























