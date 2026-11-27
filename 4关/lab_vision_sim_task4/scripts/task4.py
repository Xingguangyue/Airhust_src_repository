#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import pytesseract
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

TAKEOFF_Z = 1.5
TEXT_X, TEXT_Y = 1.0, 0.0
OCR_FRESH = 1.0

state = State()
pos = None
ocr_text = ""
ocr_stamp = 0.0
last_frame = None
bridge = CvBridge()

def state_cb(msg):
    global state
    state = msg

def pose_cb(msg):
    global pos
    pos = msg.pose.position

def read_text(frame):
    # 1. 找白色纸面
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ""
    c = max(cnts,key=cv2.contourArea)
    if cv2.contourArea(c) < 2000:
        return ""
    x,y,w,h = cv2.boundingRect(c)#求正外接矩形：boundingRect 会画一个水平、竖直的框把轮廓 c 完全包住，并返回这个框的左上角坐标 (x, y)，以及宽度 w 和高度 h
    pad = frame[y:y+h,x:x+w] 
    
    #2. 转文字方向(逆时针）
    #rot = cv2.totate(pad,cv2.ROTATE_90_COUNTERCLOCKWISE))
    rot = pad
    
    #3.放大
    big = cv2.resize(rot,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    g2 = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        txt = pytesseract.image_to_string(bw, config="--psm 7").strip()
    except Exception:
        return ""
    return "".join(ch for ch in txt if ch.isalpha()).upper()
    
def image_cb(msg):
    global ocr_text, ocr_stamp, last_frame
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    txt = read_text(frame)
    if txt:
        ocr_text = txt
        ocr_stamp = rospy.get_time()
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
    
rospy.init_node("task4_mission")

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

arm_t0 = rospy.Time.now()
while not rospy.is_shutdown() and not state.armed:
    if (rospy.Time.now() - arm_t0).to_sec() > 15:
        rospy.logerr("15s 仍未解锁，检查 COM_RC_IN_MODE / EKF")
        raise SystemExit(1)
    arm_srv(True)
    pub.publish(target)
    rate.sleep()
rospy.loginfo("armed, climbing")

go_to(0.0, 0.0, TAKEOFF_Z)
if not go_to(TEXT_X, TEXT_Y, TAKEOFF_Z):
    rospy.logwarn("[WARN] 没到达靶标正上方")

for _ in range(20):
    pub.publish(setpoint(TEXT_X, TEXT_Y, TAKEOFF_Z))
    rate.sleep()

age = rospy.get_time() - ocr_stamp
if ocr_text and age < OCR_FRESH:
    rospy.loginfo(f"[OCR] {ocr_text}")
else:
    rospy.logwarn("[OCR] 未读到文字或已过期")
    if last_frame is not None:
        cv2.imwrite("/tmp/shot_text.jpg", last_frame)

rospy.loginfo("AUTO.LAND")
mode_srv(custom_mode="AUTO.LAND")

t0 = rospy.Time.now()
while not rospy.is_shutdown() and state.armed:
    if (rospy.Time.now() - t0).to_sec() > 30:
        rospy.logwarn("降落超时，仍 armed")
        break
    rate.sleep()
rospy.loginfo(f"结束 armed={state.armed}")
