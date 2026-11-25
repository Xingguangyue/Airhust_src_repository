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
TAG_X,TAG_Y = 1.0 , 0.0

state = State()
pos = None
tag_id = None
tag_dist = 1e9
tag_stamp = 0.0
last_frame = None
bridge = CvBridge()

# 选家族
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary,params)

def state_cb(msg):
    global state
    state = msg


def pose_cb(msg):
   global pos
   pos = msg.pose.position

def detect_tag(frame):
    """返回 (编号, 到画面中心的像素距离)；没检测到返回 (None, 1e9)。"""
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    corners,ids,_ = detector.detectMarkers(gray) #detectMarkers()返回值：corners(元素为二维数组的元组），ids(二维数组),最后一个是rejectedImgPoints（和corners形状一样）
    if ids is None or len(ids) == 0:
        return None, 1e9

    h,w = frame.shape[:2]
    cx0,cy0 = w / 2.0,h / 2.0
    best_id,best_d = None,1e9
     
    for i in range(len(ids)):
        pts = corners[i].reshape(4,2)
        cx,cy = pts.mean(axis=0)
        d = ((cx-cx0)**2 + (cy - cy0)**2)**0.5
        if d < best_d:
            best_id,best_d = int(ids.flatten()[i]),d
             
    return best_id,best_d
    
def image_cb(msg):
    global tag_id,tag_dist,tag_stamp,last_frame
    frame = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8") #转格式
    tid,d = detect_tag(frame) #获取图片id,距中心位置d
    if tid is not None:
        tag_id,tag_dist = tid,d
        tag_stamp = rospy.get_time() #时间戳
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
    
rospy.init_node("task3_mission")

rospy.Subscriber("/mavros/state", State, state_cb, queue_size=10)
rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb, queue_size=10)
rospy.Subscriber("/usb_cam/image_raw", Image, image_cb, queue_size=1)

pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

rospy.wait_for_service("/mavros/cmd/arming")
arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
rospy.wait_for_service("/mavros/set_mode")
mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
     
rate = rospy.Rate(20)
target = setpoint(0.0,0.0,TAKEOFF_Z)

conn_t0 = rospy.Time.now()
while not rospy.is_shutdown() and not state.connected:
    if (rospy.Time.now() - conn_t0).to_sec() > 20:
        rospy.logerr("20s 没连上飞控，检查 task5.launch / mavros 是否起来")
        raise SystemExit(1)
    rate.sleep()
rospy.loginfo("FCU connected") #等待飞控连接

# 喂饱飞控宝宝
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
        rospy.logerr("15s 仍未解锁，检查 COM_RC_IN_MODE（应为 1）")
        raise SystemExit(1)
    arm_srv(True)
    pub.publish(target)
    rate.sleep()
rospy.loginfo("armed, climbing")
     
go_to(0.0, 0.0, TAKEOFF_Z)
go_to(TAG_X, TAG_Y, TAKEOFF_Z) #让飞机过去

#让飞机停一会
for _ in range(20):
    pub.publish(setpoint(TAG_X, TAG_Y, TAKEOFF_Z))
    rate.sleep()

tag_age = rospy.get_time() - tag_stamp 
if tag_id is not None and tag_age < 1.0:
    rospy.loginfo(f"[TAG] id = {tag_id}  偏移 {tag_dist:.0f}px")
else:
    rospy.logwarn("[TAG] 未检测到标签或信息已过期")
if not go_to(TAG_X, TAG_Y, TAKEOFF_Z):
    rospy.logwarn("[WARN] 没到达标签正上方")
    
    
if last_frame is not None:
    cv2.imwrite("/tmp/shot_tag.jpg", last_frame)
    
rospy.loginfo("AUTO.LAND")
mode_srv(custom_mode="AUTO.LAND")

t0 = rospy.Time.now()
while not rospy.is_shutdown() and state.armed:
    if (rospy.Time.now() - t0).to_sec() > 30:
        rospy.logwarn("降落超时，仍 armed")
        break
    rate.sleep()
rospy.loginfo(f"结束 armed={state.armed}")
     
     
     
     
     
     
     
     
     
     
     
