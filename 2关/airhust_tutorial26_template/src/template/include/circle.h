#pragma once

#include <cmath>
#include <ros/ros.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

#define ALTITUDE 0.8f

#define KP      0.5f    
#define V_MAX   0.5f    
#define VZ_MAX  0.3f   
#define YAW_TOL 0.15f 


mavros_msgs::PositionTarget setpoint_raw;
mavros_msgs::State current_state;


void state_cb(const mavros_msgs::State::ConstPtr &msg)
{
    current_state = *msg;
}

tf::Quaternion quat;
nav_msgs::Odometry local_pos;

double roll, pitch, yaw;

float init_position_x_take_off = 0;
float init_position_y_take_off = 0;
float init_position_z_take_off = 0;
float init_yaw_take_off = 0;
bool flag_init_position = false;

//确定并记录起飞位置
void local_pos_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    local_pos = *msg;
    tf::quaternionMsgToTF(local_pos.pose.pose.orientation, quat);
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    
    if (flag_init_position == false)
    {
        init_position_x_take_off = local_pos.pose.pose.position.x;
        init_position_y_take_off = local_pos.pose.pose.position.y;
        init_position_z_take_off = local_pos.pose.pose.position.z;
        init_yaw_take_off = yaw;
        flag_init_position = true;
        ROS_INFO("init position: (%.2f, %.2f, %.2f), yaw: %.1f deg",
                 init_position_x_take_off, init_position_y_take_off,
                 init_position_z_take_off, init_yaw_take_off * 180.0 / M_PI);
    }
}

// ============================================================
// 底层原语：给出绝对目标，算速度指令并判到达。
// 所有巡航任务（含后面的绕圈）都调它，别各自重写一套。
// 参数全是绝对坐标（米）与绝对航向（弧度）。
//
// 相对原文件唯一新增：第 6 个参数 check_yaw。
// 绕圈时机头固定，target_yaw 与 yaw 的差可能绕到 ±π 另一侧，
// fabs 会算出接近 6.28 的假误差导致永远不到达，所以那时传 false。
// ============================================================
bool cruise_to_abs(float tx, float ty, float tz, float target_yaw, float error_max,
                   bool check_yaw = true)
{
     float x_now = local_pos.pose.pose.position.x;
     float y_now = local_pos.pose.pose.position.y;
     float z_now = local_pos.pose.pose.position.z;

     float e_x = tx - x_now;
     float e_y = ty - y_now;
     float e_z = tz - z_now;
    // target_yaw 只用 0 / +-1.57 这几个值，都不跨 +-180 度，所以不做角度归一化。
    // 绕圈时机头固定在 0 附近，也不跨；真正跨 ±π 的是"机头跟随切线"那种拓展方案，
    // 那时这里必须改成归一化比较，或者（本文件的处理）直接传 check_yaw = false。
     float e_yaw = target_yaw - yaw;

    // P 控制：速度 = 增益 * 误差。误差随距离缩小，速度自动降下来
    float vx = KP * e_x;
    float vy = KP * e_y;
    float vz = KP * e_z;

    // 水平限速：按合速度等比缩放，方向不变（不是各轴独立截断）
     float v_xy = std::sqrt(vx * vx + vy * vy);
    if (v_xy > V_MAX)
    {
        vx *= V_MAX / v_xy;
        vy *= V_MAX / v_xy;
    }
    if (vz > VZ_MAX) vz = VZ_MAX;
    if (vz < -VZ_MAX) vz = -VZ_MAX;

    // 位 0/1/2 = 1 -> 忽略位置；位 3/4/5 = 0 -> 采用速度；位 10 = 0 -> 采用 yaw
    setpoint_raw.type_mask = 1 + 2 + 4 + 64 + 128 + 256 + 2048;
    setpoint_raw.coordinate_frame = 1;   // FRAME_LOCAL_NED，世界系绝对坐标

    setpoint_raw.velocity.x = vx;
    setpoint_raw.velocity.y = vy;
    setpoint_raw.velocity.z = vz;
    setpoint_raw.yaw = target_yaw;

    ROS_INFO_THROTTLE(1.0,
        "to (%.2f,%.2f,%.2f) now (%.2f,%.2f,%.2f) err(%.2f,%.2f,%.2f) v(%.2f,%.2f,%.2f)",
        tx, ty, tz, x_now, y_now, z_now, e_x, e_y, e_z, vx, vy, vz);

    bool arrived = (std::fabs(e_x) < error_max) &&
                   (std::fabs(e_y) < error_max) &&
                   (std::fabs(e_z) < error_max);
    if (check_yaw)
        arrived = arrived && (std::fabs(e_yaw) < YAW_TOL);

    if (arrived)
    {
        // 到位后把速度显式清零，别让最后一帧的残值继续被飞控采用
        setpoint_raw.velocity.x = 0;
        setpoint_raw.velocity.y = 0;
        setpoint_raw.velocity.z = 0;
    }
    return arrived;
}

// ===== 以【起飞点】为原点的绝对巡航 =====
// x/y/z 都是"距起飞点多少米"，函数内部翻译成绝对坐标
bool mission_pos_cruise(float x, float y, float z, float target_yaw, float error_max,
                        bool check_yaw = true)
{
    return cruise_to_abs(x + init_position_x_take_off,
                         y + init_position_y_take_off,
                         z + init_position_z_take_off,
                         target_yaw, error_max, check_yaw);
}

//绕圈参数
struct CircleConfig
{
	float radius = 1.0f; //半径
	float period = 30.0f; //绕完一圈的目标时长
	float theta0 = 0.0f; //起点角度
	float direction = 1; //+1逆时针，-1顺时针
	float hold_yaw = 0.0f; //绕圈的时候保持的航向
};

struct CircleProgress
{
	bool started = false; //有没有进过绕圈
	ros::Time t0; //开始绕圈时间
	float theta_end = 0.0f; //终点角度
};
bool circle_step(const CircleConfig &cfg, CircleProgress &prog, float err_max)
{
    // ---- ① 第一次进来：记起始时刻，算好终点角 ----
    if (!prog.started)
    {
        prog.started   = true;
        prog.t0        = ros::Time::now();
        prog.theta_end = (float)(cfg.theta0 + cfg.direction * 2.0 * M_PI);
    }

    const double elapsed = (ros::Time::now() - prog.t0).toSec();

    // ---- ② 目标角度：时间到之前匀速跑，到点后冻结 ----
    double theta;
    if (elapsed >= cfg.period)
        theta = prog.theta_end;
    else
        theta = cfg.theta0 + cfg.direction * 2.0 * M_PI * elapsed / cfg.period;

    // ---- ③ 极坐标转直角（相对起飞点）----
    const double x_k = cfg.radius * std::cos(theta);
    const double y_k = cfg.radius * std::sin(theta);

    // ---- ④ 交给控制层算速度、判到位 ----
    const bool arrived = mission_pos_cruise((float)x_k, (float)y_k, ALTITUDE,
                                           cfg.hold_yaw, err_max, false);

    // ---- ⑤ 时间到 且 真到了终点，才算绕完 ----
    return (elapsed >= cfg.period) && arrived;
}




































// ===== 原地悬停（位置保持，不是速度控制）=====
bool hover_started = false;
ros::Time hover_last_time;
double hover_x = 0, hover_y = 0, hover_z = 0, hover_yaw = 0;

bool hover(float time_duration)
{
    if (!hover_started)
    {
        hover_last_time = ros::Time::now();
        hover_started = true;
        hover_x = local_pos.pose.pose.position.x;
        hover_y = local_pos.pose.pose.position.y;
        hover_z = local_pos.pose.pose.position.z;
        hover_yaw = yaw;
    }

    // 位置接管；显式屏蔽速度位，避免上一段速度控制的残值干扰
    setpoint_raw.type_mask = 8 + 16 + 32 + 64 + 128 + 256 + 2048;
    setpoint_raw.coordinate_frame = 1;
    setpoint_raw.position.x = hover_x;
    setpoint_raw.position.y = hover_y;
    setpoint_raw.position.z = hover_z;
    setpoint_raw.yaw = hover_yaw;
    setpoint_raw.velocity.x = 0.0;
    setpoint_raw.velocity.y = 0.0;
    setpoint_raw.velocity.z = 0.0;

    if (ros::Time::now() - hover_last_time > ros::Duration(time_duration))
    {
        hover_started = false;
        ROS_INFO("hover done");
        return true;
    }
    return false;
}

// ===== 降落 =====
// 目标压到地面以下，让位置环持续下压；判据改成"真到高度"+"超时兜底"
float land_x = 0;
float land_y = 0;
bool land_started = false;
ros::Time land_start_time;

bool precision_land()
{
    if (!land_started)
    {
        land_x = local_pos.pose.pose.position.x;
        land_y = local_pos.pose.pose.position.y;
        land_start_time = ros::Time::now();
        land_started = true;
        ROS_INFO("landing started at (%.2f, %.2f)", land_x, land_y);
    }

    // 位置接管，速度位全部忽略（原代码漏了这一步，会把巡航残值带进来）
    setpoint_raw.type_mask = 8 + 16 + 32 + 64 + 128 + 256 + 1024 + 2048;
    setpoint_raw.coordinate_frame = 1;
    setpoint_raw.position.x = land_x;
    setpoint_raw.position.y = land_y;
    setpoint_raw.position.z = init_position_z_take_off - 0.15;   // 落到地面以下 15 cm
    setpoint_raw.velocity.x = 0.0;
    setpoint_raw.velocity.y = 0.0;
    setpoint_raw.velocity.z = 0.0;

    const double elapsed = (ros::Time::now() - land_start_time).toSec();
    const double z_rel = local_pos.pose.pose.position.z - init_position_z_take_off;

    const bool touched = (z_rel < 0.08);      // 真实高度判据
    const bool timeout = (elapsed > 8.0);     // 兜底，防止永远降不下去

    if (touched || timeout)
    {
        ROS_INFO("landing done: z_rel = %.3f, elapsed = %.1f s", z_rel, elapsed);
        land_started = false;
        return true;
    }
    return false;
}
