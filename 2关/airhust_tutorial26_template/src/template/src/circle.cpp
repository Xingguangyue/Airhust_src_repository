#include <circle.h>

#include <clocale>
#include <cstdlib>
#include <iostream>

using std::cout;
using std::endl;

//全局变量
float if_debug = 0;
float err_max = 0.2;
//打印参数
void print_param()
{
   	std::cout << "=== 控制参数 ===" << std::endl;
    	std::cout << "err_max : " << err_max << std::endl;
    	std::cout << "ALTITUDE: " << ALTITUDE << std::endl;
    	std::cout << "if_debug: " << if_debug << std::endl;
    	std::cout << "KP      : " << KP << "  (1/s)" << std::endl;
    	std::cout << "V_MAX   : " << V_MAX << "  (m/s)" << std::endl;
    	std::cout << "VZ_MAX  : " << VZ_MAX << "  (m/s)" << std::endl;
    	if (if_debug == 1) cout << "自动offboard" << std::endl;
    	else cout << "遥控器offboard" << std::endl;
}
int main(int argc,char **argv)
{
	setlocale(LC_ALL,"");
	ros::init(argc, argv, "circle");
    	ros::NodeHandle nh;
    	
    	//创建subscriber
    	ros::Subscriber state_sub =
        nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    	ros::Subscriber local_pos_sub =
        nh.subscribe<nav_msgs::Odometry>("/mavros/local_position/odom", 10, local_pos_cb);

	//创建publisher
    	ros::Publisher mavros_setpoint_pos_pub =
        nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 100);

	//创建服务器
    	ros::ServiceClient arming_client =
        nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
    	ros::ServiceClient set_mode_client =
	nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
	
	//心跳包
	ros::Rate rate(20);

	nh.param<float>("err_max", err_max, 0.2);
    	nh.param<float>("if_debug", if_debug, 0);
    	print_param();
    	
    	//读绕圈的参数
    	double p_radius = 2.0;
    	double p_period = 30.0;
    	double p_theta0 = 0.0;
	double p_yaw = 0.0;
	int p_dir = 1;
	
	nh.param<double>("circle_radius",    p_radius,  2.0);
    	nh.param<double>("circle_period",    p_period,  30.0);
    	nh.param<double>("circle_theta0",    p_theta0,  0.0);
    	nh.param<double>("hold_yaw",         p_yaw,     0.0);
    	nh.param<int>   ("circle_direction", p_dir,     1);
    	
    	CircleConfig cfg;
	cfg.radius    = (float)p_radius;
    	cfg.period    = (float)p_period;
    	cfg.theta0    = (float)p_theta0;
    	cfg.direction = (p_dir >= 0) ? 1 : -1;
    	cfg.hold_yaw  = (float)p_yaw;

	//======================起飞！===============================
	//等飞控连接
	while (ros::ok() && !current_state.connected)
    	{
        	ros::spinOnce();
        	rate.sleep();
    	}

	// 等第一帧里程计，把起飞锚点锁下来（不然前几帧锚点是 0）
	while (ros::ok() && !flag_init_position)
    	{	
        	ros::spinOnce();
        	rate.sleep();
    	}	

	mavros_msgs::SetMode offb_set_mode;
    	offb_set_mode.request.custom_mode = "OFFBOARD";

    	mavros_msgs::CommandBool arm_cmd;
    	arm_cmd.request.value = true;

    	mavros_msgs::CommandBool disarm_cmd;
    	disarm_cmd.request.value = false;   // 收尾用，先备着

	ros::Time last_request = ros::Time::now();

    	const double takeoff_z = double(init_position_z_take_off) + ALTITUDE;
    	bool z_in_range = false;
    	ros::Time z_in_range_time;

    	const ros::Time t_takeoff = ros::Time::now();

	    while (ros::ok())
    {
        // ① 还没进 OFFBOARD，每 3 秒试一次
        if (current_state.mode != "OFFBOARD" &&
            ros::Time::now() - last_request > ros::Duration(3.0))
        {
            if (if_debug == 1)
            {
                if (set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
                    ROS_INFO("Offboard enabled");
            }
            else
            {
                ROS_INFO("Waiting for OFFBOARD mode");
            }
            last_request = ros::Time::now();
        }
        else if (!current_state.armed &&
                 ros::Time::now() - last_request > ros::Duration(3.0))
        {
            // ② 已经 OFFBOARD，试解锁
            if (arming_client.call(arm_cmd) && arm_cmd.response.success)
            {
                ROS_INFO("Vehicle armed");
            }
            last_request = ros::Time::now();
        }

        // ③ 高度到位并稳住 1 秒，才算真起来
        const double z_now = local_pos.pose.pose.position.z;
        if (std::fabs(z_now - takeoff_z) < 0.2)
        {
            if (!z_in_range)
            {
                z_in_range = true;
                z_in_range_time = ros::Time::now();
            }
            else if (ros::Time::now() - z_in_range_time > ros::Duration(1.0))
            {
                break;
            }
        }
        else
        {
            z_in_range = false;
        }

        // ④ 原地拔高
        mission_pos_cruise(0, 0, ALTITUDE, init_yaw_take_off, err_max);
        mavros_setpoint_pos_pub.publish(setpoint_raw);
        ros::spinOnce();
        rate.sleep();

        // ⑤ 兜底
        if (ros::Time::now() - t_takeoff > ros::Duration(40.0))
        {
            ROS_ERROR("takeoff timeout, z=%.2f", local_pos.pose.pose.position.z);
            return 1;
        }
    }
    ROS_WARN("takeoff done, circle start");

	    
	
	//开始绕圈
	CircleProgress prog;
	const ros::Time t_circle = ros::Time::now();
    	bool circle_done = false;
    	
    	while(ros::ok() && !circle_done)
    	{
    		ros::spinOnce();
    		circle_done = circle_step(cfg,prog,err_max);
    		mavros_setpoint_pos_pub.publish(setpoint_raw); 
		rate.sleep();
		
		// 兜底
		if(ros::Time::now() - t_circle > ros::Duration(cfg.period + 20.0))
		{
			ROS_ERROR("circle timeout after %.1f s", (ros::Time::now() - t_circle).toSec());
            		break;
            	}
         }
         ROS_WARN("circle finished in %.1f s", (ros::Time::now() - t_circle).toSec());
         //绕圈结束
         
         //悬停三秒
         while(ros::ok() && !hover(3.0f))
         {
         mavros_setpoint_pos_pub.publish(setpoint_raw);
         ros::spinOnce();
         rate.sleep();
    	 }
    	 
    	 //降落
    	 while (ros::ok() && !precision_land())
    	{
        	mavros_setpoint_pos_pub.publish(setpoint_raw);
        	ros::spinOnce();
        	rate.sleep();
    	}
    	
    	// ===== 收尾：落地后交给 PX4 的 AUTO.LAND 完成上锁 =====
    // 直接 disarm 会被拒绝：我们一直在发"落到地下 15cm"的位置 setpoint，
    // 电机持续出力，PX4 的落地检测不成立 → 它认为还在空中 → DENIED。
    // 切 AUTO.LAND 让飞控自己判定落地并自动上锁。
    mavros_msgs::SetMode land_mode;
    land_mode.request.custom_mode = "AUTO.LAND";

    bool disarmed = false;
    if (set_mode_client.call(land_mode) && land_mode.response.mode_sent)
    {
        for (int i = 0; i < 300 && ros::ok(); ++i)      // 20Hz × 300 ≈ 15 秒
        {
            ros::spinOnce();
            rate.sleep();

            if (!current_state.armed)
            {
                disarmed = true;
                break;
            }
        }
    }

    if (disarmed)
    {
        ROS_INFO("disarmed, mission complete");
    }
    else
    {
        ROS_WARN("disarm failed, 请用遥控器手动上锁");
    }

    ros::shutdown();
    	return 0;
}

