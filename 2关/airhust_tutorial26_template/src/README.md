# 一.
1. x,y,z和设定的目标位置setpoint_raw.position.x/y/z偏差小于0.2（err_max),yaw和setpoint_raw.yaw偏差小于0.1度。此时返回true
2. 在执行spinOnce()时，且消息中含有Odometry 此时会回调local_pos_cb，获取msg指向的内容，更新local_pos.pose.pose.position.x/y/z
3. 这一轮后面会先执行case2中的break结束switch(),然后执行
``` 
mavros_setpoint_pos_pub.publish(setpoint_raw); //发布话题
ros::spinOnce(); //处理消息
rate.sleep(); // 停止
```
下一轮接着执行case2，直到mission_pos_cruise()返回true.

# 二.
1. 计算思路：
- 计算速度：
首先调整template.cpp中spinOnce()的位置，把它移到了switch()的前面，保证了mission_pos_cruise读到的是最新的local_pos.pose.pose.position.x/y/z,及时调整速度。
然后更改了mission_pos_cruise()，用目标值减实际值来求误差
`float e_x = x + init_position_x_take_off- local_pos.pose.pose.position.x;`
然后用误差距离乘0.5来求速度
`setpoint_raw.velocity.x = 0.5*e_x;`
- 限制速度：
```
	float v_xy = std::sqrt(setpoint_raw.velocity.x  * setpoint_raw.velocity.x + setpoint_raw.velocity.y * setpoint_raw.velocity.y);
	if (v_xy > 1.0)
    	{
    	float rate = 1 / v_xy;
        setpoint_raw.velocity.x  *= rate;
        setpoint_raw.velocity.y  *= rate;
    	}
	//if(setpoint_raw.velocity.x > 1) setpoint_raw.velocity.x = 1;
	//if(setpoint_raw.velocity.y > 1) setpoint_raw.velocity.y = 1;
	
	if (setpoint_raw.velocity.z >  0.5) setpoint_raw.velocity.z =  0.5;
    	if (setpoint_raw.velocity.z < -0.5) setpoint_raw.velocity.z = -0.5;
```
限制x,y方向的速度采用的是如果超速就保持最大速度
限制水平方向合速度的时候采用了另外的方法，因为要保证减速之后x,y方向速度的比例不会变，并保证减速之后速度在限制速度以下，所以选择x.y同乘1/v_xy
> 但是我认为其实不用限制x,y方向的速度，只需要限制水平方向合速度就已经能满足要求了。
- 到达条件：
沿用原位置函数
- 怎样停止：
将x,y,z方向速度同时归零
```
	setpoint_raw.velocity.x = 0;         
        setpoint_raw.velocity.y = 0;
        setpoint_raw.velocity.z = 0;
        ROS_INFO("到达目标点，巡航点任务完成");
	mission_pos_cruise_flag = false;
	return true;
```

# 预测：若速度计算只进行一次，后面不再进行更新
那无人机的速度只受到了最高速度的限制，失去了和目标距离的关联，会导致冲过目标点
不过，，如果运气好，在冲过目标点的时候有一瞬间满足了到达条件，那可能会是这样的
       
       _____________ <-实际轨迹
       |
       | _____
       ||    |
       ||____| <-目标轨迹
       




