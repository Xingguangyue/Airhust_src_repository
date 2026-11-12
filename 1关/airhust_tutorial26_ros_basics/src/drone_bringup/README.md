# 阶段一：修改了CMakeLists.txt 和 package.xml
## 1. 补充和修改
### 1.1 
在CMakeLists里面增加了包drone_package 
并且 
补充了add_executable(flight_node src/flight_node.cpp) 
target_link_libraries(flight_node ${catkin_LIBRARIES})
### 1.2
在package.xml中 补全了依赖，统一采用<depend></depend>的格式（因为我一开始不清楚哪些包是在编译/运行时需要依赖的 手动狗头）

---

## 2. 思考过程
### 2.1 为什么想到给CMakeLists.txt 里面的find_package 和 catkin_package 增加包drone_backend
- 第一次编译的时候报错中出现了drone_backend这个包（我忘了保存报错信息了，大概出现了吧），然后在package.xml中发现了和roscpp在 一起的drone_backend，我盲猜他俩应该出现在相同的位置，于是加进去了。

- 于是按照相似的类比思想，我把geometry_msgs std_msgs两个包补充在了package.xml中。
### 2.2 然后报错ERROR: cannot launch node of type [drone_bringup/flight_node]: Cannot locate node of type [flight_node] in package [drone_bringup]. Make sure file exists in package path and permission is set to executable (chmod +x)
 
 - 于是后面仿照example_listen的写法我又补充了
add_executable(flight_node src/flight_node.cpp)
target_link_libraries(flight_node ${catkin_LIBRARIES})

---

## 3.学到了什么
1. CMakeLists.md是掌管编译的神。里面写的是编译需要找的包和要编译的文件，并负责生成catkin_LIBRARIES库
2. package.xml则负责提供参数。<depend> <build_depend> <exec_depend>写的是三种依赖
3. ~/.zshrc中source /opt/ros/noetic/setup.zsh，source ~/airhust_ws/devel/setup.zsh都有什么用？
第一个提供的是ros需要的大的环境，而第二个则负责airhust_ws这个文件里面具体的小环境的配置
4. ros的层级概念：文件系统->计算图级（节点，话题，订阅）->社区级


# 阶段二：完善flight_node.cpp
## 1. 补充和修改 （见代码）

## 2. 学到了什么
1. 人 传感器 后端 三者之间的互动关系 ： 传感器给人距离参数 人控制后端给飞机设定坐标 后端返回给人飞机的位置坐标
