#include "ros/ros.h"
#include "sensor_msgs/Imu.h"

void imuDataCallback(const sensor_msgs::Imu::ConstPtr& msg){
  ROS_INFO("\nlinear acceleration\
      \nx: [%f]\ny:[%f]\nz:[%f]", msg->linear_acceleration.x,
      msg->linear_acceleration.y, msg->linear_acceleration.z);
}

int main(int argc, char **argv){
  ros::init(argc, argv, "drone_state");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe("/mavros/imu/data", 1000, imuDataCallback);
  ros::spin();
  return 0;
}
