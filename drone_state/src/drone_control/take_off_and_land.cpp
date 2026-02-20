#include <cstdlib>

#include <ros/ros.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandTOL.h>
#include <mavros_msgs/SetMode.h>

#include <geometry_msgs/PoseStamped.h>

int main(int argc, char **argv)
{

    int rate = 20;

    ros::init(argc, argv, "take_off_and_land");
    ros::NodeHandle n;

    ros::Rate r(rate);

    ///////////////////ARM//////////////////////
    ros::ServiceClient arming_client = n.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming"); // /mavros/cmd/arming==>topic 
    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    if (arming_client.call(arm_cmd) && arm_cmd.response.success)
    {
        ROS_INFO("Vehicle armed");
    } else {
        ROS_ERROR("Failed arming or disarming");
    }

    /////////////////TAKEOFF////////////////////
    ros::ServiceClient takeoff_client = n.serviceClient<mavros_msgs::CommandTOL>("/mavros/cmd/takeoff");
    mavros_msgs::CommandTOL takeoff_cmd;
    takeoff_cmd.request.altitude = 10;
    takeoff_cmd.request.latitude = 36.147661;
    takeoff_cmd.request.longitude = 128.394780;
    takeoff_cmd.request.min_pitch = 0;
    takeoff_cmd.request.yaw = 0;
    if(takeoff_client.call(takeoff_cmd) && takeoff_cmd.response.success){
        ROS_INFO("Okay Takeoff");
    }else{
        ROS_ERROR("Failed Takeoff");
    }

    /////////////////DO STUFF///////////////////
    sleep(10);


    ///////////////////LAND/////////////////////
    ros::ServiceClient land_client = n.serviceClient<mavros_msgs::CommandTOL>("/mavros/cmd/land");
    mavros_msgs::CommandTOL land_cmd;
    land_cmd.request.altitude = 0;
    land_cmd.request.latitude = 36.147665;
    land_cmd.request.longitude = 128.394780;
    land_cmd.request.min_pitch = 0.1;
    land_cmd.request.yaw = 0;
    if(land_client.call(land_cmd) && land_cmd.response.success){
        ROS_INFO("Okay Land");
    }else{
        ROS_ERROR("Failed Land");
    }

    while (n.ok())
    {
      ros::spinOnce();
      r.sleep();
    }

    return 0;
}
