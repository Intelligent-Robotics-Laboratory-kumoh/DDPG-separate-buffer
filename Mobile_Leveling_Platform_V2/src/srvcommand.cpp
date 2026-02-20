#include "srvcommand.h"
//#include "/home/baek/rl_ws/src/Mobile_Leveling_Platform_V2/include/Mobile_controller.h"
#include "Mobile_controller.h"
#include <cstdlib>

#define PI 3.14159265359

srvcommand::srvcommand()
{
}


int main(int argc, char **argv)
{
    ros::init(argc,argv,"input_vel");
    ros::NodeHandle nh_;
    ros::Publisher MLP_linear_cont_pub=nh_.advertise<geometry_msgs::Twist>("linearVel",100);
    ros::Publisher MLP_angular_cont_pub=nh_.advertise<Mobile_Leveling_Platform_V2::joint_vel>("AnglePos",100);
    ros::Rate loop_rate(10);


    geometry_msgs::Twist twist_msg;
    Mobile_Leveling_Platform_V2::joint_vel joint_msg;
    twist_msg.linear.x = atof(argv[1]);
    twist_msg.linear.y = 0;
    twist_msg.linear.z = 0;
    twist_msg.angular.x = 0;
    twist_msg.angular.y = 0;
    twist_msg.angular.z = 0;

    joint_msg.data = atof(argv[1]) * PI/180;//degree to radian

    ROS_INFO("recieve argv.1 = %f\n", twist_msg.linear.x);
    ROS_INFO("recieve argv.2 = %f\n", joint_msg.data);
    while(ros::ok())
    {   
        MLP_linear_cont_pub.publish(twist_msg);
        MLP_angular_cont_pub.publish(joint_msg);

        loop_rate.sleep();
    }
    return 0;
}
