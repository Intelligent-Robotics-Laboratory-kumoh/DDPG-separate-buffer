#ifndef MOBILE_CONTROLLER_H_
#define MOBILE_CONTROLLER_H_

#include <iostream>
#include <math.h>
#include <time.h>
#include <string.h>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <gazebo_msgs/LinkState.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <Mobile_Leveling_Platform_V2/SensorState.h>
#include <Mobile_Leveling_Platform_V2/joint_vel.h>
#include <Mobile_Leveling_Platform_V2/kalman.h>
#include "Mobile_controller.h"

#define WHEEL_RADIUS                    0.033     // meter

#define LEFT                            0
#define RIGHT                           1

#define MAX_LINEAR_VELOCITY             2   // m/s 0.112
#define MAX_ANGULAR_VELOCITY            2.84   // rad/s
#define VELOCITY_STEP                   0.01   // m/s
#define VELOCITY_LINEAR_X               0.01   // m/s
#define VELOCITY_ANGULAR_Z              0.1    // rad/s
#define SCALE_VELOCITY_LINEAR_X         1
#define SCALE_VELOCITY_ANGULAR_Z        1

#define DEG2RAD(x)                      (x * 0.01745329252)  // *PI/180
#define RAD2DEG(x)                      (x * 57.2957795131)  // *180/PI

#define TORQUE_ENABLE                   1       // Value for enabling the torque of motor
#define TORQUE_DISABLE                  0       // Value for disabling the torque of motor
#define g 9.8066

#define LinearStepSize 0.01
#define AngularStepSize 0.1

#define kp 0.55 // 0.5
#define ki 0.05
#define kd 0.04

using Eigen::MatrixXd;

class Mobile_Leveling_Platform
{
 public:
  Mobile_Leveling_Platform();
  ~Mobile_Leveling_Platform();
  bool init();
  bool update();

 private:
  // ROS NodeHandle
  ros::NodeHandle nh_;
  ros::NodeHandle nh_priv_;

  // ROS Parameters
  // (TODO)

  // ROS Time
  ros::Time last_cmd_vel_time_;
  ros::Time prev_update_time_;

  // ROS Topic Publishers
  ros::Publisher joint_states_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher geometry_pub_;
  ros::Publisher upperangle_pub_;
  ros::Publisher gimbal_control_pub_;
  ros::Publisher kalman_pub_;

  // ROS Topic Subscribers
  ros::Subscriber cmd_vel_sub_;
  ros::Subscriber cmd_joint_vel_;
  ros::Subscriber imu_base_vel_;
  ros::Subscriber imu_upper_vel_;
  ros::Subscriber camera_imu_vel_;
  
  // for RL
  ros::Subscriber cmd_vel_sub_RL;
  ros::Subscriber cmd_joint_vel_RL;
  ros::Subscriber imu_base_vel_RL;
  ros::Subscriber imu_upper_vel_RL;
  ros::Publisher  geometry_pub_RL;
  ros::Publisher  upperangle_pub_RL;
  //
  ros::Subscriber model_pose_;

  ros::ServiceClient req_set_pose_;
  ros::ServiceClient req_get_pose_;

  sensor_msgs::JointState joint_states_;
  nav_msgs::Odometry odom_;
  geometry_msgs::Twist geometry_;
  tf::TransformBroadcaster tf_broadcaster_;
  std_msgs::Float64 angle_vel_;

  Mobile_Leveling_Platform_V2::joint_vel joint_vel;
  Mobile_Leveling_Platform_V2::kalman kalman_vel_;

  gazebo_msgs::LinkState link_state_;

  double wheel_speed_cmd_[2];
  double goal_linear_velocity_;
  double goal_angular_velocity_;
  double cmd_vel_timeout_;

  float  wheel_acc;
  float  target_linear_vel;
  float  target_angular_vel;
  float  control_linear_vel;
  float  control_angular_vel;

  float pre_error;
  float past_error;
  float integral_error;
  float d_error;
  float past_time;
  float pre_time;
  float dt;
  float antiwindup_trig;
  float yaw;
  float angoffset;



  struct Quaternion
  {
      float x,y,z,w;
  };

  struct euler
  {
      float r,p,y;
  };

  struct camera_euler
  {
      float roll,pitch,yaw;
  };

  struct imu_accg
  {
      float x,y,z;
  };

  struct linear_acc
  {
      float x,y,z;
  };
  struct last
  {
      float x,y,z;
  };

  struct curr
  {
      float x,y,z;
  };

 struct ekf
  {   
      struct imu_accg accg;
      struct linear_acc accl;
      struct last pre;
      struct curr curr;
      struct euler angle;
      float acc;

      MatrixXd f = MatrixXd(4,4);
      MatrixXd h = MatrixXd(3,4);
      MatrixXd p = MatrixXd(4,4);
      MatrixXd pred_p = MatrixXd(4,4);
      MatrixXd z = MatrixXd(3,1);
      MatrixXd pred_x = MatrixXd(4,1);
      MatrixXd x = MatrixXd(4,1);
      MatrixXd I = MatrixXd(4,4);
      MatrixXd est_x = MatrixXd(4,1);
      MatrixXd est_p = MatrixXd(4,4); 
      MatrixXd y = MatrixXd(3,1);
      MatrixXd Q = MatrixXd(4,4);
      MatrixXd K = MatrixXd(4,3);
      MatrixXd R = MatrixXd(3,3);
      MatrixXd S = MatrixXd(3,3);
      MatrixXd zero_mat = MatrixXd(4,4);

      float dt;
  };


  float  odom_pose_[3];
  float  odom_vel_[3];
  double pose_cov_[36];

  std::string joint_states_name_[2];

  double last_position_[2];
  double last_velocity_[2];

  double wheel_seperation_;
  double turning_radius_;
  double robot_radius_;

  // Function prototypes
  float constrain(float input,float low,float high);
  float checkLinearLimitVelocity(float vel);
  float checkAngularLimitVelocity(float vel);
  float makeSimpleProfile(float output, float input, float slop);
  void imuCalculation_base(const sensor_msgs::ImuConstPtr &imu_base);
  void imuCalculation_upper(const sensor_msgs::ImuConstPtr &imu_upper);
  void gimbal_control(const sensor_msgs::ImuConstPtr &camera_imu);
  float QuaterToEuler(float x, float y, float z, float w);
  float EulerToQuater(float roll, float pitch, float yaw);
  void EKF_acc(void);
  void Mat_init(void);
  //void updatepose(const gazebo_msgs::ModelStatesConstPtr &mp);
  void updategeometry(void);
  void commandVelocityCallback(const geometry_msgs::TwistConstPtr &cmd_vel_msg);
  bool updateOdometry(ros::Duration diff_time);
  void updateJoint(void);
  void updateTF(geometry_msgs::TransformStamped& odom_tf);
  void angleOffset(const Mobile_Leveling_Platform_V2::joint_velConstPtr &jointoffset);
};

#endif // TURTLEBOT3_FAKE_H_
