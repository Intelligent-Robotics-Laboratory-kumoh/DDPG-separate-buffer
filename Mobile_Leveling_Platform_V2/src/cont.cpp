#include "Mobile_controller.h"

/////////////////////얘뭐지?////////////////////////////
Mobile_Leveling_Platform::Mobile_Leveling_Platform()
: nh_priv_("~")
{
  //Init fake turtlebot node
  bool init_result = init();
  ROS_ASSERT(init_result);
}

Mobile_Leveling_Platform::~Mobile_Leveling_Platform()
{
}
///////////////////////////////////////////////////////

bool Mobile_Leveling_Platform::init()
{
  // initialize ROS parameter

  wheel_seperation_ = 0.67;
  turning_radius_   = 0.1435;
  robot_radius_     = 0.220;

  nh_.param("wheel_left_joint_name", joint_states_name_[LEFT],  std::string("front_L_motor_rev_joint"));
  nh_.param("wheel_right_joint_name", joint_states_name_[RIGHT],  std::string("fornt_R_motor_rev_joint"));
  nh_.param("joint_states_frame", joint_states_.header.frame_id, std::string("base_link"));
  nh_.param("odom_frame", odom_.header.frame_id, std::string("odom"));
  nh_.param("base_frame", odom_.child_frame_id, std::string("base_link"));

  // initialize variables
  wheel_speed_cmd_[LEFT]  = 0.0;
  wheel_speed_cmd_[RIGHT] = 0.0;
  goal_linear_velocity_   = 0.0;
  goal_angular_velocity_  = 0.0;
  cmd_vel_timeout_        = 1.0;
  last_position_[LEFT]    = 0.0;
  last_position_[RIGHT]   = 0.0;
  last_velocity_[LEFT]    = 0.0;
  last_velocity_[RIGHT]   = 0.0;

  past_error    = 0;
  pre_error     = 0;
  integral_error= 0;
  d_error       = 0;
  pre_time      = 0;
  past_time     = 0;
  dt            = 0;
  antiwindup_trig = 0;
  wheel_acc    = 0;

  double pcov[36] = { 0.1,   0,   0,   0,   0, 0,
                        0, 0.1,   0,   0,   0, 0,
                        0,   0, 1e6,   0,   0, 0,
                        0,   0,   0, 1e6,   0, 0,
                        0,   0,   0,   0, 1e6, 0,
                        0,   0,   0,   0,   0, 0.2};
  memcpy(&(odom_.pose.covariance),pcov,sizeof(double)*36);
  memcpy(&(odom_.twist.covariance),pcov,sizeof(double)*36);

  odom_pose_[0] = 0.0;
  odom_pose_[1] = 0.0;
  odom_pose_[2] = 0.0;

  odom_vel_[0] = 0.0;
  odom_vel_[1] = 0.0;
  odom_vel_[2] = 0.0;

  joint_states_.name.push_back(joint_states_name_[LEFT]);
  joint_states_.name.push_back(joint_states_name_[RIGHT]);
  joint_states_.position.resize(2,0.0);
  joint_states_.velocity.resize(2,0.0);
  joint_states_.effort.resize(2,0.0);

  Mat_init();

  // initialize publishers
  joint_states_pub_ = nh_.advertise<sensor_msgs::JointState>("joint_states_1", 100);
  odom_pub_         = nh_.advertise<nav_msgs::Odometry>("odom_1", 100);
  geometry_pub_     = nh_.advertise<geometry_msgs::Twist>("twist2gazebo", 100);
  upperangle_pub_   = nh_.advertise<std_msgs::Float64>("MLP/joint2_position_controller/command",100);
  //upperangle_pub_RL   = nh_.advertise<std_msgs::Float64>("MLP/joint2_position_controller/command",100);
  //gimbal_control_pub_ = nh_.advertise<gazebo_msgs::LinkState>("/gazebo/set_link_state",100);
  kalman_pub_ = nh_.advertise<Mobile_Leveling_Platform_V2::kalman>("kalman",100);

  // initialize subscribers
  cmd_vel_sub_  = nh_.subscribe("linearVel", 100,  &Mobile_Leveling_Platform::commandVelocityCallback, this);
  cmd_joint_vel_= nh_.subscribe("AnglePos", 100 , &Mobile_Leveling_Platform::angleOffset,this);
  
  // for RL
  imu_base_vel_ = nh_.subscribe("imu_base", 100, &Mobile_Leveling_Platform::imuCalculation_base, this);
  imu_upper_vel_= nh_.subscribe("imu_upper", 100, &Mobile_Leveling_Platform::imuCalculation_upper, this);

  // for pbvs
  //imu_base_vel_ = nh_.subscribe("platform/imu_base", 100, &Mobile_Leveling_Platform::imuCalculation_base, this);
  //imu_upper_vel_= nh_.subscribe("platform/imu_upper", 100, &Mobile_Leveling_Platform::imuCalculation_upper, this);

  //camera_imu_vel_ = nh_.subscribe("/imu_camera",1, &Mobile_Leveling_Platform::gimbal_control, this);
  //model_pose_   = nh_.subscribe("/gazebo/model_states", 100, &Mobile_Leveling_Platform::updatepose, this);

  prev_update_time_ = ros::Time::now();

  // initialize client

  //req_set_pose_ = nh_.serviceClient<gazebo_msgs::SetModelState>("gazebo/set_model_state",100);
  return true;
}

/*******************************************************************************
* Callback function for cmd_vel msg
*******************************************************************************/
void Mobile_Leveling_Platform::commandVelocityCallback(const geometry_msgs::TwistConstPtr& cmd_vel_msg)
{
  last_cmd_vel_time_ = ros::Time::now();

  target_linear_vel = goal_linear_velocity_  = cmd_vel_msg->linear.x;
  target_angular_vel = goal_angular_velocity_ = cmd_vel_msg->angular.z;

  wheel_speed_cmd_[LEFT]  = goal_linear_velocity_ + (goal_angular_velocity_ * wheel_seperation_ / 2);
  wheel_speed_cmd_[RIGHT]  = goal_linear_velocity_ - (goal_angular_velocity_ * wheel_seperation_ / 2);

  target_linear_vel = checkLinearLimitVelocity(target_linear_vel);
  target_angular_vel = checkAngularLimitVelocity(target_angular_vel);
  control_linear_vel = makeSimpleProfile(control_linear_vel, target_linear_vel, wheel_acc);
  control_angular_vel = makeSimpleProfile(control_angular_vel, target_angular_vel, wheel_acc);
}
void Mobile_Leveling_Platform::updategeometry(void)
{
    geometry_.linear.x = control_linear_vel;
    geometry_.linear.y = 0;
    geometry_.linear.z = 0;
    geometry_.angular.x = 0;
    geometry_.angular.y = 0;
    geometry_.angular.z = 0;
}
/*******************************************************************************
* make velocity profile
*******************************************************************************/
float Mobile_Leveling_Platform::constrain(float input, float low, float high)
{
    if(input<low)
        input=low;
    else if(input>high)
        input=high;
    else
        input=input;

    return input;
}

float Mobile_Leveling_Platform::checkLinearLimitVelocity(float vel)
{
    vel=constrain(vel,-MAX_LINEAR_VELOCITY,MAX_LINEAR_VELOCITY);
    return vel;
}

float Mobile_Leveling_Platform::checkAngularLimitVelocity(float vel)
{
    vel=constrain(vel,-MAX_ANGULAR_VELOCITY,MAX_ANGULAR_VELOCITY);
    return vel;
}

float Mobile_Leveling_Platform::makeSimpleProfile(float output, float input, float slop)
{

    if(input>0)
    {
        if(input>output)
        {
            slop = slop + 0.1;
            output = fmin(input,output+slop);
        }
        else if(input<output)
        {
            slop = slop - 0.1;
            output = fmin(input,output-slop);
        }
        else
            output = input;

        return output;
        }

    else
    {
        {
        if(input>output)
        {
            slop = slop + 0.1;
            output = fmin(input,output+slop);
        }
        else if(input<output)
        {
            slop = slop - 0.1;
            output = fmin(input,output-slop);
        }
        else
            output = input;

        return output;
        }
    }
/*
    pre_time = clock();
    dt = (pre_time - past_time)/1000;
    past_time = pre_time;
    if(dt>1.5) 
    {
        ROS_WARN("High dt : %f",dt);
        output = output;
        return output;
    }

    pre_error = input-output;
    integral_error += pre_error*dt;
    d_error = pre_error-past_error;
    past_error = pre_error;

    antiwindup_trig = 0.9*input;

    if(output < antiwindup_trig)
    output = (pre_error*kp+integral_error*ki+d_error*kd/dt);
    else
    output = (pre_error*kp+(integral_error-0.03/kp)*ki+d_error*kd/dt);
    ROS_INFO("send de/dt = %f", d_error/dt);
    ROS_INFO("send pre = %f", pre_error);
    ROS_INFO("send past = %f", past_error);
    ROS_INFO("send dt = %f", dt);
    return output;
*/

}

/*******************************************************************************
* IMU Quaternion -> euler, Leveling maintenance
*******************************************************************************/
void Mobile_Leveling_Platform::imuCalculation_base(const sensor_msgs::ImuConstPtr &imu_base)
{

    Quaternion q;
    euler e;
    ekf ekf;

    float upper_roll;

    q.x = imu_base->orientation.x;
    q.y = imu_base->orientation.y;
    q.z = imu_base->orientation.z;
    q.w = imu_base->orientation.w;

    ekf.pre.x = ekf.accg.x;
    ekf.pre.y = ekf.accg.y;
    ekf.pre.z = ekf.accg.z;

    ekf.accg.x = imu_base->linear_acceleration.x;
    ekf.accg.y = imu_base->linear_acceleration.y;
    ekf.accg.z = imu_base->linear_acceleration.z;

    // roll (x-axis rotation)
    float sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1 - 2 * (q.w * q.x + q.y * q.y);
    e.r = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        e.p = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        e.p = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    e.y = std::atan2(siny_cosp, cosy_cosp);

    //ROS_INFO("base roll  : %f",RAD2DEG(e.r));
    //ROS_INFO("base pitch : %f",RAD2DEG(e.p));
    //ROS_INFO("base yaw   : %f",RAD2DEG(e.y));

    upper_roll = (floor(e.r*100))*0.01;

    angle_vel_.data = -upper_roll-DEG2RAD(0.572958)+angoffset;
    
    ekf.angle.p = e.r;
    ekf.angle.r = e.p;
    //EKF_acc();

    //ROS_INFO("check  : %f",RAD2DEG(upper_roll));
}
void Mobile_Leveling_Platform::imuCalculation_upper(const sensor_msgs::ImuConstPtr &imu_upper)
{
    Quaternion q;
    euler e;

    q.x = imu_upper->orientation.x;
    q.y = imu_upper->orientation.y;
    q.z = imu_upper->orientation.z;
    q.w = imu_upper->orientation.w;

    // roll (x-axis rotation)
    float sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1 - 2 * (q.w * q.x + q.y * q.y);
    e.r = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        e.p = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        e.p = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    e.y = std::atan2(siny_cosp, cosy_cosp);

    //ROS_INFO("upper roll  : %f",RAD2DEG(e.r));
    //ROS_INFO("upper pitch : %f",RAD2DEG(e.p));
    //ROS_INFO("upper yaw   : %f",RAD2DEG(e.y));


}

void Mobile_Leveling_Platform::angleOffset(const Mobile_Leveling_Platform_V2::joint_velConstPtr &jointoffset)
{
    angoffset = jointoffset->data;
}

void Mobile_Leveling_Platform::gimbal_control(const sensor_msgs::ImuConstPtr &camera_imu)
{
    Quaternion q;
    euler e;
    camera_euler camera_euler;

    q.w = camera_imu->orientation.w;
    q.x = camera_imu->orientation.x;
    q.y = camera_imu->orientation.y;
    q.z = camera_imu->orientation.z;

    //Mobile_Leveling_Platform::QuaterToEuler(q.x, q.y, q.z, q.w);

    float sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1 - 2 * (q.w * q.x + q.y * q.y);
    e.r = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        e.p = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        e.p = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    e.y = std::atan2(siny_cosp, cosy_cosp);

    camera_euler.roll = e.r;
    camera_euler.pitch = e.p;

    link_state_.link_name          = "iris::cgo3_camera_link";
    link_state_.twist.angular.z    = 0;
    link_state_.pose.orientation.x = 0;
    link_state_.pose.orientation.y = 0;
    link_state_.pose.orientation.z = 0;
    link_state_.pose.orientation.w = 1;
    link_state_.reference_frame    = "iris::cgo3_camera_link";

    int roll_ok = 0;

    //ROS_INFO("camera_roll: %f",camera_euler.roll);
    //ROS_INFO("camera_pitch: %f", camera_euler.pitch);

    float dt = 0.01;
    float KP_roll = 500;
    float KI_roll = 10;

    float angle_err_roll = camera_euler.roll; 
    float angle_err_pitch = camera_euler.pitch;
    float angle_err_roll_int = angle_err_roll_int + angle_err_roll*dt;
    float angle_err_pitch_int = angle_err_pitch_int + angle_err_pitch*dt;

    float angular_vel_roll = angle_err_roll*KP_roll+angle_err_roll_int*KI_roll;
    float angular_vel_pitch = angle_err_pitch*KP_roll+angle_err_pitch_int*KI_roll;

    link_state_.twist.angular.x = -angular_vel_roll;
    link_state_.twist.angular.y = -angular_vel_pitch;
}

void Mobile_Leveling_Platform::EKF_acc(void)
{
    ekf ekf;
    ros::Time current_time, last_time;

    last_time = current_time;
    current_time = ros::Time::now();
	ekf.dt = (current_time.toSec() - last_time.toSec());
    if(ekf.dt!=0)
    {
        ekf.accl.x = ekf.accl.x + (ekf.accg.x - ekf.pre.x) - g * sin(ekf.angle.p);
        ekf.accl.y = ekf.accl.y + (ekf.accg.y - ekf.pre.y) - g * sin(ekf.angle.r);
        ekf.accl.z = ekf.accl.z + (ekf.accg.z - ekf.pre.z) - g * cos(ekf.angle.r);
        ekf.acc = sqrt(pow(ekf.accl.x,2)+pow(ekf.accl.y,2)+pow(ekf.accl.z,2));
        if(ekf.accl.x != 0 || ekf.accl.y !=0 || ekf.accl.z != 0)
        {
            ekf.f(0,0) = 1;
            ekf.f(0,1) = 2*ekf.accl.x;//ekf.accg.x/ekf.acc;
            ekf.f(0,2) = 2*ekf.accl.y;//ekf.accg.y/ekf.acc;
            ekf.f(0,3) = 2*ekf.accl.z;//ekf.accg.z/ekf.acc;
            ekf.h(0,1) = 1;
            ekf.h(1,2) = 1;
            ekf.h(2,3) = 1;

            ekf.pred_x = ekf.f * ekf.x;
            ekf.pred_p = ekf.f * ekf.p * ekf.f.transpose() + ekf.Q;

            ekf.z << ekf.accl.x, ekf.accl.y, ekf.accl.z;
            ekf.y = ekf.z - ekf.h * ekf.pred_x;

            ekf.S = ekf.h * ekf.pred_p * ekf.h.transpose() + ekf.R;

            ekf.K = ekf.pred_p * ekf.h.transpose() * ekf.S.inverse();
            ekf.est_x = ekf.pred_x + ekf.K * ekf.y;
            try
            {   
                if(0) throw ekf.K;
                else
                {
                    ekf.est_p = (ekf.I - ekf.K * ekf.h) * ekf.pred_p;
                    ekf.x = ekf.est_x;
                    ekf.p = ekf.est_p;
                    kalman_vel_.estimate_acc = ekf.est_x(0,0);
                    kalman_vel_.estimate_acc_xl = ekf.est_x(1,0);
                    kalman_vel_.estimate_acc_yl = ekf.est_x(2,0);
                    kalman_vel_.estimate_acc_zl = ekf.est_x(3,0);
                }
            }
            catch(float err){ROS_ERROR("ERROR : %f",err);}
        }
        else
        {
            kalman_vel_.estimate_acc = ekf.acc;
            kalman_vel_.estimate_acc_xl = ekf.accl.x;
            kalman_vel_.estimate_acc_yl = ekf.accl.y;
            kalman_vel_.estimate_acc_zl = ekf.accl.z;            
        }
        
    }

}

void Mobile_Leveling_Platform::Mat_init(void)
{
    ekf ekf;
    float R_vel = 1;
    ekf.zero_mat << 0,0,0,0,
                    0,0,0,0,
                    0,0,0,0,
                    0,0,0,0;
    ekf.I << 1,0,0,0,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1;
    ekf.Q << ekf.I;
    ekf.f = ekf.zero_mat;
    ekf.p = ekf.zero_mat;
    ekf.pred_p = ekf.zero_mat;
    ekf.est_p = ekf.zero_mat;
    ekf.h << 0,0,0,0,
             0,0,0,0,
             0,0,0,0;
    ekf.est_x << 0,0,0,0;
    ekf.pred_x = ekf.est_x;
    ekf.x = ekf.est_x;
    ekf.z << 0,0,0;
    ekf.y << 0,0,0;
    ekf.R << R_vel,R_vel,R_vel,
             R_vel,R_vel,R_vel,
             R_vel,R_vel,R_vel;
    ekf.K << 0,0,0,
             0,0,0,
             0,0,0,
             0,0,0;
    ekf.S << 0,0,0,0,0,0,0,0,0;
}

float Mobile_Leveling_Platform::EulerToQuater(float roll, float pitch, float yaw)
{
    Quaternion q;
    euler e;

    float cp = cos(e.p*0.5);
    float sp = sin(e.p*0.5);
    float cr = cos(e.r*0.5);
    float sr = sin(e.r*0.5);
    float cy = cos(0);
    float sy = sin(0);

    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q.w,q.x,q.y,q.z;
}

float Mobile_Leveling_Platform::QuaterToEuler(float x, float y, float z, float w)
{
    Quaternion q;
    euler e;

    // roll (x-axis rotation)
    float sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1 - 2 * (q.w * q.x + q.y * q.y);
    e.r = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        e.p = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        e.p = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    e.y = std::atan2(siny_cosp, cosy_cosp);

    return e.r, e.p, e.y;
}


/*******************************************************************************
* Calculate the odometry
*******************************************************************************/
bool Mobile_Leveling_Platform::updateOdometry(ros::Duration diff_time)
{
  double wheel_l, wheel_r; // rotation value of wheel [rad]
  double delta_s, delta_theta;
  double v[2], w[2];

  wheel_l = wheel_r     = 0.0;
  delta_s = delta_theta = 0.0;

  v[LEFT]  = wheel_speed_cmd_[LEFT];
  w[LEFT]  = v[LEFT] / WHEEL_RADIUS;  // w = v / r
  v[RIGHT] = wheel_speed_cmd_[RIGHT];
  w[RIGHT] = v[RIGHT] / WHEEL_RADIUS;

  last_velocity_[LEFT]  = w[LEFT];
  last_velocity_[RIGHT] = w[RIGHT];

  wheel_l = w[LEFT]  * diff_time.toSec();
  wheel_r = w[RIGHT] * diff_time.toSec();

  if(isnan(wheel_l))
  {
    wheel_l = 0.0;
  }

  if(isnan(wheel_r))
  {
    wheel_r = 0.0;
  }

  last_position_[LEFT]  += wheel_l;
  last_position_[RIGHT] += wheel_r;

  delta_s     = WHEEL_RADIUS * (wheel_r + wheel_l) / 2.0;
  delta_theta = WHEEL_RADIUS * (wheel_r - wheel_l) / wheel_seperation_;

  // compute odometric pose
  odom_pose_[0] += delta_s * cos(odom_pose_[2] + (delta_theta / 2.0));
  odom_pose_[1] += delta_s * sin(odom_pose_[2] + (delta_theta / 2.0));
  odom_pose_[2] += delta_theta;

  // compute odometric instantaneouse velocity
  odom_vel_[0] = delta_s / diff_time.toSec();     // v
  odom_vel_[1] = 0.0;
  odom_vel_[2] = delta_theta / diff_time.toSec(); // w

  odom_.pose.pose.position.x = odom_pose_[0];
  odom_.pose.pose.position.y = odom_pose_[1];
  odom_.pose.pose.position.z = 0;
  odom_.pose.pose.orientation = tf::createQuaternionMsgFromYaw(odom_pose_[2]);

  // We should update the twist of the odometry
  odom_.twist.twist.linear.x  = odom_vel_[0];
  odom_.twist.twist.angular.z = odom_vel_[2];

  return true;
}

/*******************************************************************************
* Calculate the joint states
*******************************************************************************/
void Mobile_Leveling_Platform::updateJoint(void)
{
  joint_states_.position[LEFT]  = last_position_[LEFT];
  joint_states_.position[RIGHT] = last_position_[RIGHT];
  joint_states_.velocity[LEFT]  = last_velocity_[LEFT];
  joint_states_.velocity[RIGHT] = last_velocity_[RIGHT];
}

/*******************************************************************************
* Calculate the TF
*******************************************************************************/
void Mobile_Leveling_Platform::updateTF(geometry_msgs::TransformStamped& odom_tf)
{
  odom_tf.header = odom_.header;
  odom_tf.child_frame_id = odom_.child_frame_id;
  odom_tf.transform.translation.x = odom_.pose.pose.position.x;
  odom_tf.transform.translation.y = odom_.pose.pose.position.y;
  odom_tf.transform.translation.z = odom_.pose.pose.position.z;
  odom_tf.transform.rotation = odom_.pose.pose.orientation;
}

/*******************************************************************************
* Update function
*******************************************************************************/
bool Mobile_Leveling_Platform::update()
{
  ros::Time time_now = ros::Time::now();
  ros::Duration step_time = time_now - prev_update_time_;
  prev_update_time_ = time_now;

  // zero-ing after timeout
  if((time_now - last_cmd_vel_time_).toSec() > cmd_vel_timeout_)
  {
    wheel_speed_cmd_[LEFT]  = 0.0;
    wheel_speed_cmd_[RIGHT] = 0.0;
  }

  // odom
  updateOdometry(step_time);
  odom_.header.stamp = time_now;
  //odom_pub_.publish(odom_);

  // joint_states
  updateJoint();
  joint_states_.header.stamp = time_now;
  //joint_states_pub_.publish(joint_states_);

  // tf
  geometry_msgs::TransformStamped odom_tf;
  updateTF(odom_tf);
  //tf_broadcaster_.sendTransform(odom_tf);

  // geometry
  updategeometry();
  geometry_pub_.publish(geometry_);
  upperangle_pub_.publish(angle_vel_);

  //geometry_pub_RL.publish(geometry_);
  //upperangle_pub_RL.publish(angle_vel_);

  //kalman_pub_.publish(kalman_vel_);
  //gimbal_control_pub_.publish(link_state_);

  //req_set_pose_.call(set_model_state_);
  return true;
}
/*******************************************************************************
* Main function
*******************************************************************************/
int main(int argc, char* argv[])
{
  ros::init(argc, argv, "control");
  ros::NodeHandle nh;
  Mobile_Leveling_Platform MLP;
  ros::Rate loop_rate(20);

  while (ros::ok())
  {
    MLP.update();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

