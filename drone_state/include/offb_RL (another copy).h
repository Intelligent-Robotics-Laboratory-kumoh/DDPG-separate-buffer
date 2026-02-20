#ifndef OFFBOARD_H_
#define OFFBOARD_H_

#include "offb_RL.h"
#include <iostream>
#include <tuple>
#include <ctime>
#include <ros/ros.h>
#include <ros/time.h>
#include <math.h>
#include </home/baek/rl_ws/src/drone_state/include/matrix/matrix/math.hpp>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/NavSatFix.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandTOL.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/MountConfigure.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/PositionTarget.h>
#include <aruco_msgs/MarkerRecogn.h>
#include <geometry_msgs/TwistStamped.h>

#include <drl_uav/getEnvinfo.h>
#include <drl_uav/getLandingState.h>
#include <drl_uav/getEnvinfo_att.h>
#include <drl_uav/log_data.h>

#include <Mobile_Leveling_Platform_V2/joint_vel.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/ContactsState.h>
#include <gazebo_msgs/GetJointProperties.h>
#include <gazebo_msgs/SetLinkState.h>
#include <gazebo_msgs/GetLinkState.h>
#include <fstream>
#include <string>
#include <tuple>

#define ARM 0
#define OFFBOARD 1
#define GPS_TRACKING 2
#define VISION_TRACKING 3
#define LAND 4
#define OFF 5
#define RL 6        // for RL training
#define RESET 7
#define MLPSTART 8
#define MLPSTOP  9
#define INIT 10
#define STABLE 11

#define EARTH_RADIUS 6371.009   /// calculation a distance used gps data
#define LAT30 110.852
#define LAT45 111.132
#define SEC2USEC 1000000.0f ///slewrate parameter
#define MPC_ACC_HOR 3       
#define MPC_XY_CRUISE 12
#define MAXARMINGTRY 100     /// 

#define RAD2DEG 180/M_PI
#define DEG2RAD M_PI/180

#define EXP 2.71828182846   /// sigmoid Variable gain Parameter
#define BETA_MAX 3.5
#define GRADIENT -0.7//-0.7
#define SIGMOID_TH 3.5//3

#define WEIGHT 0.5  ///LPF Parameter

#define MAXEPISODE 100000000
#define MAXSTEP 200
#define MLPVELOCITY -0.4

#define MAX_VXY  1
#define MIN_VXY -1
#define MAX_VZ   0
#define MIN_VZ  -1

//#define ENV_RAMOS 
//#define TRACKING
    class offboard
    {
        public :
        offboard();
        ~offboard();
        int run();
        bool initializing();
        float XYtransform(float &x_frame, float &y_frame);
        float rotatetransform(float &x_frame, float &y_frame, float angle);
        float HMGtransform(float &x_frame, float &y_frame, float angle, float z, float x_offset, float y_offset);
        float saturation(float min, float max, float input);
        float PID_controller(float p_gain, float i_gain, float d_gain, float observe, float goal);
        void pub_pose(float targetX, float targetY, float targetZ, float ori_x, float ori_y, float ori_z, float ori_w);
        void pub_vel(float targetVX, float targetVY, float targetVZ);
        void pub_sp_raw(float targetVX, float targetVY, float targetVZ, float yaw);
        void gimbal_set();
        std::tuple<float, float, float, float, float, float, float> 
             get_link_state(std::string link_name, bool get_orientation, bool get_position, std::string ref_frame);        
        
        static int drone_state_;

        struct mlp_ori_ {static float mlp_roll,mlp_pitch,mlp_yaw;}mlp_ori_;
        struct mlp_gps_ {static float lat,lon,alt; }mlp_gps;
        struct globalpose_ {float mlp_x,mlp_y,mlp_z,drone_x,drone_y,drone_z,mlp_roll,mlp_pitch,mlp_yaw;} globalpose_;
        struct globalori_ {float mlp_x,mlp_y,mlp_z,mlp_w;} globalori_;
        struct globalvel_ {float mlp_x,mlp_y,mlp_z;}globalvel_;
        struct iris_gps_{ float lat,lon,alt; } iris_gps_;
        struct gps_distance_ { static float x,y,meter; } gps_distance;
        struct mark_pixel_{ float x,y,z; } mark_pixel;
        struct iris_local_pose_{static float x,y,z; } iris_local_pose;
        struct Angle_{ static float roll[3],pitch[3];
                       static float x,y,z,w;
                       static float yaw; } angle;
        struct iris_vel_{static float x,y,z,yaw_speed;}iris_vel;
        struct target_{
                        static float curr_x, curr_y;
                        float pre_x, pre_y;
                        float distance[4];
                        float vel_x, vel_y;
                        float int_x, int_y, dif_x, dif_y;
                    }target;
        struct pid_{
                    float error,prev_error,int_error;
                   }pid;
        struct gimbal_{ std::string roll_joint_name = "iris::cgo3_horizontal_arm_joint";
                        std::string pitch_joint_name = "iris::cgo3_camera_joint";
                        std::string yaw_joint_name = "iris::cgo3_vertical_joint";
                        std::string roll_link_name = "iris::cgo3_horizontal_arm_link";
                        std::string pitch_link_name = "iris::cgo3_camera_link";
                        std::string yaw_link_name = "iris::cgo3_vertical_arm_link";
                        float joint_position[3];
                        float link_orientation[3];}gimbal;
        std::string marker_id;
        float path_iris_gps_lat = 0;
        float path_iris_gps_lon = 0;
        float view_angle = 40*DEG2RAD;
        static float dt;
        float descent_rate = 0;
        float mark_buf[3][2] = {0,};

        float beta = 0;
        float sec = 0;
        float init_target_x = 0;
        float init_target_y = 0;
        float init_velocity_x = 0;
        float init_velocity_y = 0;
        float init_rel_pose_x =0;
        float init_rel_pose_y =0;
        float des_iris_alt = 0;
        float trackingTime = 0;

        float mlp_traj[3][300];
        float drone_traj[3][300];

        uint16_t mask_set;
        int arming_fail_cnt = 0;
        static bool markerrecogn;
        bool aruco_done = false;
        bool init_state = true;
        bool test = false;
        int landmode;
        bool gpson = false;
        static bool write;
        int past_episode;
        private :
        ros::NodeHandle offb_nh;
        ros::NodeHandle nh_priv;

        ros::Time current_time, last_time;

        mavros_msgs::AttitudeTarget att_;
        mavros_msgs::State current_state;
        mavros_msgs::CommandBool arm_cmd;
        mavros_msgs::SetMode offb_set_mode;
        mavros_msgs::MountConfigure mount_configure;
        mavros_msgs::PositionTarget sp_raw;
        geometry_msgs::Twist vel;
        geometry_msgs::PoseStamped pose;
        gazebo_msgs::GetJointProperties JointState;
        gazebo_msgs::GetLinkState LinkState;
        gazebo_msgs::SetLinkState setlinkstate;

        ros::Subscriber state_sub;
        ros::Publisher local_pos_pub;
        ros::Publisher local_vel_pub;
        ros::Publisher local_att_pub;
        ros::Publisher local_sp_pub;
        ros::ServiceClient arming_client;
        ros::ServiceClient set_mode_client;
        ros::ServiceClient joint_property_client;
        ros::Subscriber model_global_pose;        
        ros::Subscriber MLP_GPS;
        ros::Subscriber iris_GPS;
        ros::Subscriber iris_local_pos_sub;
        ros::Subscriber marker_pose;
        ros::Subscriber iris_imu;
        ros::Subscriber MarkerRecogn;
        ros::Subscriber iris_vel_sub;

        ros::ServiceClient mount_client;
        ros::ServiceClient setlink_client;
        ros::ServiceClient getlink_client;

        void  iris_initializing();
        bool  iris_arming();
        void  iris_offb_start();
        void  iris_GPS_tracking();
        void  iris_VISION_tracking();
        void  Pub_targetpose();
        void  iris_landing();
        bool  check_trackingtime();
        float Quater_Rotation(float roll, float pitch, float yaw);
        void  set_link_orientation(std::string name, float roll, float pitch, float yaw);
        void  iris_att_pub(float roll, float pitch, float yaw, float thrust);
        void  get_joint_state();
        float TransformationQE(std::string euler, float x, float y, float z, float w);
        void  state_cb(const mavros_msgs::State::ConstPtr& msg);
        void  get_iris_localposition(const geometry_msgs::PoseStampedConstPtr &localP);
        void  get_MLP_position(const sensor_msgs::NavSatFixConstPtr &MLP_GPS );
        void  get_iris_position(const sensor_msgs::NavSatFixConstPtr &iris_GPS);
        void  model_globalpose(const gazebo_msgs::ModelStatesConstPtr &global_);
        void  get_iris_vel(const geometry_msgs::TwistStampedConstPtr &uav_vel);
        float distance_measure(float platform_lat, float platform_lon, float drone_lat, float drone_lon);
        void  markrecogn_callback(const aruco_msgs::MarkerRecogn::ConstPtr& markRecogn_);
        void  slewrate(float &sp_x, float &sp_y, float &sp_vel_x, float &sp_vel_y);
        void  Quarter2Euler(const geometry_msgs::PoseStampedConstPtr &iris_angle);
        void  markerPixelGet(const geometry_msgs::PointStampedConstPtr &markerpixel);
        bool  anglecheck(float roll, float pitch);
        float LPF(float data_pre, float data_now);
        float sigmoid_beta(float init_target_distance, float error);
        float descent_flight(float vel, float goal);
        void  dataWrite(float target_x, float target_y, float target_z,
                        float pose_x, float pose_y, float pose_z, 
                        float init_marker_x, float init_marker_y, int episode, int end_step);


    };

    class RL_env
    {
        public :
        RL_env();
        ~RL_env();
        bool initializing();

        struct state_{ static float vel[3];
                       static float rel_pose[3];
                       static float rel_pose_dot[3];
                       static float yaw;
                       static float yaw_speed; }state;
        struct action_{ static float vel[3];
                        static float landing_cmd;
                        static float pitch,roll,yaw_speed,thrust; }action;
        struct env_info_{struct state_ state;
                         struct action_ action;
                         static float reward;
                         static bool restart;
                         static int episode;
                         static int step;
                         static bool markrecogn;
                         static bool reset_error;
                         }env_info;

        static float save_iris_x, save_iris_y;          
        static int mlp_state_;
        static float real_distance;
        static float mlp_distance;
        static float landing_start_alt;
        static bool contact_recogn_mark;
        static bool contact_recogn_upper;

        static float real_x;
        static float real_y;
        static float real_z;
        static bool error;
        const float alpha = 10;
        const float gamma = 0.1;
        static int success_region_step;
        static float angle;
        static float pre_shape;
        float pre_target_x, pre_target_y;
        float rd_pos = 0;
        
        void contact_mark(const gazebo_msgs::ContactsStateConstPtr &mark_contact);
        void contact_upper(const gazebo_msgs::ContactsStateConstPtr &upper_contact);
        void SetModelvelReset(float vx, float vy, float vz);
        void SetModelReset(bool trans, std::string model_name, float pos_x, float pos_y, float pos_z, float vx, float vy, float vz);
        void MLPvelPub(float setpointV, float setpointAng);
        void MLPcontrol(void);
        void exchange_stateNaction();
        void get_uavstate();
        void iris_RL();
        float calcul_reward(offboard offb, int step, bool negative);
        void publish_logdata(float uav_x, float uav_y, float uav_z,
                             float target_x, float target_y, float target_z, 
                             int episode, int restart, int landingstate, int step);
        gazebo_msgs::ModelState reset_pose;
        gazebo_msgs::SetModelState reset;
        gazebo_msgs::ContactsState mark_cs, upper_cs;

#ifdef ENV_RAMOS
        drl_uav::getEnvinfo_att RLenvinfo;
#else
        drl_uav::getEnvinfo RLenvinfo;
#endif

        geometry_msgs::Twist mlp_pose_msg;
        Mobile_Leveling_Platform_V2::joint_vel mlp_angle_msg;
        drl_uav::log_data sendlogdata;

        ros::Publisher  mlp_vel_pub;
        ros::Publisher  mlp_ang_pub;
        ros::Publisher  logdata_pub;
        ros::Subscriber mark_contact_sub;
        ros::Subscriber upper_contact_sub;
        ros::ServiceClient reset_client;
        ros::ServiceClient state_client;

        private :
        ros::NodeHandle RL_nh;
        ros::NodeHandle RL_priv_nh;
    };
    float offboard::dt=0;
    float offboard::Angle_::yaw, offboard::Angle_::roll[3], offboard::Angle_::pitch[3];
    float offboard::Angle_::x, offboard::Angle_::y, offboard::Angle_::z, offboard::Angle_::w;
    float offboard::mlp_gps_::lat, offboard::mlp_gps_::lon, offboard::mlp_gps_::alt = 0;
    float offboard::iris_local_pose_::x, offboard::iris_local_pose_::y, offboard::iris_local_pose_::z = 0;
    int   offboard::drone_state_ = 0;
    float offboard::target_::curr_x, offboard::target_::curr_y = 0;
    float offboard::iris_vel_::x, offboard::iris_vel_::y, offboard::iris_vel_::z = 0, offboard::iris_vel_::yaw_speed = 0;
    bool  offboard::markerrecogn;
    float offboard::gps_distance_::x, offboard::gps_distance_::y;
    float offboard::gps_distance_::meter;
    float offboard::mlp_ori_::mlp_pitch=0, offboard::mlp_ori_::mlp_roll=0, offboard::mlp_ori_::mlp_yaw=0; 
    float RL_env::real_distance = 0;
    int   RL_env::mlp_state_ = MLPSTOP;
    int   RL_env::env_info_::step;
    int   RL_env::env_info_::episode;
    float RL_env::state_::rel_pose[3];
    float RL_env::state_::vel[3];
    float RL_env::state_::yaw;
    float RL_env::state_::yaw_speed;
    float RL_env::save_iris_x=0, RL_env::save_iris_y = 0;
    float RL_env::action_::vel[3];
    float RL_env::action_::landing_cmd;
    float RL_env::env_info_::reward;
    float RL_env::mlp_distance;
    bool  RL_env::env_info_::markrecogn;
    int RL_env::success_region_step;
    float RL_env::action_::roll, RL_env::action_::pitch, RL_env::action_::yaw_speed, RL_env::action_::thrust;
    float RL_env::state_::rel_pose_dot[3];
    bool RL_env::env_info_::restart;
    float RL_env::angle; float RL_env::pre_shape;
    float RL_env::real_x, RL_env::real_y, RL_env::real_z, RL_env::landing_start_alt;
    bool RL_env::env_info_::reset_error, RL_env::contact_recogn_mark=false, RL_env::contact_recogn_upper=false;
    bool offboard::write=false;
#endif

