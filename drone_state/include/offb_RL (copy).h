#ifndef OFFBOARD_H_
#define OFFBOARD_H_

#include "offb_RL.h"
#include <iostream>
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
#include <aruco_msgs/MarkerRecogn.h>
#include <geometry_msgs/TwistStamped.h>

#include <drl_uav/getEnvinfo.h>
#include <drl_uav/getRLflag.h>

#include <Mobile_Leveling_Platform_V2/joint_vel.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/ModelStates.h>
#include <fstream>
#include <string>

#define INIT 10
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

#define WEIGHT 0.8  ///LPF Parameter

#define MAXEPISODE 100000
#define MAXSTEP 1000
#define MLPVELOCITY -0.4

    class offboard
    {
        public :
        offboard();
        ~offboard();
        int run();
        bool initializing();
        float XYtransform(float &x_frame, float &y_frame);
        float rotatetransform(float &x_frame, float &y_frame, float angle);
        float saturation(float min, float max, float input);
        void pub_pose(float targetX, float targetY, float targetZ);
        void pub_vel(float targetVX, float targetVY, float targetVZ);
        void gimbal_set();

        static int drone_state_;

        struct mlp_gps_ {static float lat,lon,alt; }mlp_gps;
        struct globalpose_ {float mlp_x,mlp_y,mlp_z,drone_x,drone_y,drone_z;} globalpose_;
        struct iris_gps_{ float lat,lon,alt; } iris_gps_;
        struct gps_distance_ { float x,y,meter; } gps_distance_;
        struct mark_pixel_{ float x,y,z; } mark_pixel;
        struct iris_local_pose_{static float x,y,z; } iris_local_pose;
        struct Angle_{ static float roll[3],pitch[3];
                              float x,y,z,w;
                       static float yaw; } angle;
        struct iris_vel_{static float x,y,z;}iris_vel;
        struct target_{
                        static float curr_x, curr_y;
                        float pre_x, pre_y;
                        float distance[4];
                        float vel_x, vel_y;
                        float int_x, int_y, dif_x, dif_y;
                    }target;
        float path_iris_gps_lat = 0;
        float path_iris_gps_lon = 0;
        float view_angle = 40*DEG2RAD;
        float dt = 0;
        float descent_rate = 0;
        float mark_buf[3][2] = {0,};

        float beta = 0;
        float sec = 0;
        float init_target_x = 0;
        float init_target_y = 0;
        float des_iris_alt = 0;
        float trackingTime = 0;

        int arming_fail_cnt = 0;
        static bool markerrecogn;
        bool aruco_done = false;
        bool init_state = true;
        bool test = false;
        int landmode;
        bool gpson = false;
        bool write;
        
        private :
        ros::NodeHandle offb_nh;
        ros::NodeHandle nh_priv;

        ros::Time current_time, last_time;

        mavros_msgs::State current_state;
        mavros_msgs::CommandBool arm_cmd;
        mavros_msgs::SetMode offb_set_mode;
        mavros_msgs::MountConfigure mount_configure;
        geometry_msgs::Twist vel;
        geometry_msgs::PoseStamped pose;

        ros::Subscriber state_sub;
        ros::Publisher local_pos_pub;
        ros::Publisher local_vel_pub;
        ros::ServiceClient arming_client;
        ros::ServiceClient set_mode_client;
        ros::Subscriber mlp_local_pose;        
        ros::Subscriber MLP_GPS;
        ros::Subscriber iris_GPS;
        ros::Subscriber iris_local_pos_sub;
        ros::Subscriber marker_pose;
        ros::Subscriber iris_imu;
        ros::Subscriber MarkerRecogn;
        ros::Subscriber iris_vel_sub;

        ros::ServiceClient mount_client;

        void iris_initializing();
        bool iris_arming();
        void iris_offb_start();
        void iris_GPS_tracking();
        void iris_VISION_tracking();
        void Pub_targetpose();
        void iris_landing();
        bool check_trackingtime();
        
        void  state_cb(const mavros_msgs::State::ConstPtr& msg);
        void  get_iris_localposition(const geometry_msgs::PoseStampedConstPtr &localP);
        void  get_MLP_position(const sensor_msgs::NavSatFixConstPtr &MLP_GPS );
        void  get_iris_position(const sensor_msgs::NavSatFixConstPtr &iris_GPS);
        void  mlp_global(const gazebo_msgs::ModelStatesConstPtr &global_);
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
                        float init_marker_x, float init_marker_y);

    };

    class RL_env
    {
        public :
        RL_env();
        ~RL_env();
        bool initializing();

        struct state_{ float vel[3];
                       float rel_pose[3];
                       float yaw; }state;
        struct action_{ static float vel[3];
                        static int landing; }action;
        struct env_info_{struct state_ state;
                         struct action_ action;
                         static float reward;
                         bool restart;
                         static int episode;
                         static int step;
                         bool markrecogn;
                         }env_info;

        static float save_iris_x, save_iris_y;          
        static int mlp_state_;
        static float real_distance;
        static float mlp_distance;
        const float alpha = 10;
        const float gamma = 0.1;
        static int success_region_step;
        float angle;

        void SetModelvelReset(float vx, float vy, float vz);
        void SetModelReset(bool trans, std::string model_name, float pos_x, float pos_y, float pos_z, float vx, float vy, float vz);
        void MLPvelPub(float setpointV, float setpointAng);
        void MLPcontrol(void);
        void exchange_stateNaction(void);
        void get_uavstate(offboard offb);
        void iris_RL(void);
        float calcul_reward(bool land_command, int step);
        bool check_reset();

        gazebo_msgs::ModelState reset_pose;
        gazebo_msgs::SetModelState reset;
        drl_uav::getRLflag flag;
        drl_uav::getEnvinfo RLenvinfo;
        geometry_msgs::Twist mlp_pose_msg;
        Mobile_Leveling_Platform_V2::joint_vel mlp_angle_msg;

        ros::Publisher  mlp_vel_pub;
        ros::Publisher  mlp_ang_pub;
        ros::ServiceClient reset_client;
        ros::ServiceClient rlflag_client;
        ros::ServiceClient state_client;

        private :
        ros::NodeHandle RL_nh;
        ros::NodeHandle RL_priv_nh;
    };
    float offboard::Angle_::yaw, offboard::Angle_::roll[3], offboard::Angle_::pitch[3];
    float offboard::mlp_gps_::lat, offboard::mlp_gps_::lon, offboard::mlp_gps_::alt = 0;
    float offboard::iris_local_pose_::x, offboard::iris_local_pose_::y, offboard::iris_local_pose_::z = 0;
    int   offboard::drone_state_ = 0;
    float offboard::target_::curr_x, offboard::target_::curr_y = 0;
    float offboard::iris_vel_::x, offboard::iris_vel_::y, offboard::iris_vel_::z = 0;
    bool  offboard::markerrecogn = false;
    float RL_env::real_distance = 0;
    int   RL_env::mlp_state_ = MLPSTOP;
    int   RL_env::env_info_::step, RL_env::env_info_::episode = 0;
    float RL_env::save_iris_x=0, RL_env::save_iris_y = 0;
    float RL_env::action_::vel[3];
    int   RL_env::action_::landing;
    float RL_env::env_info_::reward;
    float RL_env::mlp_distance;
    int RL_env::success_region_step;
#endif
