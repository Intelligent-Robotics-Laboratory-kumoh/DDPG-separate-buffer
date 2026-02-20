#include <iostream>
#include <ros/ros.h>
#include <ros/time.h>
#include <math.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/NavSatFix.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandTOL.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <aruco_msgs/MarkerRecogn.h>
#include <geometry_msgs/TwistStamped.h>
#include </home/baek/rl_ws/src/drone_state/include/matrix/matrix/math.hpp>
#include <gazebo_msgs/ModelStates.h>
#include <fstream>
#include <string>

#define EARTH_RADIUS 6371.009   /// calculation a distance used gps data
#define LAT30 110.852
#define LAT45 111.132
#define SEC2USEC 1000000.0f ///slewrate parameter
#define MPC_ACC_HOR 3       
#define MPC_XY_CRUISE 12
#define MAXARMINGTRY 10     /// no use

#define RAD2DEG 180/M_PI
#define DEG2RAD M_PI/180

#define EXP 2.71828182846   /// sigmoid Variable gain Parameter
#define BETA_MAX 4
#define GRADIENT -1.5
#define SIGMOID_TH 3

#define WEIGHT 0.8  ///LPF Parameter


float get_iris_position(const sensor_msgs::NavSatFixConstPtr &iris_GPS);
float get_MLP_position(const sensor_msgs::NavSatFixConstPtr &MLP_GPS);
float distance_measure(float platform_lat, float platform_lon, float drone_lat, float drone_lon);
void  markrecogn(const aruco_msgs::MarkerRecogn::ConstPtr& markRecogn_);
void  get_iris_vel(const geometry_msgs::TwistStampedConstPtr &iris_vel);
float XYtranform(float &x_frame, float &y_frame);
void  slewrate(float &sp_x, float &sp_y, float &sp_vel_x, float &sp_vel_y);
void  dataWrite(float target_x, float target_y, float target_z, float pose_x, float pose_y, float pose_z);
bool  anglecheck(float roll, float pitch);
float LPF(float data_pre, float data_now);
float saturation(float min, float max, float input);

struct mlp_gps_ { float lat,lon,alt; } mlp_gps_;
struct mlp_glopos_ {float x,y,z;} mlp_glopos_;
struct iris_gps_{ float lat,lon,alt; } iris_gps_;
struct gps_distance_ { float x,y,meter; } gps_distance_;
struct mark_pixel_{ float x,y,z; } mark_pixel;
struct iris_local_pose_{ float x,y,z; } iris_local_pose;
struct Angle_{ float roll[3],pitch[3],yaw,x,y,z,w; } angle;
struct target_{
                float curr_x, curr_y, pre_x, pre_y;
                float distance[4];
                float vel_x, vel_y;
                float int_x, int_y, dif_x, dif_y;
              }target;


float path_iris_gps_lat = 0;
float path_iris_gps_lon = 0;
float vel_x, vel_y;
float view_angle = 40*DEG2RAD;
float dt = 0;
float target_distance = 0;
float target_distance_pre = 0;
float descent_rate = 0;
float mark_buf[3][2] = {0,};

float beta = 0;
float sec = 0;
float init_target_x = 0;
float init_target_y = 0;

int arming_fail_cnt = 0;
bool markerrecogn = false;
bool aruco_done = false;
bool init_state = true;
bool test = false;

std::string drone_state_ ;
ros::Time current_time, last_time;
mavros_msgs::State current_state;

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

float get_iris_localposition(const geometry_msgs::PoseStampedConstPtr &localP)
{
    iris_local_pose.x = localP->pose.position.x;
    iris_local_pose.y = localP->pose.position.y;
    iris_local_pose.z = localP->pose.position.z;
    return iris_local_pose.x, iris_local_pose.y, iris_local_pose.z;
}

float get_MLP_position(const sensor_msgs::NavSatFixConstPtr &MLP_GPS )
{
    mlp_gps_.lat = MLP_GPS->latitude;
    mlp_gps_.lon = MLP_GPS->longitude;
    mlp_gps_.alt = MLP_GPS->altitude;

    return mlp_gps_.alt, mlp_gps_.lon, mlp_gps_.lat;
}

float mlp_global(const gazebo_msgs::ModelStatesConstPtr &mlp_global_)
{
    mlp_glopos_.x = mlp_global_->pose[2].position.x;
    mlp_glopos_.y = mlp_global_->pose[2].position.y;
    mlp_glopos_.z = mlp_global_->pose[2].position.z;

    return mlp_glopos_.x, mlp_glopos_.y, mlp_glopos_.z;
}

float get_iris_position(const sensor_msgs::NavSatFixConstPtr &iris_GPS)
{
    iris_gps_.lat = iris_GPS->latitude;
    iris_gps_.lon = iris_GPS->longitude;
    iris_gps_.alt = iris_GPS->altitude;

    return iris_gps_.alt, iris_gps_.lat, iris_gps_.lon;
}

void get_iris_vel(const geometry_msgs::TwistStampedConstPtr &iris_vel)
{
    vel_x = iris_vel->twist.linear.x;
    vel_y = iris_vel->twist.linear.y;
}

float LPF(float data_pre, float data_now)
{
    float lpf_output = WEIGHT*data_pre + (1-WEIGHT)*data_now;
    return lpf_output;
}

float saturation(float min, float max, float input)
{
    float output = 0;
    if(input < min) return output = min;
    else if(input < max) return output = input;
    else return output = max;
}

float distance_measure(float platform_lat, float platform_lon, float drone_lat, float drone_lon)
{
    float avglat = (mlp_gps_.lat+iris_gps_.lat)/2;
    float lat = LAT30+(avglat-30)*((LAT45-LAT30)/15);
    float lon = cos(avglat*M_PI/180)*2*M_PI*EARTH_RADIUS;

    float lat_d = lat;
    float lat_m = lat_d / 60;
    float lat_s = lat_m / 60;

    float lon_d = lon / 360;
    float lon_m = lon_d / 60;
    float lon_s = lon_m / 60;

    float platform_lat_d = int(platform_lat);
    float platform_lat_m = int((platform_lat - int(platform_lat_d)) * 60);
    float platform_lat_s = ((platform_lat - int(platform_lat_d)) * 60 - int(platform_lat_m)) * 60;

    float drone_lat_d = int(drone_lat);
    float drone_lat_m = int((drone_lat - int(drone_lat_d)) * 60);
    float drone_lat_s = ((drone_lat - int(drone_lat_d)) * 60 - int(drone_lat_m)) * 60;

    float lat_d_distance = platform_lat_d - drone_lat_d;
    float lat_m_distance = platform_lat_m - drone_lat_m;
    float lat_s_distance = platform_lat_s - drone_lat_s;

    float lat_distance = (lat_d_distance * lat_d) + (lat_m_distance * lat_m) + (lat_s_distance * lat_s);

    float platform_lon_d = int(platform_lon);
    float platform_lon_m = int((platform_lon - int(platform_lon_d)) * 60);
    float platform_lon_s = ((platform_lon - int(platform_lon_d)) * 60 - int(platform_lon_m)) * 60;

    float drone_lon_d = int(drone_lon);
    float drone_lon_m = int((drone_lon - int(drone_lon_d)) * 60);
    float drone_lon_s = ((drone_lon - int(drone_lon_d)) * 60 - int(drone_lon_m)) * 60;

    float lon_d_distance = platform_lon_d - drone_lon_d;
    float lon_m_distance = platform_lon_m - drone_lon_m;
    float lon_s_distance = platform_lon_s - drone_lon_s;

    float lon_distance = (lon_d_distance * lon_d) + (lon_m_distance * lon_m) + (lon_s_distance * lon_s);
    float distance = sqrt(lat_distance*lat_distance + lon_distance*lon_distance);

    gps_distance_.meter = distance * 1000;
    gps_distance_.x = lat_distance * 1000;
    gps_distance_.y = lon_distance * 1000;

    return gps_distance_.x, gps_distance_.y, gps_distance_.meter;
}

float markerPixelGet(const geometry_msgs::PointStampedConstPtr &markerpixel)
{
        mark_buf[0][0] = mark_buf[2][0];
        mark_buf[0][1] = mark_buf[2][1];

        if(!init_state) 
        {
            target.distance[0] = target.distance[1];
            target.distance[2] = target.distance[3];
        }

        mark_buf[1][0] = markerpixel->point.x;
        mark_buf[1][1] = markerpixel->point.y;
        
        mark_buf[2][0] = LPF(mark_buf[0][0], mark_buf[1][0]); 
        mark_buf[2][1] = LPF(mark_buf[0][1], mark_buf[1][1]);

        float correction_x = ((tan(view_angle)-tan(view_angle-angle.pitch[1]))*400/(iris_local_pose.z-mlp_gps_.alt)/tan(view_angle));
        float correction_y = ((tan(view_angle)-tan(view_angle-angle.roll[1]))*400/(iris_local_pose.z-mlp_gps_.alt)/tan(view_angle));

        float target_x = mark_buf[2][0]+correction_x;
        float target_y = mark_buf[2][1]+correction_y;

        target.curr_x = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[2][0]);
        target.curr_y = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[2][1]);

//        target.curr_x = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[1][0]);
//        target.curr_y = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[1][1]);

//        target.curr_x = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-saturation(0,800,target_x));
//        target.curr_y = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-saturation(0,800,target_y));
        
//        ROS_INFO("correction_x : %f, mark_x : %f, pitch : %f",target_x, mark_buf[2][0], angle.pitch[1]);
//        ROS_INFO("correction_y : %f, mark_y : %f, roll : %f",target_y, mark_buf[2][1], angle.roll[1]);

        target.dif_x = (target.pre_x - target.curr_x)/dt;
        target.dif_y = (target.pre_x - target.curr_x)/dt;

        target.pre_x = target.curr_x;
        target.pre_y = target.curr_y;

        XYtranform(target.curr_x,target.curr_y);
        slewrate(target.curr_x, target.curr_x, target.vel_x, target.vel_y);

        target.distance[1] = abs(sqrt(target.curr_x*target.curr_x + target.curr_y*target.curr_y));
        target.distance[3] = (target.distance[1] - target.distance[0])/dt;
        //ROS_INFO("distance : %f", target.distance[0]);

    return target.curr_y, target.curr_x, target.pre_x, target.pre_y;
}

float Quarter2Euler(const geometry_msgs::PoseStampedConstPtr &iris_angle)
{
    angle.x = iris_angle->pose.orientation.x;
    angle.y = iris_angle->pose.orientation.y;
    angle.z = iris_angle->pose.orientation.z;
    angle.w = iris_angle->pose.orientation.w;

    angle.roll[1] = angle.roll[0];
    angle.pitch[1] = angle.pitch[0];

    // roll (x-axis rotation)
    float sinr_cosp = 2 * (angle.w * angle.x + angle.y * angle.z);
    float cosr_cosp = 1 - 2 * (angle.w * angle.x + angle.y * angle.y);
    angle.roll[0] = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2 * (angle.w * angle.y - angle.z * angle.x);
    if (std::abs(sinp) >= 1)
        angle.pitch[0] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angle.pitch[0] = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (angle.w * angle.z + angle.x * angle.y);
    double cosy_cosp = 1 - 2 * (angle.y * angle.y + angle.z * angle.z);
    angle.yaw = std::atan2(siny_cosp, cosy_cosp);

    angle.roll[2] = (angle.roll[0]-angle.roll[1])/dt;
    angle.pitch[2] = (angle.pitch[0]-angle.roll[1])/dt;
    return angle.roll,angle.pitch,angle.yaw;
}

void markrecogn(const aruco_msgs::MarkerRecogn::ConstPtr& markRecogn_)
{
   markerrecogn = markRecogn_->MarkRecogn;
}

float sigmoid_beta(float init_target_distance, float error)
{
    /* vel
    float init_pose = abs(1/init_target_distance);
    float sigbeta = 0;
    if((abs(target.distance[4]-target.distance[3]))<0.01 || sec > 0)
    {
        sec += 0.001;
        sigmoid = (BETA_MAX-init_pose)/(1+pow(EXP,GRADIENT*(sec-SIGMOID_TH)))+init_pose;
        sigbeta = sigmoid;

        if(sigbeta < 0) sigbeta = 0;
        //ROS_INFO("beta : %f", sigbeta);
        
    }
    else sigbeta = (sec==0) ? init_targetpose : sigmoid; 
        //ROS_INFO("beta : %f", sigbeta);*/

    /* pose
    float init_targetpose = 0.1;
    sec += 0.001;

    //if(target.distance[1] > 2 && sec >= SIGMOID_TH) sec -= 0.001;
    
    float init_sigbeta = ((BETA_MAX-init_targetpose)/(1+pow(EXP,GRADIENT*(sec-SIGMOID_TH)))+init_targetpose)/(1+pow(EXP,-10*(sec-0.3)));

    if(init_sigbeta >= init_targetpose)
    {
        float sigbeta = (BETA_MAX-init_targetpose)/(1+pow(EXP,GRADIENT*(sec-SIGMOID_TH)))+init_targetpose; 
        //ROS_INFO("sigbeta : %f", sigbeta);
        return init_sigbeta;
    }
    else
    {
        //ROS_INFO("beta : %f", init_sigbeta);
        return init_sigbeta;
    }*/
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    float init = 0.5;

    float sigbeta;
    if(error <= 0.5*init_target_distance) test = true;
    if(test)
    {   
        sec += 0.001;
        sigbeta = (BETA_MAX-init)/(1+pow(EXP,GRADIENT*(sec-SIGMOID_TH)))+init+(1-init)/(1+pow(EXP, 10*(sec-0.1)));
        ROS_INFO("sigbeta : %f", sigbeta);
    }
    else
    {   
        sigbeta = 1;
        //sigbeta = (1-init)/(1+pow(EXP,10*(sec-0.3)));
        //ROS_INFO("sigbeta : %f", sigbeta);
    }
    return sigbeta;
    
}

void slewrate(float &sp_x, float &sp_y, float &sp_vel_x, float &sp_vel_y)
{
	matrix::Vector2f sp_curr(sp_x, sp_y);
	matrix::Vector2f _sp_pev(0, 0) ;
	matrix::Vector2f _sp_pev_prev(0, 0);

    last_time = current_time;
    current_time = ros::Time::now();

	dt = (current_time.toSec() - last_time.toSec());

	if (dt < 1) {
		// bad dt, can't divide by it
		return;
	}

	dt /= SEC2USEC;

	if (!last_time.toSec()) {
		// running the first time since switching to precland

		// assume dt will be about 50000us
		dt = 50000 / SEC2USEC;

		// set a best guess for previous setpoints for smooth transition
		_sp_pev_prev(0) = _sp_pev(0) - vel_x * dt;
		_sp_pev_prev(1) = _sp_pev(1) - vel_y * dt;
	}
    //last_time = ros::Time::now();
	// limit the setpoint speed to the maximum cruise speed
	matrix::Vector2f sp_vel = (sp_curr - _sp_pev) / dt; // velocity of the setpoints

	if (sp_vel.length() > MPC_XY_CRUISE)
    {
		sp_vel = sp_vel.normalized() * MPC_XY_CRUISE;
		sp_curr = _sp_pev + sp_vel * dt;
	}

	// limit the setpoint acceleration to the maximum acceleration
	matrix::Vector2f sp_acc = (sp_curr - _sp_pev * 2 + _sp_pev_prev) / (dt * dt); // acceleration of the setpoints

	if (sp_acc.length() > MPC_ACC_HOR) 
    {
		sp_acc = sp_acc.normalized() * MPC_ACC_HOR;
		sp_curr = _sp_pev * 2 - _sp_pev_prev + sp_acc * (dt * dt);
	}

	// limit the setpoint speed such that we can stop at the setpoint given the maximum acceleration/deceleration
	float max_spd = sqrtf(MPC_ACC_HOR * ((matrix::Vector2f)(_sp_pev - matrix::Vector2f(sp_x,
			      sp_y))).length());
	sp_vel = (sp_curr - _sp_pev) / dt; // velocity of the setpoints

	if (sp_vel.length() > max_spd*2) {
		sp_vel = sp_vel.normalized() * max_spd*2;
		sp_curr = _sp_pev + sp_vel * dt;
	}

	_sp_pev_prev = _sp_pev;
	_sp_pev = sp_curr;

	sp_x = sp_curr(0);
	sp_y = sp_curr(1);
    sp_vel_x = sp_vel(0);
    sp_vel_y = sp_vel(1);
}
float descent_flight(float vel, float goal)
{
    float descent_rate = vel*dt;
    float desire_height = iris_local_pose.z - descent_rate;

    if(desire_height<goal)  
    {
        desire_height = goal;
    }

    return desire_height;
}

float XYtranform(float &x_frame, float &y_frame)
{
    float curr_xframe = x_frame;
    float curr_yframe = y_frame;  
    return y_frame = curr_xframe, x_frame = curr_yframe;
}

void dataWrite(float pose_x, float pose_y, float pose_z, float mlp_x, float mlp_y, float mlp_z)
{
    std::ofstream fout;

    fout.open("/home/baek/data/pose_x.txt",std::ios_base::app);
    if(fout.is_open())
    {
        fout<<"\n";
        fout<<pose_x;
        fout.close();
    }

    fout.open("/home/baek/data/pose_y.txt",std::ios_base::app);
    if(fout.is_open())
    {
        fout<<"\n";
        fout<<pose_y;
        fout.close();
    }

    fout.open("/home/baek/data/pose_z.txt",std::ios_base::app);
    if(fout.is_open())
    {
        fout<<"\n";
        fout<<pose_z;
        fout.close();
    }

    fout.open("/home/baek/data/mlp_x.txt",std::ios_base::app);
    if(fout.is_open())
    {
        fout<<"\n";
        fout<<mlp_x;
        fout.close();
    }

    fout.open("/home/baek/data/mlp_y.txt",std::ios_base::app);
    if(fout.is_open())
    {
        fout<<"\n";
        fout<<mlp_y;
        fout.close();
    }

    fout.open("/home/baek/data/mlp_z.txt",std::ios_base::app);
    if(fout.is_open())
    {
        fout<<"\n";
        fout<<mlp_z;
        fout.close();
    }

}


int main(int argc, char **argv)
{
    mlp_gps_.alt = 0;
    mlp_gps_.lat = 0;
    mlp_gps_.lon = 0;
    iris_gps_.alt = 0;
    iris_gps_.lat = 0;
    iris_gps_.lon = 0;

    bool land_trig = false;
    float des_iris_alt = 10;
    int land_mode = 0;

    float ref_lat_iris = iris_gps_.lat;
    float ref_lon_iris = iris_gps_.lon;

    ros::init(argc, argv, "offb_node");
    ros::NodeHandle nh;

    mavros_msgs::SetMode offb_set_mode;
    geometry_msgs::PoseStamped pose;
    geometry_msgs::Twist vel;
    mavros_msgs::CommandBool arm_cmd;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("/mavros/state", 10, state_cb);
    ros::Subscriber mlp_local_pose = nh.subscribe<gazebo_msgs::ModelStates>
            ("/gazebo/model_states", 100, &mlp_global);
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("/mavros/setpoint_position/local", 10);
    ros::Publisher local_vel_pub = nh.advertise<geometry_msgs::Twist>
            ("/mavros/setpoint_velocity/cmd_vel_unstamped", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("/mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("/mavros/set_mode");
    ros::Subscriber MLP_GPS = nh.subscribe<sensor_msgs::NavSatFix>
            ("/platform/MLP/fix",100,&get_MLP_position);
    ros::Subscriber iris_GPS = nh.subscribe<sensor_msgs::NavSatFix>
            ("/mavros/global_position/global",100,&get_iris_position);
    ros::Subscriber iris_local_pos_sub = nh.subscribe<geometry_msgs::PoseStamped>
            ("/mavros/local_position/pose",100,&get_iris_localposition);
    ros::Subscriber marker_pose = nh.subscribe<geometry_msgs::PointStamped>
            ("/aruco_single/pixel",100,&markerPixelGet);
    ros::Subscriber iris_imu = nh.subscribe<geometry_msgs::PoseStamped>
            ("/mavros/local_position/pose",100,&Quarter2Euler);
    ros::Subscriber MarkerRecogn = nh.subscribe<aruco_msgs::MarkerRecogn>
            ("/aruco_single/markerRecogn",100,&markrecogn);
    ros::Subscriber iris_vel = nh.subscribe<geometry_msgs::TwistStamped>
            ("/mavros/local_position/velocity_local",100,&get_iris_vel);

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(100);

    while(ros::ok())
    {
        //distance_measure(mlp_gps_.lat, mlp_gps_.lon, iris_gps_.lat, iris_gps_.lon);

        //target_distance = abs(sqrt(target.curr_x*target.curr_x + target.curr_y*target.curr_y));

        //ROS_INFO("distance : %f", target_distance);


        //send a few setpoints before starting
        if(init_state)
        {
            for(int i = 10; ros::ok() && i > 0; --i){
                local_pos_pub.publish(pose);
                ros::spinOnce();
                rate.sleep();
            }
        }

        if(!markerrecogn && !init_state && iris_local_pose.z < 2)
        {
                mavros_msgs::SetMode land_set_mode;
                land_set_mode.request.custom_mode = "AUTO.LAND";
                if( set_mode_client.call(land_set_mode) &&  land_set_mode.response.mode_sent)
                {
                    ROS_INFO("Land Start");
                    sleep(1);
                }
                if(!current_state.armed) 
                {
                    ROS_INFO("Landing is over. Turn off offb system");
                    return 0;
                }
        }

        if(!land_trig)
        {
            if(target.distance[1] <= 1.5 && !init_state && markerrecogn && iris_local_pose.z <= 5.1) 
            {
                ROS_INFO("landing trigger ON");
                land_trig = true;
            }
            
            if(markerrecogn && !init_state)
            {   
                pose.pose.position.x = iris_local_pose.x + target.curr_x * sigmoid_beta(init_target_x,target.curr_x);//beta;
                pose.pose.position.y = iris_local_pose.y + target.curr_y * sigmoid_beta(init_target_y,target.curr_y);//beta;
                pose.pose.position.z = descent_flight(15, 5);
                beta = beta + 0.002;
		        if(beta > 1.1)	beta = 1.1;
            }

            else
            {
                if(init_state) // 마커 인식 ㅇ,X 초기 ㅇ
                {
                    pose.pose.position.x = iris_local_pose.x;
                    pose.pose.position.y = iris_local_pose.y;
                    pose.pose.position.z = des_iris_alt;
                    
                    init_target_x = iris_local_pose.x + target.curr_x;
                    init_target_y = iris_local_pose.y + target.curr_y;

                    if(iris_local_pose.z >= des_iris_alt-0.1)	init_state=false;
                }

                else //마커 인식 O,X 초기 X
                {                 
                    pose.pose.position.x = iris_local_pose.x + target.curr_x * sigmoid_beta(init_target_x,target.curr_x); //beta;
                    pose.pose.position.y = iris_local_pose.y + target.curr_y * sigmoid_beta(init_target_y,target.curr_y); //beta;
                    pose.pose.position.z = descent_flight(15, 5);
                    beta = beta + 0.002;                    
                    if(beta > 1.1)	beta = 1.1;
                }
            }

            if(current_state.mode != "OFFBOARD")
            {
                offb_set_mode.request.custom_mode = "OFFBOARD";
                set_mode_client.call(offb_set_mode);
            }

            if(current_state.armed != true )
            {
                arm_cmd.request.value = true;

                if (arming_client.call(arm_cmd) && arm_cmd.response.success)
                {
                    ROS_INFO("Vehicle armed");
                } 
                else 
                {
                    ROS_ERROR("Failed arming or disarming");
                }
            }

            ros::Time last_request = ros::Time::now();

            if( current_state.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(5.0)))
            {
                if( set_mode_client.call(offb_set_mode) &&  offb_set_mode.response.mode_sent)
                {
                    ROS_INFO("Offboard enabled");
                }
                last_request = ros::Time::now();
            }

            else
            {
                if( !current_state.armed && (ros::Time::now() - last_request > ros::Duration(5.0)))
                {
                    if( arming_client.call(arm_cmd) && arm_cmd.response.success)
                    {
                        ROS_INFO("Vehicle armed");
                    }
                    last_request = ros::Time::now();
                }
            }       
                local_pos_pub.publish(pose);
        }

        else
        {
            pose.pose.position.x = iris_local_pose.x + target.curr_x * sigmoid_beta(init_target_x,target.curr_x);//beta/2;
            pose.pose.position.y = iris_local_pose.y + target.curr_y * sigmoid_beta(init_target_y,target.curr_y);//beta/2;
            pose.pose.position.z = descent_flight(10,mlp_gps_.alt+1);
            beta = beta + 0.005;
            local_pos_pub.publish(pose);

            if((target.distance[1] <= 0.25 && iris_local_pose.z <= mlp_gps_.alt+1.1 )|| land_mode)
            {
                land_mode = 1;
                pose.pose.position.x = iris_local_pose.x + target.curr_x * sigmoid_beta(init_target_x,target.curr_x);//beta;
                pose.pose.position.y = iris_local_pose.y + target.curr_y * sigmoid_beta(init_target_y,target.curr_y);//beta;
                pose.pose.position.z = descent_flight(20,mlp_gps_.alt);
                local_pos_pub.publish(pose);

                if(!markerrecogn)
                {
                    mavros_msgs::SetMode land_set_mode;
                    land_set_mode.request.custom_mode = "AUTO.LAND";
                    if( set_mode_client.call(land_set_mode) &&  land_set_mode.response.mode_sent)
                    {
                        ROS_INFO("Land Start");
			            sleep(1);
                    }
                    if(!current_state.armed) 
                    {
                        ROS_INFO("Landing is over. Turn off offb system");
                        return 0;
                    }
                }
            }
        }

        dataWrite(iris_local_pose.x, iris_local_pose.y, iris_local_pose.z,
                  mlp_glopos_.x    , mlp_glopos_.y    , mlp_glopos_.z     );

        ros::spinOnce();
        rate.sleep();
    }

    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}

/*
bool iris_arming()
{
    mavros_msgs::CommandBool arm_cmd;

    if(current_state.armed != true )
    {
        arm_cmd.request.value = true;
     
        if(arming_client.call(arm_cmd) && arm_cmd.response.success)
        {
            ROS_INFO("Vehicle armed");
            drone_state_ = "OFFBOARD"
            return true;
        } 
        else 
        {
            ROS_ERROR("Failed arming or disarming");
            arming_fail_cnt = arming_fail_cnt+1;

            if(arming_fail_cnt == MAXARMINGTRY)
            {
                ROS_ERROR("ARMING IS FAIL. SYSTEM OFF");
                drone_state_ = "OFF";
            }

            return false;
        }
    }
}

void iris_offb_start()
{
    mavros_msgs::SetMode offb_set_mode;

    if(iris_arming())
    {  
        if(current_state.mode != "OFFBOARD")
        {
            offb_set_mode.request.custom_mode = "OFFBOARD";
            set_mode_client.call(offb_set_mode);
        }

            pose.pose.position.x = iris_local_pose.x;
            pose.pose.position.y = iris_local_pose.y;
            pose.pose.position.z = 8;

        if(iris_local_pose.z > DESIRED_ALT) drone_state_ = "GPS_TRACKING";
        else drone_state_= "OFFBOARD"; 
    }

    else 
    {
        drone_state_ = "ARMING";
    }

}

void iris_GPS_tracking()
{
    distance_measure(mlp_gps_.lat, mlp_gps_.lon, iris_gps_.lat, iris_gps_.lon);

    if(!markerrecogn)
    {

        pose.pose.position.x = iris_local_pose.x + gps_distance_.x;
        pose.pose.position.y = iris_local_pose.y + gps_distance_.y;
        pose.pose.position.z = 8;
    }

    else
    {
        if(gps_distance_.meter < 1) drone_state_ = "VISION_TRACKING";
        else drone_state_ = "GPS_TRACKING";
    }

    if(check_trackingtime()) 
    {
        ROS_ERROR("TRACKING TIME IS TOO LONG");
        drone_state_ = "OFF"
    }
}

void iris_vision_tracking()
{
    pose.pose.position.x = iris_local_pose.x + target_y[1] * cos(angle.roll) + vel_x * dt;
    pose.pose.position.y = iris_local_pose.y + target_x[1] * cos(angle.pitch) + vel_y * dt;
    pose.pose.position.z = 5;

    if(check_trackingtime())
    {
        ROS_ERROR("TRACKING TIME IS TOO LONG");
        drone_state_ = "OFF"
    }
}

bool check_trackingtime()
{

}

void 
*/

/*
void flight_mode_change()
{
    switch(drone_state_)
    {
        case : "ARM"
        iris_arming();
        break;
        
        case : "OFFBOARD"
        break;
        
        case : "GPS_TRACKING"
        break;
        
        case : "VISION_TRACKING"
        break;

        case : "LAND"
        break;
        
        case : "OFF"
        break;

        case : "DONE"
        break;
    }

}*/
