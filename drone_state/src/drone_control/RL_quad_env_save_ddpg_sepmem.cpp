#include "offb_RL.h"
#include <random>
#include <chrono>
#include <time.h>

#include <dirent.h>     // 폴더 목록을 읽기 위함
#include <algorithm>    // 정렬을 위함
#include <vector>
#include <sys/stat.h>   // 폴더 확인을 위함



//#define ENV_RAMOS 
//#define GAZEBO_DATA
//#define TRACKING

std::random_device rd;
std::mt19937 gen(rd());

std::uniform_real_distribution<float>  dis(-180,180);//-90~90

auto current = std::chrono::system_clock::now();
auto duration = current.time_since_epoch();
auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
std::random_device prd;
std::mt19937 pgen(millis);
std::random_device vrd;
std::mt19937 vgen(millis);
std::uniform_real_distribution<float> vdis(-0.2,0.2);

int nTestflag;
bool bTestFinished;

offboard::offboard()
: nh_priv("~")
{
  //Init offboard node
  bool offb_init_result = offboard::initializing();
  ROS_ASSERT(offb_init_result);
}

offboard::~offboard()
{
}

RL_env::RL_env()
: RL_priv_nh("~")
{
  bool RL_init_result = RL_env::initializing();
  ROS_ASSERT(RL_init_result);
}

RL_env::~RL_env()
{
}

bool offboard::initializing() // initialzing
{
    drone_state_ = ARM;
    landmode = 0;
    arming_fail_cnt = 0;
    init_target_x = 0;
    init_target_y = 0;
#ifdef TRACKING
    des_iris_alt = 8;
#endif
#ifndef TRACKING
    des_iris_alt = 8;
#endif
    trackingTime = 0;
    write = false;
    sec   = 0;

    mask_set = (~(sp_raw.IGNORE_VX |
                  sp_raw.IGNORE_VY |
                  sp_raw.IGNORE_VZ |
                  sp_raw.IGNORE_YAW_RATE) & 0x0fff);

    pid.prev_error = 0;
    pid.int_error = 0;
    pid.error = 0;

    local_pos_pub =     offb_nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);
    local_vel_pub =     offb_nh.advertise<geometry_msgs::Twist>("/mavros/setpoint_velocity/cmd_vel_unstamped", 10);
    local_sp_pub =     offb_nh.advertise<mavros_msgs::PositionTarget>("mavros/setpoint_raw/local",10);
    local_att_pub =     offb_nh.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 10);
    arming_client =     offb_nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    set_mode_client =   offb_nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    mount_client  =     offb_nh.serviceClient<mavros_msgs::MountConfigure>("/mavros/mount_control/configure");
    joint_property_client = offb_nh.serviceClient<gazebo_msgs::GetJointProperties>("/gazebo/get_joint_properties");
    getlink_client = offb_nh.serviceClient<gazebo_msgs::GetLinkState>("/gazebo/get_link_state");
    setlink_client = offb_nh.serviceClient<gazebo_msgs::SetLinkState>("/gazebo/set_link_state");
    state_sub =         offb_nh.subscribe("/mavros/state", 10, &offboard::state_cb, this);
    model_global_pose =    offb_nh.subscribe("/gazebo/model_states", 10, &offboard::model_globalpose, this);
    MLP_GPS =           offb_nh.subscribe("/platform/MLP/fix",10,&offboard::get_MLP_position, this);
    iris_GPS =          offb_nh.subscribe("/mavros/global_position/raw/fix",10,&offboard::get_iris_position, this);
    iris_local_pos_sub= offb_nh.subscribe("/mavros/local_position/pose",10,&offboard::get_iris_localposition, this);
    marker_pose =       offb_nh.subscribe("/aruco_single/pixel",10,&offboard::markerPixelGet, this);
    iris_imu =          offb_nh.subscribe("/mavros/local_position/pose",10,&offboard::Quarter2Euler, this);
    MarkerRecogn =      offb_nh.subscribe("/aruco_single/markerRecogn",10,&offboard::markrecogn_callback, this);
    iris_vel_sub =      offb_nh.subscribe("/mavros/local_position/velocity_local",10,&offboard::get_iris_vel, this);    

    return true;
}

bool RL_env::initializing()
{
    reset_client  =     RL_nh.serviceClient<gazebo_msgs::SetModelState>("gazebo/set_model_state");
#ifdef ENV_RAMOS
    state_client  =     RL_nh.serviceClient<drl_uav::getEnvinfo_att>("exchange_state_action");
#else
    state_client  =     RL_nh.serviceClient<drl_uav::getEnvinfo>("exchange_state_action");
#endif

    mlp_vel_pub   =     RL_nh.advertise<geometry_msgs::Twist>("/platform/linearVel",10);
    mlp_ang_pub   =     RL_nh.advertise<Mobile_Leveling_Platform_V2::joint_vel>("/platform/AnglePos",10);
    logdata_pub   =     RL_nh.advertise<drl_uav::log_data>("getLogdata",10);
    mark_contact_sub  = RL_nh.subscribe("/platform/mark_contact",10,&RL_env::contact_mark,this);
    upper_contact_sub = RL_nh.subscribe("/platform/upper_contact",10,&RL_env::contact_upper,this);

    return true;
}

////////////////////////Get a data from topic message//////////////////////////////
void offboard::state_cb(const mavros_msgs::State::ConstPtr& msg)
{
    current_state = *msg;
}

void offboard::get_iris_localposition(const geometry_msgs::PoseStampedConstPtr &localP)
{
    iris_local_pose.x = localP->pose.position.x;
    iris_local_pose.y = localP->pose.position.y;
    iris_local_pose.z = localP->pose.position.z;
}

void offboard::get_MLP_position(const sensor_msgs::NavSatFixConstPtr &MLP_GPS )
{
    mlp_gps.lat = MLP_GPS->latitude;
    mlp_gps.lon = MLP_GPS->longitude;
    mlp_gps.alt = MLP_GPS->altitude;
    gpson = true;
}

void offboard::model_globalpose(const gazebo_msgs::ModelStatesConstPtr &global_)
{
    RL_env RL_;
    
    int mlp_idx = -1, uav_idx = -1;

    std::string platform = "Mobile_Leveling_Platform_V2";
    std::string uav = "iris";

    for(int idx=0;; idx++)
    {
        if(platform.compare(global_->name[idx])==0) mlp_idx = idx;
        if(uav.compare(global_->name[idx])==0) uav_idx = idx;
        if(mlp_idx != -1 && uav_idx != -1) break;
    }
    globalpose_.mlp_x = global_->pose[mlp_idx].position.x;
    globalpose_.mlp_y = global_->pose[mlp_idx].position.y;
    globalpose_.mlp_z = global_->pose[mlp_idx].position.z;

    globalvel_.mlp_x = global_->twist[mlp_idx].linear.x;
    globalvel_.mlp_y = global_->twist[mlp_idx].linear.y;    
    globalvel_.mlp_z = global_->twist[mlp_idx].linear.z;
/*
    globalori_.mlp_x = global_->pose[mlp_idx].orientation.x;
    globalori_.mlp_y = global_->pose[mlp_idx].orientation.y;
    globalori_.mlp_z = global_->pose[mlp_idx].orientation.z;
    globalori_.mlp_w = global_->pose[mlp_idx].orientation.w;

    mlp_ori_.mlp_roll  = TransformationQE("roll", globalori_.mlp_x,globalori_.mlp_y,globalori_.mlp_z,globalori_.mlp_w);
    mlp_ori_.mlp_pitch = TransformationQE("pitch",globalori_.mlp_x,globalori_.mlp_y,globalori_.mlp_z,globalori_.mlp_w);
    mlp_ori_.mlp_yaw   = TransformationQE("yaw",  globalori_.mlp_x,globalori_.mlp_y,globalori_.mlp_z,globalori_.mlp_w);
*/
    globalpose_.drone_x = global_->pose[uav_idx].position.x;
    globalpose_.drone_y = global_->pose[uav_idx].position.y;
    globalpose_.drone_z = global_->pose[uav_idx].position.z;

    RL_.real_distance = sqrt((globalpose_.drone_x-globalpose_.mlp_x)*(globalpose_.drone_x-globalpose_.mlp_x)
                            +(globalpose_.drone_y-globalpose_.mlp_y)*(globalpose_.drone_y-globalpose_.mlp_y));
    RL_.mlp_distance = sqrt(globalpose_.mlp_x*globalpose_.mlp_x+globalpose_.mlp_y*globalpose_.mlp_y);

    RL_.real_x = globalpose_.mlp_x - globalpose_.drone_x;
    RL_.real_y = globalpose_.mlp_y - globalpose_.drone_y;
    RL_.real_z = globalpose_.drone_z - globalpose_.mlp_z;
}

void offboard::get_joint_state()
{
    JointState.request.joint_name = gimbal.roll_joint_name;
    joint_property_client.call(JointState);
    gimbal.joint_position[0] = JointState.response.position[0];    

    JointState.request.joint_name = gimbal.pitch_joint_name;
    joint_property_client.call(JointState);
    gimbal.joint_position[1] =JointState.response.position[0];    

    JointState.request.joint_name = gimbal.yaw_joint_name;
    joint_property_client.call(JointState);
    gimbal.joint_position[2] =  JointState.response.position[0];    
}

std::tuple<float, float, float, float, float, float, float> 
offboard::get_link_state(std::string link_name, bool get_orientation, bool get_position, std::string ref_frame)
{
    float ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0;
    float pose_x = 0, pose_y = 0, pose_z = 0;
    LinkState.request.link_name = link_name;
    LinkState.request.reference_frame = ref_frame;
    getlink_client.call(LinkState);
    if(get_orientation)
    {
        ori_x = LinkState.response.link_state.pose.orientation.x;     
        ori_y = LinkState.response.link_state.pose.orientation.y;
        ori_z = LinkState.response.link_state.pose.orientation.z;
        ori_w = LinkState.response.link_state.pose.orientation.w;
    }
    if(get_position)
    {
        pose_x = LinkState.response.link_state.pose.position.x;
        pose_y = LinkState.response.link_state.pose.position.y;
        pose_z = LinkState.response.link_state.pose.position.z;                
    }

    return std::make_tuple(pose_x, pose_y, pose_z, ori_x, ori_y, ori_z, ori_w);
}

void offboard::get_iris_position(const sensor_msgs::NavSatFixConstPtr &iris_GPS)
{
    iris_gps_.lat = iris_GPS->latitude;
    iris_gps_.lon = iris_GPS->longitude;
    iris_gps_.alt = iris_GPS->altitude;
}

void offboard::get_iris_vel(const geometry_msgs::TwistStampedConstPtr &uav_vel)
{
    iris_vel.x = uav_vel->twist.linear.x;
    iris_vel.y = uav_vel->twist.linear.y;
    iris_vel.z = uav_vel->twist.linear.z;
    iris_vel.yaw_speed = uav_vel->twist.angular.z;
}

void offboard::markerPixelGet(const geometry_msgs::PointStampedConstPtr &markerpixel)
{
    RL_env RL_;
    if(marker_id.compare("no marker")!=0)
    {
        mark_buf[0][0] = mark_buf[2][0];
        mark_buf[0][1] = mark_buf[2][1];

        target.distance[0] = target.distance[1];
        target.distance[2] = target.distance[3];
            
//        float pixel_x_raw = markerpixel->point.x-400;
//       float pixel_y_raw = markerpixel->point.y-400;

        mark_buf[1][0] = markerpixel->point.x;
        mark_buf[1][1] = markerpixel->point.y;

        mark_buf[2][0] = LPF(mark_buf[0][0], mark_buf[1][0]); 
        mark_buf[2][1] = LPF(mark_buf[0][1], mark_buf[1][1]);

    //    float correction_x = ((tan(view_angle)-tan(view_angle+angle.roll[0]))*400/(iris_local_pose.z-mlp_gps.alt)/tan(view_angle));
    //    float correction_y = ((tan(view_angle)-tan(view_angle-angle.pitch[0]))*400/(iris_local_pose.z-mlp_gps.alt)/tan(view_angle));

    //    float target_x = mark_buf[2][0]+correction_x;
    //    float target_y = mark_buf[2][1]+correction_y;

        //distance from camera to target / (half pixel size) * tan(angle of view/2) * (goal(pixel) - mark position(pixel))
    //   target.curr_x = ((iris_local_pose.z)/400)*(tan(-angle.roll[0])+tan(view_angle+angle.roll[0])) * (400-mark_buf[2][0]) - 0.8*(iris_local_pose.z)*tan(-angle.roll[0]);	// only LPF
    //   target.curr_y = ((iris_local_pose.z)/400)*(tan(angle.pitch[0])+tan(view_angle-angle.pitch[0])) * (400-mark_buf[2][1]) - 0.8*(iris_local_pose.z)*tan(angle.pitch[0]);
        float translation[2] = { 0, 0};
        if(marker_id.compare("center")==0)         {translation[0] =  0; translation[1] =  0;}
        else if(marker_id.compare("leftup")==0)    {translation[0] = -1; translation[1] =  1;}
        else if(marker_id.compare("rightup")==0)   {translation[0] = -1; translation[1] = -1;}
        else if(marker_id.compare("leftdown")==0)  {translation[0] =  1; translation[1] =  1;}
        else if(marker_id.compare("rightdown")==0) {translation[0] =  1; translation[1] = -1;}

        const float offset = 0.625;
        float offset_x = offset*translation[0];
        float offset_y = offset*translation[1];
//        HMGtransform(target.curr_x,target.curr_y,0,1,offset_x,offset_y);
        rotatetransform(offset_x,offset_y,-1*DEG2RAD*(angle.yaw));
        target.curr_x = ((iris_local_pose.z-mlp_gps.alt)/400)*tan(view_angle) * (400-mark_buf[2][0]) + offset_x;	// only LPF
        target.curr_y = ((iris_local_pose.z-mlp_gps.alt)/400)*tan(view_angle) * (400-mark_buf[2][1]) + offset_y;

    //    HMGtransform(target.curr_x,target.curr_y,0,1,offset_x,offset_y);
    //     std::cout<<marker_id<<std::endl;
    //    ROS_INFO("x : %f, y : %f",target.curr_x, target.curr_y);           
    //    target.curr_x = ((iris_local_pose.z-mlp_gps.alt)/400)*tan(view_angle) * (400-mark_buf[1][0]) + offset_x; // no LPF
    //    target.curr_y = ((iris_local_pose.z-mlp_gps.alt)/400)*tan(view_angle) * (400-mark_buf[1][1]) + offset_y;

    //    target.curr_x = ((iris_local_pose.z-mlp_gps.alt)/400)*tan(view_angle) * (400-saturation(0,800,target_x)); // correction and LPF
    //    target.curr_y = ((iris_local_pose.z-mlp_gps.alt)/400)*tan(view_angle) * (400-saturation(0,800,target_y));

        target.dif_x = (target.pre_x - target.curr_x)/dt;
        target.dif_y = (target.pre_x - target.curr_x)/dt;

        target.pre_x = target.curr_x;
        target.pre_y = target.curr_y;

        XYtransform(target.curr_x,target.curr_y);
        slewrate(target.curr_x, target.curr_x, target.vel_x, target.vel_y);

        target.distance[1] = abs(sqrt(target.curr_x*target.curr_x + target.curr_y*target.curr_y));
        target.distance[3] = (target.distance[1] - target.distance[0])/dt;
    }
}

void offboard::Quarter2Euler(const geometry_msgs::PoseStampedConstPtr &iris_angle)
{
    float x = iris_angle->pose.orientation.x;
    float y = iris_angle->pose.orientation.y;
    float z = iris_angle->pose.orientation.z;
    float w = iris_angle->pose.orientation.w;

    angle.roll[1] = angle.roll[0];
    angle.pitch[1] = angle.pitch[0];

    // roll (x-axis rotation)
    float sinr_cosp = 2 * (w * x + y * z);
    float cosr_cosp = 1 - 2 * (w * x + y * y);
    angle.roll[0] = std::atan2(sinr_cosp, cosr_cosp)*RAD2DEG;

    // pitch (y-axis rotation)
    float sinp = 2 * (w * y - z * x);
    if (std::abs(sinp) >= 1)
        angle.pitch[0] = std::copysign(M_PI / 2, sinp)*RAD2DEG; // use 90 degrees if out of range
    else
        angle.pitch[0] = std::asin(sinp)*RAD2DEG;

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (w * z + x * y);
    double cosy_cosp = 1 - 2 * (y * y + z * z);
    angle.yaw = std::atan2(siny_cosp, cosy_cosp)*RAD2DEG;

    angle.roll[2] = (angle.roll[0]-angle.roll[1])/dt;
    angle.pitch[2] = (angle.pitch[0]-angle.roll[1])/dt;
}

float offboard::TransformationQE(std::string euler, float x, float y, float z, float w)
{
    float roll, pitch, yaw;
    if(euler.compare("roll")==0)
    {
        float sinr_cosp = 2 * (w * x + y * z);
        float cosr_cosp = 1 - 2 * (w * x + y * y);
        roll = std::atan2(sinr_cosp, cosr_cosp)*RAD2DEG;
        return roll;
    }
    // pitch (y-axis rotation)
    else if(euler.compare("pitch")==0)
    {
        float sinp = 2 * (w * y - z * x);
        if (std::abs(sinp) >= 1)
            pitch = std::copysign(M_PI / 2, sinp)*RAD2DEG; // use 90 degrees if out of range
        else
            pitch = std::asin(sinp)*RAD2DEG;
        return pitch;    
    }
    // yaw (z-axis rotation)
    else if(euler.compare("yaw")==0)
    {
        double siny_cosp = 2 * (w * z + x * y);
        double cosy_cosp = 1 - 2 * (y * y + z * z);
        yaw = std::atan2(siny_cosp, cosy_cosp)*RAD2DEG;
        return yaw;
    }
}

float offboard::Quater_Rotation(float roll, float pitch, float yaw)
{
    angle.w = cos(roll/2)*cos(pitch/2)*cos(yaw/2)+sin(roll/2)*sin(pitch/2)*sin(yaw/2);
    angle.x = sin(roll/2)*cos(pitch/2)*cos(yaw/2)-cos(roll/2)*sin(pitch/2)*sin(yaw/2);
    angle.y = cos(roll/2)*sin(pitch/2)*cos(yaw/2)+sin(roll/2)*cos(pitch/2)*sin(yaw/2);
    angle.z = cos(roll/2)*cos(pitch/2)*sin(yaw/2)+sin(roll/2)*sin(pitch/2)*cos(yaw/2);
}

void offboard::markrecogn_callback(const aruco_msgs::MarkerRecogn::ConstPtr& markRecogn_)
{
   markerrecogn = markRecogn_->MarkRecogn;
   marker_id = markRecogn_->Marker_id;
}

void RL_env::contact_mark(const gazebo_msgs::ContactsStateConstPtr &mark_contact)
{
    if(!mark_contact->states.empty())
    {
        if(mark_contact->states[0].total_wrench.force.z>0) contact_recogn_mark = true;
        else contact_recogn_mark = false;
    }
}
void RL_env::contact_upper(const gazebo_msgs::ContactsStateConstPtr &upper_contact)
{
    if(!upper_contact->states.empty())
    {
        if(upper_contact->states[0].total_wrench.force.z>0) contact_recogn_upper = true;
        else contact_recogn_upper = false;
    }
}
//////////////////////////////////filter/////////////////////////////////////
float offboard::LPF(float data_pre, float data_now)
{
    float lpf_output = WEIGHT*data_pre + (1-WEIGHT)*data_now;
    return lpf_output;
}

////////////////////////////constrain function//////////////////////////////
float offboard::saturation(float min, float max, float input)
{
    float output = 0;
    if(input < min) return output = min;
    else if(input < max) return output = input;
    else return output = max;
}

///////////////////////////////////distance_measurement(GPS)//////////////////////////////////////////////
float offboard::distance_measure(float platform_lat, float platform_lon, float drone_lat, float drone_lon)
{
    float avglat = (mlp_gps.lat+iris_gps_.lat)/2;
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

    gps_distance.meter = distance * 1000;
    gps_distance.x = lat_distance * 1000;
    gps_distance.y = lon_distance * 1000;

    XYtransform(gps_distance.x,gps_distance.y);

    return gps_distance.x, gps_distance.y, gps_distance.meter;
}
float offboard::sigmoid_beta(float init_target_distance, float error)
{
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float init = 1/abs(init_target_distance)*0.5;
          init = saturation(0,0.5,init);
          init = 0;
    float sigbeta;
    if(abs(error) <= 0.5*init_target_distance) test = true;
    else test = true;
    if(test)
    {   
        sec += 0.001;
        sigbeta = (BETA_MAX-init)/(1+pow(EXP,GRADIENT*(sec-SIGMOID_TH)))+init;
    }
    else
    {   
        sigbeta = 1;
    }
    ROS_INFO("sigbeta : %f",sigbeta);
    return sigbeta;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
}
//////////////////////////////////smooth speed/////////////////////////////////////
void offboard::slewrate(float &sp_x, float &sp_y, float &sp_vel_x, float &sp_vel_y)
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
		_sp_pev_prev(0) = _sp_pev(0) - iris_vel.x * dt;
		_sp_pev_prev(1) = _sp_pev(1) - iris_vel.y * dt;
	}

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
float offboard::PID_controller(float p_gain, float i_gain, float d_gain, float observe, float goal)
{
    pid.error = goal - observe;
    pid.int_error += pid.error*dt;
    float fliter_error = LPF(pid.prev_error,pid.error);
    float dev_error = (fliter_error - pid.prev_error)/dt;
    pid.prev_error = fliter_error;
    float control_output = p_gain * pid.error + pid.int_error * i_gain + dev_error * d_gain;
    return control_output;
}
///////////////////descent flight function///////////////////
float offboard::descent_flight(float vel, float goal)
{
    float descent_rate = vel*dt;
    float desire_height = iris_local_pose.z - descent_rate;

    if(desire_height<goal)  
    {
        desire_height = goal;
    }

    return desire_height;
}

//////////////////////transformation//////////////////////////////
float offboard::XYtransform(float &x_frame, float &y_frame)
{
    float curr_xframe = x_frame; 
    float curr_yframe = y_frame;  
    return y_frame = curr_xframe, x_frame = curr_yframe;
}

float offboard::rotatetransform(float &x_frame, float &y_frame, float angle)
{
    float curr_xframe =  cosf(angle)* x_frame + sinf(angle)* y_frame;
    float curr_yframe = -sinf(angle)* x_frame + cosf(angle)* y_frame;
    return x_frame = curr_xframe, y_frame = curr_yframe;
}

float offboard::HMGtransform(float &x_frame, float &y_frame, float angle, float z, float x_offset, float y_offset)
{
    float curr_xframe =  cosf(angle)* x_frame + sinf(angle)* y_frame + z * x_offset;
    float curr_yframe = -sinf(angle)* x_frame + cosf(angle)* y_frame + z * y_offset; 
    return x_frame = curr_xframe, y_frame = curr_yframe;
}
void RL_env::publish_logdata(float uav_x, float uav_y, float uav_z,
                             float target_x, float target_y, float target_z, 
                             int episode, int restart, int landingstate, int step)
{
    sendlogdata.uav_x = uav_x;
    sendlogdata.uav_y = uav_y;
    sendlogdata.uav_z = uav_z;
    sendlogdata.target_x = target_x;
    sendlogdata.target_y = target_y;
    sendlogdata.target_z = target_z;
    sendlogdata.episode = episode;
    sendlogdata.restart = restart;
    sendlogdata.landingstate = landingstate;
    sendlogdata.step = step;
    logdata_pub.publish(sendlogdata);
}
////////////save a logdata(position of LMP and drone)/////////////// 
std::string getLatestSessionDir() {
    std::string base_path = "/home/baek/ddpg_result/ddpg_sepmem";
    DIR *dir;
    struct dirent *ent;
    std::vector<std::string> folders;

    if ((dir = opendir(base_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            // 폴더(디렉토리)인 것만 골라냄 ( . 과 .. 제외)
            if (ent->d_type == DT_DIR) {
                std::string name = ent->d_name;
                if (name != "." && name != "..") {
                    folders.push_back(name);
                }
            }
        }
        closedir(dir);
    }

    if (folders.empty()) return "";

    // 날짜_시간 형식(YYYYMMDD_HHMMSS)이므로 정렬 시 가장 마지막 것이 최신
    std::sort(folders.begin(), folders.end());
    return base_path + "/" + folders.back();
}

void offboard::dataWrite(float pose_x, float pose_y, float pose_z,
                         float mlp_x, float mlp_y, float mlp_z, 
                         float arg_step, float arg_restart, int episode, int end_step)
{
    static int last_saved_episode = -1;
    // 리셋 직후 step이 0이 되는 경우를 대비해 에피소드 중 최대 스텝을 기억합니다.
    static int max_step_in_episode = 0; 
    std::ofstream fout;

    int current_step = (int)arg_step;
    bool is_restart = (arg_restart > 0.5f);

    // [핵심 수정] restart 여부와 관계없이 현재 데이터를 버퍼에 저장합니다.
    // 이렇게 해야 34스텝째의 마지막 위치 데이터가 버퍼에 기록됩니다.
    if(current_step < 10000) {
        drone_traj[0][current_step] = pose_x; 
        drone_traj[1][current_step] = pose_y; 
        drone_traj[2][current_step] = pose_z;
        mlp_traj[0][current_step] = mlp_x; 
        mlp_traj[1][current_step] = mlp_y; 
        mlp_traj[2][current_step] = mlp_z;

        if (current_step > max_step_in_episode) {
            max_step_in_episode = current_step;
        }
    }

    std::string session_path = getLatestSessionDir();
    if (session_path == "") return;

    // 폴더 생성 로직
    std::string traj_dir = session_path + "/trajectory";
    std::string err_dir = session_path + "/error";
    mkdir(traj_dir.c_str(), 0777);
    mkdir(err_dir.c_str(), 0777);

    // 에피소드 종료 시점 저장 로직
    if(is_restart && (episode != last_saved_episode)) {
        
        // 최종 저장 스텝 결정 (end_step이 0인 경우 기억해둔 max_step 사용)
        int final_step = (end_step <= 0) ? max_step_in_episode : end_step;
        std::string ep_suffix = "_ep" + std::to_string(episode) + ".csv";

        // [A] Trajectory 저장 (trajectory 폴더)
        fout.open(traj_dir + "/trajectory" + ep_suffix); 
        if(fout.is_open()) {
            fout << "Step,Drone_X,Drone_Y,Drone_Z,MLP_X,MLP_Y,MLP_Z\n";
            for(int i = 0; i <= final_step; i++) { // i <= final_step 까지 저장
                fout << i << "," 
                     << drone_traj[0][i] << "," << drone_traj[1][i] << "," << drone_traj[2][i] << ","
                     << mlp_traj[0][i] << "," << mlp_traj[1][i] << "," << mlp_traj[2][i] + 1.1 << "\n";
            }
            fout.close();
        }

        // [B] Error 저장 (error 폴더)
        fout.open(err_dir + "/error" + ep_suffix);
        if(fout.is_open()) {
            fout << "Step,Error_X,Error_Y,Error_Z\n";
            for(int i = 0; i <= final_step; i++) {
                fout << i << ","
                     << mlp_traj[0][i] - drone_traj[0][i] << ","
                     << mlp_traj[1][i] - drone_traj[1][i] << ","
                     << mlp_traj[2][i] - drone_traj[2][i] + 1.1 << "\n";
            }
            fout.close();
        }

        // [C] Landing Summary 저장 (누적)
        std::string summary_file = session_path + "/landing_summary.csv";
        bool exists = (access(summary_file.c_str(), F_OK) != -1);
            fout.open(summary_file, std::ios_base::app);
        if(fout.is_open()){
            if(!exists) fout << "Episode,Success\n";
            // sqrt(rel_x^2 + rel_y^2) <= 0.75 ==> rel_x <= 0.75 & rel_y <= 0.75 
            bool is_success = (abs(mlp_x - pose_x) <= 0.75) && (abs(mlp_y - pose_y) <= 0.75) && (pose_z -mlp_z < 1);
            fout << episode << "," << (is_success ? 1 : 0) << "\n";
            fout.close();
        }

        last_saved_episode = episode;
        max_step_in_episode = 0; // 다음 에피소드를 위해 초기화
        ROS_INFO("Saved Episode %d up to final step %d", episode, final_step);
    }
}
////////////////////////////////Reset Model//////////////////////////////
void RL_env::SetModelReset(bool trans, std::string model_name, float pos_x, 
             float pos_y, float pos_z, float vx=0, float vy=0, float vz=0)
{
    offboard offb;
//    std::uniform_real_distribution<float>  gdis(0,pos_y);
    if(trans)
    {
//        rd_pos = gdis((pgen));
        if(nTestflag) 
        {   if(angle >= 360)
            {
                angle  =   0;
                bTestFinished = true;
                return;
            }
            else if((env_info.episode >= 1) && (env_info.episode % TEST_ITERATION_CNT == 0)) 
            {angle += DEG2RAD * 1;}

        }
        else
        {
            angle = DEG2RAD*(dis(gen));
        }
        
        offb.rotatetransform(pos_x,pos_y,angle);//pos_y
        if(model_name == "iris")
        {
            save_iris_x = pos_x;
            save_iris_y = pos_y;//pos_y
        }
    }

    reset_pose.model_name = model_name;
    reset_pose.pose.position.x = pos_x;
    reset_pose.pose.position.y = pos_y;
    reset_pose.pose.position.z = pos_z;
    reset_pose.pose.orientation. x = 0;
    reset_pose.pose.orientation. y = 0;
    reset_pose.pose.orientation. z = 0;
    reset_pose.pose.orientation. w = 1;
    reset_pose.twist.linear.x =  vx;
    reset_pose.twist.linear.y =  vy;
    reset_pose.twist.linear.z =  vz;
    reset_pose.twist.angular.x = 0;
    reset_pose.twist.angular.y = 0; 
    reset_pose.twist.angular.z = 0;
    
    reset.request.model_state = reset_pose;
}

void RL_env::SetModelvelReset(float vx, float vy, float vz)
{
    offboard offb;
    offb.rotatetransform(vx,vy,angle);
    float noise[2];
    noise[0] = vdis(vgen);
    vx = vx + noise[0];
    noise[1] = vdis(vgen);
    vy = vy + noise[1];
    offb.pub_vel(vx,vy,vz);

    env_info.action.vel[0] = vx;
    env_info.action.vel[1] = vy;
    env_info.action.vel[2] = 0;
}
void RL_env::get_uavstate()
{
    offboard offb;
#ifdef GAZEBO_DATA
    env_info.state.rel_pose_dot[0] = real_x - env_info.state.rel_pose[0];
    env_info.state.rel_pose_dot[1] = real_y - env_info.state.rel_pose[1];
    ROS_INFO("xdot : %f, ydot : %f",env_info.state.rel_pose_dot[0],env_info.state.rel_pose_dot[1]);
    env_info.state.rel_pose[0] = real_x;
    env_info.state.rel_pose[1] = real_y;
//    env_info.state.rel_pose[2] = real_z-offb.mlp_gps.alt;//offb.iris_local_pose.z - offb.mlp_gps.alt;
    ROS_INFO("z : %f",env_info.state.rel_pose[2]);
#endif
#ifndef TRACKING
    env_info.state.rel_pose_dot[0] = offb.target.curr_x - env_info.state.rel_pose[0];//offb.globalvel_.mlp_x - offb.iris_vel.x;//
    env_info.state.rel_pose_dot[1] = offb.target.curr_y - env_info.state.rel_pose[1];//offb.globalvel_.mlp_y - offb.iris_vel.y;//
    env_info.state.rel_pose_dot[2] = (offb.iris_local_pose.z-offb.mlp_gps.alt) - env_info.state.rel_pose[2];//offb.globalvel_.mlp_z - offb.iris_vel.z;//
    env_info.state.rel_pose[0] = offb.target.curr_x;
    env_info.state.rel_pose[1] = offb.target.curr_y;
    env_info.state.rel_pose[2] = offb.iris_local_pose.z-offb.mlp_gps.alt;
    env_info.state.yaw_speed   = offb.iris_vel.yaw_speed;
    ROS_INFO("x : %f, y : %f, z : %f",env_info.state.rel_pose[0]*4/3/(8*tan(80*3.141592/2/180)),env_info.state.rel_pose[1]*4/3/(8*tan(80*3.141592/2/180)),2*(env_info.state.rel_pose[2]-3.375)/6.5);
#endif
#ifdef TRACKING
    env_info.state.rel_pose_dot[0] = real_x - env_info.state.rel_pose[0];
    env_info.state.rel_pose_dot[1] = real_y - env_info.state.rel_pose[1];
    env_info.state.rel_pose[0] = real_x;
    env_info.state.rel_pose[1] = real_y;
#endif

#ifdef ENV_RAMOS

#else
    env_info.state.vel[0]      = offb.iris_vel.x;
    env_info.state.vel[1]      = offb.iris_vel.y;
    env_info.state.vel[2]      = offb.iris_vel.z;
#endif    
    env_info.state.yaw         = offb.angle.yaw*DEG2RAD;
    env_info.markrecogn        = offb.markerrecogn;
}

void RL_env::exchange_stateNaction()
{
    RLenvinfo.request.rel_x      = env_info.state.rel_pose[0];
    RLenvinfo.request.rel_y      = env_info.state.rel_pose[1];
    RLenvinfo.request.rel_z      = env_info.state.rel_pose[2];

#ifdef ENV_RAMOS
    RLenvinfo.request.rel_pose_xdot = env_info.state.rel_pose_dot[0];
    RLenvinfo.request.rel_pose_ydot = env_info.state.rel_pose_dot[1];
#else
    RLenvinfo.request.rel_pose_xdot = env_info.state.rel_pose_dot[0];
    RLenvinfo.request.rel_pose_ydot = env_info.state.rel_pose_dot[1];
    RLenvinfo.request.rel_pose_zdot = env_info.state.rel_pose_dot[2];
#endif

    RLenvinfo.request.reward     = env_info.reward;
    RLenvinfo.request.episode    = env_info.episode;
    RLenvinfo.request.step       = env_info.step;
    RLenvinfo.request.yaw        = env_info.state.yaw;
    RLenvinfo.request.yaw_speed  = env_info.state.yaw_speed;
    RLenvinfo.request.done       = env_info.restart;
    RLenvinfo.request.reset_error      = env_info.reset_error;
    RLenvinfo.request.mark_recogn = env_info.markrecogn;
    ROS_INFO("reward : %f", RLenvinfo.request.reward);
    while(!state_client.call(RLenvinfo))
    {ROS_ERROR("Error : could not get action");}

#ifdef ENV_RAMOS
    env_info.action.roll         = RLenvinfo.response.roll;
    env_info.action.pitch        = RLenvinfo.response.pitch;
    env_info.action.yaw_speed    = RLenvinfo.response.yaw_speed;
    env_info.action.thrust       = (RLenvinfo.response.thrust+1)/2;
#else
    #ifndef TRACKING
    env_info.action.vel[0]       = RLenvinfo.response.uav_vx;
    env_info.action.vel[1]       = RLenvinfo.response.uav_vy;
    env_info.action.vel[2]       = (RLenvinfo.response.uav_vz-1)/2;
    env_info.action.yaw_speed    = RLenvinfo.response.uav_yaw_sp;
    #else
    env_info.action.vel[0]       = RLenvinfo.response.uav_vx;
    env_info.action.vel[1]       = RLenvinfo.response.uav_vy;
    #endif
#endif

}

void RL_env::MLPvelPub(float setpointV, float setpointAng=0)
{
    mlp_pose_msg.linear.x = setpointV;
    mlp_angle_msg.data = setpointAng*DEG2RAD;

    mlp_vel_pub.publish(mlp_pose_msg);
    mlp_ang_pub.publish(mlp_angle_msg);
}

void RL_env::MLPcontrol()
{
    switch(mlp_state_)
    {
        case MLPSTART :
//        MLPvelPub(0);
        MLPvelPub(MLPVELOCITY);
        break;

        case MLPSTOP :
        MLPvelPub(0);
        break;
    }
}
float RL_env::calcul_reward(offboard offb, int step, bool negative=true)
{
    float reward = 0;
    float reset_reward = 0;
    ROS_INFO("marker %d",offb.markerrecogn);
    if(negative) reset_reward = -10000;
    else reset_reward = 0;
    //고도유지
    //if(abs((offb.iris_local_pose.z-offb.mlp_gps.alt)-10) < 2) {reward = 1;}
    //else {reward=0;}
    /*
    if(mlp_distance >= 45 ) 
    {
     reward = reset_reward;
     env_info.restart = true;
     ROS_WARN("mlp distance reset : %f, reward : %f",mlp_distance,env_info.reward);
     return reward;
    }*/
#ifndef TRACKING
    if(offb.drone_state_ == LAND)
    {
     int score_markContact  = contact_recogn_mark ? 1:0; 
     int score_upperContact = contact_recogn_upper ? 1:0;
     if(!score_markContact && !score_upperContact) reward = real_distance * -100 * landing_start_alt; 
     else reward = -200 * (2*score_upperContact-score_markContact)*landing_start_alt;
     reward = reward > 0 ? 0 : reward;
     ROS_WARN("Landing command : %f, Landing score : %f",env_info.action.landing_cmd,reward);
     env_info.restart = true;
     return reward;
    }

    if(abs(real_x) <= 0.75 && abs(real_y) <= 0.75 && env_info.state.rel_pose[2] <= 1)
    {
            ROS_WARN("contact upper : %d, contact mark : %d", contact_recogn_mark ,contact_recogn_upper);
            reward = 10000;
            ROS_WARN("reward : %f",reward);
            env_info.restart = true;
            offb.write = true;
            ROS_WARN("write : %d",offb.write);    
                
            return reward;
    }
#endif
    if(!offb.markerrecogn)
    {
#ifndef GAZEBO_DATA

/*    #ifndef TRACKING
        if(abs(real_x) >= 0.75 || abs(real_y) >= 0.75 || env_info.state.rel_pose[2] >= 0.75)
        {
            //reward = -200*env_info.state.rel_pose[2] -500*real_distance;//250 200
            reward = reset_reward;
            env_info.restart = true;
            ROS_INFO("real x distance : %f, real y distance : %f",abs(real_x),abs(real_y));
            ROS_INFO("iris altitude : %f",offb.iris_local_pose.z);
            ROS_WARN("mark reset, reward : %f",reward);
            return reward;
        }
        else //if//(abs(real_x) <= 0.75 && abs(real_y) <= 0.75 && env_info.state.rel_pose[2] <= 1)//(contact_recogn_mark || contact_recogn_upper)
        {
            ROS_WARN("contact upper : %d, contact mark : %d", contact_recogn_mark ,contact_recogn_upper);
            reward = 400000;
            env_info.restart = true;
            offb.write = true;
            ROS_WARN("write : %d",offb.write);        
            return reward;
        }
    #else
            reward = reset_reward;
            env_info.restart = true;
            ROS_WARN("mark reset, reward : %f",reward);
            return reward;
    #endif
        ROS_ERROR("ERROR");
        env_info.restart = true;
        return reward;     */
        reward = reset_reward;
        env_info.restart = true;
        ROS_WARN("real x distance : %f, real y distance : %f",abs(real_x),abs(real_y));
        ROS_WARN("iris altitude : %f",env_info.state.rel_pose[2]);
        ROS_WARN("mark reset, reward : %f",reward);
        return reward;
#endif
    }
#ifdef TRACKING
    else if(real_distance > 10) 
    {
     reward = reset_reward;
     env_info.restart = true;
     ROS_WARN("distance reset : %f, reward : %f",real_distance,env_info.reward);
     return reward;
    }

    else if(offb.iris_local_pose.z > 15)
    {
     reward = reset_reward; 
     env_info.restart = true;
     ROS_WARN("altitude reset : %f, reward : %f",offb.iris_local_pose.z,env_info.reward);
     return reward;
    }

    else if((abs(offb.angle.roll[0])>=170)||(abs(offb.angle.pitch[0])>=170))
    {
     reward = reset_reward;
     env_info.restart = true;
     ROS_WARN("angle reset, roll : %f, pitch : %f, reward : %f",offb.angle.roll[0],offb.angle.pitch[0],env_info.reward);
     return reward;
    }
#endif
    else if(env_info.step > MAXSTEP) 
    {
     reward = reset_reward;
     ROS_WARN("step reward : %f",reward);
     env_info.restart = true;
     return reward;

    }

    #ifdef ENV_RAMOS
    float shape = 100*(sqrt(pow(env_info.state.rel_pose[0],2)+pow(env_info.state.rel_pose[1],2)+pow(env_info.state.rel_pose[2],2)+pow(env_info.state.yaw,2)))        
                  +10*(sqrt(pow(env_info.state.rel_pose_dot[0],2)+pow(env_info.state.rel_pose_dot[1],2)))
                   +1*(sqrt(pow(env_info.action.roll,2)+pow(env_info.action.pitch,2)+pow(env_info.action.yaw_speed,2)+pow(env_info.action.thrust,2)));
    #else
        #ifdef TRACKING
        float shape = -100*sqrt(pow(real_distance,2)+pow(env_info.state.rel_pose[2],2));
        #else
        float shape = -400*real_distance-150*sqrt(pow(env_info.state.rel_pose[2],2))//-500*pow(env_info.state.yaw,2) //0.4 10 400, 150//0.4 10 400 250
                      -100*sqrt(pow(env_info.state.rel_pose_dot[0],2)+pow(env_info.state.rel_pose_dot[1],2)+pow(env_info.state.rel_pose_dot[2],2))
                      -50*sqrt(pow(env_info.action.vel[0],2)+pow(env_info.action.vel[1],2)+pow(env_info.action.vel[2],2));//+pow(env_info.state.yaw,2));//-100*sqrt(pow(real_distance,2)+pow(env_info.state.rel_pose[2]-1,2))
//                      -50*sqrt(pow(env_info.state.rel_pose_dot[0],2)+pow(env_info.state.rel_pose[1],2));
//  21.11.07 100 100 10
        #endif
    #endif
    if(step == 1) reward = -1000;
    else     reward = negative ? shape-pre_shape-1000 : shape-pre_shape+1000;
    reward = shape;
//    if((negative && reward > 0)||(!negative && reward < 0)) reward = 0;

    ROS_INFO("shaping reward : %f, pre shaping : %f",shape, pre_shape);
    pre_shape = shape;
    env_info.restart = false;
    return reward;
}

void offboard::pub_pose(float targetX, float targetY, float targetZ,
                        float ori_x = 0, float ori_y = 0, float ori_z = 0, float ori_w = 1)
{  
    pose.pose.position.x = targetX;
    pose.pose.position.y = targetY;
    pose.pose.position.z = targetZ;
    pose.pose.orientation.x = ori_x;
    pose.pose.orientation.y = ori_y;
    pose.pose.orientation.z = ori_z;
    pose.pose.orientation.w = ori_w;

    local_pos_pub.publish(pose);
}

void offboard::pub_vel(float targetVX, float targetVY, float targetVZ)
{
    vel.linear.x = targetVX;
    vel.linear.y = targetVY;
    vel.linear.z = targetVZ;
    local_vel_pub.publish(vel);
}

void offboard::pub_sp_raw(float targetVX, float targetVY, float targetVZ, float yaw_speed)
{
    sp_raw.header.stamp = ros::Time::now();
    sp_raw.coordinate_frame = 1;
    sp_raw.type_mask = mask_set;
    sp_raw.velocity.x = targetVX;
    sp_raw.velocity.y = targetVY;
    sp_raw.velocity.z = targetVZ;
    sp_raw.yaw_rate = yaw_speed;

    local_sp_pub.publish(sp_raw);
}
void offboard::iris_att_pub(float roll, float pitch, float yaw_speed, float thrust)
{   
    struct q_{float x,y,z,w;}q;
    float cp = cos(pitch*0.5);
    float sp = sin(pitch*0.5);
    float cr = cos(roll*0.5);
    float sr = sin(roll*0.5);
    float cy = cos(0);
    float sy = sin(0);

    q.x = cr * cp * sy - sr * sp * cy; 
    q.y = sr * cp * cy - cr * sp * sy; 
    q.z = cr * sp * cy + sr * cp * sy; 
    q.w = cr * cp * cy + sr * sp * sy; 

	att_.orientation.x = q.x;
	att_.orientation.y = q.y;
	att_.orientation.z = q.z;
	att_.orientation.w = q.w;
    att_.body_rate.z   = yaw_speed;
    att_.thrust        = thrust;
    local_att_pub.publish(att_);
}

void offboard::gimbal_set()
{
    mount_configure.request.mode = 0;
    mount_configure.request.stabilize_roll = 0;//true;
    mount_configure.request.stabilize_pitch = 0;//true;
    mount_configure.request.stabilize_yaw  = true;//true;
    mount_configure.request.roll_input = 0;
    mount_configure.request.pitch_input = 0;
    mount_configure.request.yaw_input = 0;

    mount_client.call(mount_configure);
}

void offboard::set_link_orientation(std::string name, float roll, float pitch, float yaw)
{
    Quater_Rotation(DEG2RAD*roll,DEG2RAD*pitch,DEG2RAD*(yaw-angle.yaw));
    ROS_INFO("roll : %f, pitch : %f, yaw : %f", roll, pitch, angle.yaw-yaw);
    ROS_INFO("x : %f, y : %f, z : %f, w : %f",angle.x, angle.y, angle.z, angle.w);
    setlinkstate.request.link_state.link_name = name;
    setlinkstate.request.link_state.pose.orientation.x = angle.x; 
    setlinkstate.request.link_state.pose.orientation.y = angle.y;
    setlinkstate.request.link_state.pose.orientation.z = angle.z;
    setlinkstate.request.link_state.pose.orientation.w = angle.w;
    setlinkstate.request.link_state.reference_frame = "iris::base_link";
    if(!setlink_client.call(setlinkstate))
    {
        ROS_WARN("set link state fail");
    }
}
/////////////////////gps and vision tracking//////////////////////////
void offboard::iris_initializing()
{
    RL_env RL_;
    sec      = 0;
    landmode = 0;
    arming_fail_cnt = 0;
    init_target_x = 0;
    init_target_y = 0;
    trackingTime = 0;
    RL_.mlp_state_ = MLPSTART;

    pid.prev_error = 0;
    pid.int_error = 0;
    pid.error = 0;

    RL_.contact_recogn_mark = false;
    RL_.contact_recogn_upper= false;
    RL_.env_info.state.rel_pose[0] = 0;
    RL_.env_info.state.rel_pose[1] = 0;
    RL_.env_info.state.rel_pose[2] = 0;    
    RL_.env_info.state.rel_pose_dot[0] = 0;
    RL_.env_info.state.rel_pose_dot[1] = 0;
    RL_.env_info.state.rel_pose_dot[2] = 0;
    RL_.env_info.reward = 0;
    RL_.env_info.restart = false;
    RL_.env_info.reset_error = false;
#ifdef ENV_RAMOS
    RL_.env_info.action.thrust = 0;
#endif
    RL_.get_uavstate();
    drone_state_ = RL;
}

bool offboard::iris_arming()
{
    if(current_state.armed != true )
    {
        arm_cmd.request.value = true;
     
        if(arming_client.call(arm_cmd) && arm_cmd.response.success)
        {
            ROS_INFO("Vehicle armed");
            drone_state_ = OFFBOARD;
            return true;
        } 
        else 
        {
            ROS_ERROR("Failed arming or disarming");
            arming_fail_cnt = arming_fail_cnt+1;

            if(arming_fail_cnt == MAXARMINGTRY)
            {
                ROS_ERROR("ARMING IS FAIL. SYSTEM OFF");
                drone_state_ = OFF;
            }

            return false;
        }
    }
}


void offboard::iris_offb_start()
{
    RL_env RL_;

    if(iris_arming())
    {  
        if(current_state.mode != "OFFBOARD")
        {
            offb_set_mode.request.custom_mode = "OFFBOARD";
            set_mode_client.call(offb_set_mode);
        }

        if(iris_local_pose.z > des_iris_alt-0.1) 
        {
//            ROS_INFO("error_x : %f, error_y : %f",RL_.real_x-target.curr_x,RL_.real_y-target.curr_y);
            ROS_INFO("GPS Tracking Start");
            RL_.mlp_state_ = MLPSTART;
            drone_state_ = GPS_TRACKING;

        }
        else 
        {
            ROS_INFO("Current altitude : %f", iris_local_pose.z);
            drone_state_= OFFBOARD; 
        }
    }

    else 
    {
        drone_state_ = ARM;
    }
}

void offboard::iris_GPS_tracking()
{
    RL_env RL_;

    if(markerrecogn)
    //if(iris_local_pose.z - des_iris_alt < 0.1)
    {
        init_target_x = iris_local_pose.x + target.curr_x;
        init_target_y = iris_local_pose.y + target.curr_y;
        init_rel_pose_x = -1*RL_.real_x;
        init_rel_pose_y = -1*RL_.real_y;
        init_velocity_x = iris_vel.x;
        init_velocity_y = iris_vel.y;
        RL_.env_info.reward = 0;
        RL_.exchange_stateNaction();
        drone_state_  = RL;
        ROS_INFO("Vision Tracking Start, target distance is : %f, X : %f, Y : %f", target.distance[1], init_target_x, init_target_y);
    }

    else
    {
        drone_state_ = GPS_TRACKING;
    }
}

void RL_env::iris_RL()
{
    offboard offb;
    env_info.step++;
    get_uavstate();
    env_info.reward = calcul_reward(offb,env_info.step,true);

    if(env_info.restart)
    {
    #ifndef GAZEBO_DATA
//        float image_distance = offb.iris_local_pose.z*tan(offb.view_angle);
//        ROS_WARN("distance limit : %f",image_distance);
        if(env_info.step <= MAXSTEP )
        {
//            if((env_info.step > 3) && (((abs(real_x) > image_distance) || (abs(real_y) > image_distance)) 
//                || (env_info.state.rel_pose[2] > 5 || env_info.state.rel_pose[2] < 1)))
            if(env_info.step > 3)// || (env_info.step == 1 && env_info.state.rel_pose[2] > 5))
            {
                env_info.reset_error = false;
                ROS_ERROR("Reset");
            }
            else 
            {
                env_info.reset_error = true;
                ROS_ERROR("DONT SEND STATE");
            }
        }
        else
        {
            env_info.reset_error = false;
            ROS_ERROR("Reset");
        }

        exchange_stateNaction();
        pre_shape = 0;
        offb.drone_state_ = RESET;
    #else
        exchange_stateNaction();
        ROS_ERROR("Reset");
        pre_shape = 0;
        offb.drone_state_ = RESET;
    #endif
    }
    else
    {   
        exchange_stateNaction();
        offb.drone_state_ = env_info.action.landing_cmd > 0.7 ? LAND : RL;
        if(offb.drone_state_ == LAND) landing_start_alt = env_info.state.rel_pose[2];
    }
}

void offboard::Pub_targetpose()
{
    RL_env RL_;
    float goal_x = 0, goal_y = 0, goal_z = 0;
    float goal_vx = 0, goal_vy = 0, goal_vz = 0;
    float goal_roll = 0, goal_pitch = 0, goal_yaw_speed = 0, goal_thrust = 0;
    float init_thrust = 0.56;

    switch(drone_state_)
    {
        case OFFBOARD :
        {
            goal_x = iris_local_pose.x;
            goal_y = iris_local_pose.y;
            goal_z = des_iris_alt;
        }
        break;

        case GPS_TRACKING :
        {
            if(gpson) distance_measure(mlp_gps.lat, mlp_gps.lon, iris_gps_.lat, iris_gps_.lon);

            goal_x = iris_local_pose.x + gps_distance.x;
            goal_y = iris_local_pose.y + gps_distance.y;
            goal_z = des_iris_alt;
        }
        break;

        case RL :
        {              
        #ifdef ENV_RAMOS
            goal_roll = RL_.env_info.action.roll*30*DEG2RAD;
            goal_pitch = RL_.env_info.action.pitch*30*DEG2RAD;
            goal_yaw_speed = RL_.env_info.action.yaw_speed;
            goal_thrust = saturation(0,1,RL_.env_info.action.thrust);
            //saturation(0,1,init_thrust+RL_.env_info.action.thrust);
            //saturation(0.46,0.66,init_thrust+RL_.env_info.action.thrust);
            ROS_INFO("roll : %f, pitch : %f, yawSP : %f, thrust : %f", goal_roll,goal_pitch,goal_yaw_speed,goal_thrust);
        #else
            goal_vx = saturation(MIN_VXY,MAX_VXY,RL_.env_info.action.vel[0]);
            goal_vy = saturation(MIN_VXY,MAX_VXY,RL_.env_info.action.vel[1]);
            #ifdef TRACKING
            goal_vz = PID_controller(1.1,0.0,0,globalpose_.drone_z,8);
            goal_yaw_speed = PID_controller(2,0.02,0,angle.yaw*DEG2RAD,0.00f);
            #else
            goal_vz = saturation(MIN_VZ ,MAX_VZ ,RL_.env_info.action.vel[2]);
            goal_yaw_speed = PID_controller(2,0.02,0,angle.yaw*DEG2RAD,0.00f);
 //           goal_yaw_speed = RL_.env_info.action.yaw_speed;
            #endif

            ROS_INFO("vx : %f, vy : %f, vz : %f, ysp : %f", goal_vx,goal_vy,goal_vz,goal_yaw_speed);
            ROS_INFO("error_x : %f, error_y : %f", RL_.env_info.state.rel_pose[0]-RL_.real_x, RL_.env_info.state.rel_pose[1]-RL_.real_y);
        #endif
        }
        break;

        default :
        break;
    }

    if(drone_state_ == RL)
#ifdef ENV_RAMOS
    iris_att_pub(goal_roll,goal_pitch,goal_yaw_speed,goal_thrust);
#else
    pub_sp_raw(goal_vx,goal_vy,goal_vz,goal_yaw_speed);
#endif

    else pub_pose(goal_x, goal_y, goal_z,0,0,0,1);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
void offboard::iris_VISION_tracking()
{
    ROS_INFO("distane : %f, landmode : %d",target.distance[1],landmode);
    if(landmode==0)
    {
        if((target.distance[1] <= 1.5) && (iris_local_pose.z <= 5.1))  landmode = 1;
    }
    else if(landmode==1)
    {
        if((target.distance[1] <= 0.25) && (iris_local_pose.z <= mlp_gps.alt+1.1)) landmode = 2;
    }

    if(!markerrecogn)
    {
        if(landmode==2) drone_state_ = RESET;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////

void offboard::iris_landing()
{
    RL_env RL_;
    offboard offb;
    mavros_msgs::SetMode land_set_mode;
    land_set_mode.request.custom_mode = "AUTO.LAND";
    if( set_mode_client.call(land_set_mode) &&  land_set_mode.response.mode_sent)
    {
        ROS_INFO("Land Start");
        drone_state_ = LAND;
    }
    
    ROS_INFO("iris_alt : %f, contact_mark : %d, contact_upper : %d",iris_local_pose.z,RL_.contact_recogn_mark,RL_.contact_recogn_upper);

    if(iris_local_pose.z < 0.05 || (RL_.contact_recogn_mark || RL_.contact_recogn_upper)) 
    {
        ROS_WARN("LANDING OVER");
        RL_.env_info.reward = RL_.calcul_reward(offb,RL_.env_info.step,true);
        RL_.exchange_stateNaction();
        if(RL_.env_info.episode < MAXEPISODE) drone_state_ = RESET;
        else drone_state_ = OFF;
    }   
}

bool offboard::check_trackingtime()
{
    RL_env RL_;
    if(drone_state_ != RL) return false;
    else 
    {
        float mlp_distance = sqrt(globalpose_.mlp_x*globalpose_.mlp_x+globalpose_.mlp_y*globalpose_.mlp_y);
        trackingTime += dt;
        ROS_INFO("mlp_distance = %f", mlp_distance);
        if(mlp_distance >= 45) return true; //trackingTime >= 600
        //else if(!markerrecogn) return false;
        else return false;
    }
}

int offboard::run()
{
    static RL_env RL_;
    //get_joint_state();
    //set_link_orientation(gimbal.yaw_link_name,0,0,0);
    switch(drone_state_)
    {
        case INIT:
        iris_initializing();

        case ARM :
        iris_arming();
        break;
        
        case OFFBOARD :
        iris_offb_start();
        break;

        case GPS_TRACKING :
        iris_GPS_tracking();
        break;

        case RL :
        RL_.iris_RL();
        //iris_VISION_tracking();
        break;

        case LAND :
        iris_landing();
        break;

        case OFF :

        //exit(0);   // system off
        break;

        case RESET :
        {
            if(abs(globalvel_.mlp_y) > 0.01) RL_.mlp_state_ = MLPSTOP;
            else
            {
                RL_.SetModelReset(true,"iris", 0.0, init_rel_pose_y+0.2, des_iris_alt, 0.0, 0.0, 0.0); // 9.3 // -2.7
                if(RL_.reset_client.call(RL_.reset))
                {
                    RL_.SetModelReset(false,"Mobile_Leveling_Platform_V2", 0.0, 0.0, 0.20);
                    if(RL_.reset_client.call(RL_.reset)) 
                    {
                    #ifndef GAZEBO_DATA
                        if(!RL_.env_info.reset_error) RL_.env_info.episode++;
                        else ROS_ERROR("DONT COUNT EPISODE");
                    #else
                        RL_.env_info.episode++;
                    #endif
                        RL_.env_info.step = 0;
                        drone_state_ = STABLE;
                    }
                    else 
                    {
                        drone_state_ = RESET;
                        ROS_ERROR("reset service require error, LMP");
                    }
                }
                else 
                {
                    drone_state_ = RESET;
                    ROS_ERROR("reset service require error, iris");
                }
                if(RL_.env_info.episode > MAXEPISODE || bTestFinished) 
                {
                    ROS_WARN("TEST OVER, UAV XY VEL = %d, UAV Z VEL = %d ,MLP Vel = %d",MAX_VXY,MAX_VZ,MLPVELOCITY);
                    drone_state_ = OFF;
                }

            }
        }
        break;

        case STABLE :

            pub_pose(iris_local_pose.x,iris_local_pose.y,des_iris_alt); // 9.4
            if(current_state.mode != "OFFBOARD")
            {
                offb_set_mode.request.custom_mode = "OFFBOARD";
                set_mode_client.call(offb_set_mode);
            }
#ifdef GAZEBO_DATA
            if((abs(abs(iris_local_pose.z)-8) < 0.1))//  && abs(angle.roll[0]) < 3 && abs(angle.pitch[0]) < 3 && abs(angle.yaw) < 0.05
            {
#else
            if((abs(abs(iris_local_pose.z)-des_iris_alt) < 0.1) && markerrecogn)// && abs(iris_vel.x) < 0.1 && abs(iris_vel.x) < 0.1 && abs(iris_vel.x) < 0.1)
            {
#endif                
                {
//                    RL_.SetModelvelReset(0,4.653,0);//0,4.653,0
                    RL_.SetModelvelReset(0,init_velocity_y,0);//0,4.653,0
                    drone_state_ = INIT;
                }
            }
            else 
            {
                RL_.SetModelReset(false,"iris", RL_.save_iris_x,RL_.save_iris_y, des_iris_alt, 0.0, 0.0, 0.0); //RL_.save_iris_x,RL_.save_iris_y // 9.4 ==> 20
                RL_.reset_client.call(RL_.reset);
                drone_state_ = STABLE;
            }
            //ROS_WARN("real_z : %f",globalpose_.drone_z);

        break;

        default :
        break;
    }

    RL_.MLPcontrol();
    if(!(drone_state_==LAND || drone_state_==RESET || drone_state_== STABLE || drone_state_== INIT)) Pub_targetpose();

    int current_ep_step = RL_.env_info.step;

    dataWrite(globalpose_.drone_x, globalpose_.drone_y, globalpose_.drone_z,
              globalpose_.mlp_x  , globalpose_.mlp_y  , mlp_gps.alt,
              (float)current_ep_step, 
              RL_.env_info.restart ? 1.0f : 0.0f, 
              RL_.env_info.episode, 
              current_ep_step);
              
    return 0;
}

//////////////////////////////main///////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "offb_node");
    offboard offb;
    RL_env RL_;
    mavros_msgs::State current_state;
    offb.gimbal_set();

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(30); 

    nTestflag   =  atoi(argv[1]);

    while(ros::ok())
    {
        offb.run();
        ros::spinOnce();
        rate.sleep();
    }

    // wait for FCU connection
    while(ros::ok() && !current_state.connected)
    {
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}

