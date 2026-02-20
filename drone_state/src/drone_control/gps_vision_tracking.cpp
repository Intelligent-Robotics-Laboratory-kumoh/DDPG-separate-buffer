#include "offboard.h"

offboard::offboard()
: nh_priv("~")
{
  //Init offboard node
  bool init_result = initializing();
  ROS_ASSERT(init_result);
}

offboard::~offboard()
{
}

bool offboard::initializing() // initialing
{
    drone_state_ = ARM;
    landmode = 0;
    arming_fail_cnt = 0;
    init_target_x = 0;
    init_target_y = 0;
    des_iris_alt = 9.5;
    trackingTime = 0;
    write = true;

    local_pos_pub =     offb_nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);
    local_vel_pub =     offb_nh.advertise<geometry_msgs::Twist>("/mavros/setpoint_velocity/cmd_vel_unstamped", 10);
    arming_client =     offb_nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    set_mode_client =   offb_nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    state_sub =         offb_nh.subscribe("/mavros/state", 10, &offboard::state_cb, this);
    mlp_local_pose =    offb_nh.subscribe("/gazebo/model_states", 10, &offboard::mlp_global, this);
    MLP_GPS =           offb_nh.subscribe("/platform/MLP/fix",10,&offboard::get_MLP_position, this);
    iris_GPS =          offb_nh.subscribe("/mavros/global_position/raw/fix",10,&offboard::get_iris_position, this);
    iris_local_pos_sub= offb_nh.subscribe("/mavros/local_position/pose",10,&offboard::get_iris_localposition, this);
    marker_pose =       offb_nh.subscribe("/aruco_single/pixel",10,&offboard::markerPixelGet, this);
    iris_imu =          offb_nh.subscribe("/mavros/local_position/pose",10,&offboard::Quarter2Euler, this);
    MarkerRecogn =      offb_nh.subscribe("/aruco_single/markerRecogn",10,&offboard::markrecogn_callback, this);
    iris_vel =          offb_nh.subscribe("/mavros/local_position/velocity_local",10,&offboard::get_iris_vel, this);    
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
    mlp_gps_.lat = MLP_GPS->latitude;
    mlp_gps_.lon = MLP_GPS->longitude;
    mlp_gps_.alt = MLP_GPS->altitude;
    gpson = true;
}

void offboard::mlp_global(const gazebo_msgs::ModelStatesConstPtr &global_)
{
    globalpose_.mlp_x = global_->pose[4].position.x;
    globalpose_.mlp_y = global_->pose[4].position.y;
    globalpose_.mlp_z = global_->pose[4].position.z;

    globalpose_.drone_x = global_->pose[5].position.x;
    globalpose_.drone_y = global_->pose[5].position.y;
    globalpose_.drone_z = global_->pose[5].position.z;
}

void offboard::get_iris_position(const sensor_msgs::NavSatFixConstPtr &iris_GPS)
{
    iris_gps_.lat = iris_GPS->latitude;
    iris_gps_.lon = iris_GPS->longitude;
    iris_gps_.alt = iris_GPS->altitude;
}

void offboard::get_iris_vel(const geometry_msgs::TwistStampedConstPtr &iris_vel)
{
    vel_x = iris_vel->twist.linear.x;
    vel_y = iris_vel->twist.linear.y;
}

void offboard::markerPixelGet(const geometry_msgs::PointStampedConstPtr &markerpixel)
{
    mark_buf[0][0] = mark_buf[2][0];
    mark_buf[0][1] = mark_buf[2][1];

    target.distance[0] = target.distance[1];
    target.distance[2] = target.distance[3];
        
    mark_buf[1][0] = markerpixel->point.x;
    mark_buf[1][1] = markerpixel->point.y;
        
    mark_buf[2][0] = LPF(mark_buf[0][0], mark_buf[1][0]); 
    mark_buf[2][1] = LPF(mark_buf[0][1], mark_buf[1][1]);

    float correction_x = ((tan(view_angle)-tan(view_angle-angle.pitch[1]))*400/(iris_local_pose.z-mlp_gps_.alt)/tan(view_angle));
    float correction_y = ((tan(view_angle)-tan(view_angle-angle.roll[1]))*400/(iris_local_pose.z-mlp_gps_.alt)/tan(view_angle));

    float target_x = mark_buf[2][0]+correction_x;
    float target_y = mark_buf[2][1]+correction_y;

	//distance from camera to target / (half pixel size) * tan(angle of view/2) * (goal(pixel) - mark position(pixel))
    target.curr_x = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[2][0]);	// only LPF
    target.curr_y = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[2][1]);

//    target.curr_x = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[1][0]); // no LPF
//    target.curr_y = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-mark_buf[1][1]);

//    target.curr_x = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-saturation(0,800,target_x)); // correction and LPF
//    target.curr_y = ((iris_local_pose.z-mlp_gps_.alt)/400)*tan(view_angle) * (400-saturation(0,800,target_y));

    target.dif_x = (target.pre_x - target.curr_x)/dt;
    target.dif_y = (target.pre_x - target.curr_x)/dt;

    target.pre_x = target.curr_x;
    target.pre_y = target.curr_y;

    XYtransform(target.curr_x,target.curr_y);
    slewrate(target.curr_x, target.curr_x, target.vel_x, target.vel_y);

    target.distance[1] = abs(sqrt(target.curr_x*target.curr_x + target.curr_y*target.curr_y));
    target.distance[3] = (target.distance[1] - target.distance[0])/dt;
}

void offboard::Quarter2Euler(const geometry_msgs::PoseStampedConstPtr &iris_angle)
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
}

void offboard::markrecogn_callback(const aruco_msgs::MarkerRecogn::ConstPtr& markRecogn_)
{
   markerrecogn = markRecogn_->MarkRecogn;
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

    XYtransform(gps_distance_.x,gps_distance_.y);

    return gps_distance_.x, gps_distance_.y, gps_distance_.meter;
}

/////////////////////////Variable K - sigmoid form//////////////////////////////////
float offboard::sigmoid_beta(float init_target_distance, float error)
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
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    /*
    float init = 0.3;

    float sigbeta;
    if(error <= 0.3*init_target_distance) test = true;
    if(test)
    {   
        sec += 0.001;
        sigbeta = (BETA_MAX-init)/(1+pow(EXP,GRADIENT*(sec-SIGMOID_TH)))+init+(1-init)/(1+pow(EXP, 10*(sec-0.1)));
        //sigbeta = (BETA_MAX-init)/(1+pow(EXP,GRADIENT*(sec-SIGMOID_TH)))+init;
        //ROS_INFO("sigbeta : %f", sigbeta);
    }
    else
    {   
        sigbeta = 1;
        //sigbeta = (1-init)/(1+pow(EXP,10*(sec-0.3)));
        //ROS_INFO("sigbeta : %f", sigbeta);
    }
    */
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float init = 1/abs(init_target_distance)*0.5;
          init = saturation(0,0.5,init);
          init = 0.2;
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
		_sp_pev_prev(0) = _sp_pev(0) - vel_x * dt;
		_sp_pev_prev(1) = _sp_pev(1) - vel_y * dt;
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
    float iris_yaw_rad = atan2(y_frame,x_frame);
    float curr_xframe =  cos(iris_yaw_rad)* x_frame + sin(iris_yaw_rad)* y_frame;
    float curr_yframe = -sin(iris_yaw_rad)* x_frame + cos(iris_yaw_rad)* y_frame;
    return x_frame = curr_yframe, y_frame = curr_xframe;
}

////////////save a logdata(position of MLP and drone)/////////////// 
void offboard::dataWrite(float pose_x, float pose_y, float pose_z,
                         float mlp_x, float mlp_y, float mlp_z, 
                         float init_marker_x, float init_marker_y)
{
    std::ofstream fout;
    if(current_state.armed == true)
    {
        fout.open("/home/baek/data/pose_x.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<pose_x;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/pose_y.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<pose_y;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/pose_z.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<pose_z;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/mlp_x.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<mlp_x;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/mlp_y.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<mlp_y;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/mlp_z.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<mlp_z;
            fout<<"\n";
            fout.close();
        }
/*
        fout.open("/home/baek/data/mlp_gps.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<"mlp_lat : "<<mlp_gps_.lat<<" : mlp_lon : "<<mlp_gps_.lon;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/drone_gps.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<"drone_lat : "<<iris_gps_.lat<<" : drone_lon : "<<iris_gps_.lon;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/gps2meter.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<"x : "<<gps_distance_.x<<" : y : "<<gps_distance_.y;
            fout<<"\n";
            fout.close();
        }

        fout.open("/home/baek/data/vision.txt",std::ios_base::app);
        if(fout.is_open())
        {
            fout<<"x : "<<target.curr_x<<" : y : "<<target.curr_y<<" : roll : "<<angle.roll[0]<<" : pitch : "<<angle.pitch[0];
            fout<<"\n";
            fout.close();
        }*/



        if((init_target_x != 0 && init_target_y != 0) && write)
        {
            fout.open("/home/baek/data/marker.txt");
            if(fout.is_open())
            {
                fout<<pose_x;
                fout<<"\n";
                fout<<pose_y;
                fout<<"\n";
                fout<<"\n";
                fout<<mlp_x;
                fout<<"\n";
                fout<<mlp_y;
                fout<<"\n";
                fout.close();
            }
            write = false;
        }
    }
}

/////////////////////gps and vision tracking//////////////////////////
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
    if(iris_arming())
    {  
        if(current_state.mode != "OFFBOARD")
        {
            offb_set_mode.request.custom_mode = "OFFBOARD";
            set_mode_client.call(offb_set_mode);
        }

        if(iris_local_pose.z > des_iris_alt-0.1) 
        {
            ROS_INFO("GPS Tracking Start");
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
    if(markerrecogn)
    {
        init_target_x = iris_local_pose.x + target.curr_x;
        init_target_y = iris_local_pose.y + target.curr_y;
        drone_state_  =  VISION_TRACKING;
        ROS_INFO("Vision Tracking Start, target distance is : %f, X : %f, Y : %f", target.distance[1], init_target_x, init_target_y);
    }

    else drone_state_ = GPS_TRACKING;
}

void offboard::iris_VISION_tracking()
{
    ROS_INFO("distane : %f, landmode : %d",target.distance[1],landmode);
    if(landmode==0)
    {
        if((target.distance[1] <= 1.5) && (iris_local_pose.z <= 5.1))  landmode = 1;
    }
    else if(landmode==1)
    {
        if((target.distance[1] <= 0.25) && (iris_local_pose.z <= mlp_gps_.alt+1.1)) landmode = 2;
    }

    if(!markerrecogn)
    {
        if(landmode==2) drone_state_ = LAND;
    }
}

void offboard::Pub_targetpose()
{
    float goal_x = 0;
    float goal_y = 0;
    float goal_z = 0;

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
            if(gpson)
            {
                distance_measure(mlp_gps_.lat, mlp_gps_.lon, iris_gps_.lat, iris_gps_.lon);
            }
            goal_x = iris_local_pose.x + gps_distance_.x;
            goal_y = iris_local_pose.y + gps_distance_.y;
            goal_z = des_iris_alt;
            //ROS_INFO("gps x : %f",gps_distance_.x );
            //ROS_INFO("gps y : %f",gps_distance_.y );
        }
        break;

        case VISION_TRACKING :
        {
            goal_x = iris_local_pose.x + target.curr_x * sigmoid_beta(init_target_x,target.curr_x);
            goal_y = iris_local_pose.y + target.curr_y * sigmoid_beta(init_target_y,target.curr_y);
            switch(landmode)
            {   
                case 0 :
                goal_z = descent_flight(10,5);
                break;
                case 1 :
                goal_z = descent_flight(15,mlp_gps_.alt+1);
                break;
                case 2 :
                goal_z = descent_flight(20,mlp_gps_.alt);
                default :
                break;
            }
        }
        break;

        default :
        break;
    }
    pub_pose(goal_x, goal_y, goal_z);
}

void offboard::pub_pose(float targetX, float targetY, float targetZ)
{  
    pose.pose.position.x = targetX;
    pose.pose.position.y = targetY;
    pose.pose.position.z = targetZ;
    local_pos_pub.publish(pose);
}

void offboard::iris_landing()
{
    mavros_msgs::SetMode land_set_mode;
    land_set_mode.request.custom_mode = "AUTO.LAND";
    if( set_mode_client.call(land_set_mode) &&  land_set_mode.response.mode_sent)
    {
        ROS_INFO("Land Start, Visual distance : %f, pixel : %f, alt : %f",target.distance[1], mark_buf[2][0], iris_local_pose.z-mlp_gps_.alt);
        sleep(1);
    }
    if(!current_state.armed) 
    {
        ROS_INFO("Landing is over. Turn off offb system");
        drone_state_ = OFF;
    }
}

bool offboard::check_trackingtime()
{
    if(drone_state_ != VISION_TRACKING) return false;
    else 
    {
        float mlp_distance = sqrt(globalpose_.mlp_x*globalpose_.mlp_x+globalpose_.mlp_y*globalpose_.mlp_y);
        trackingTime += dt;
        ROS_INFO("mlp_distance = %f", mlp_distance);
        if(mlp_distance >= 45) return true; //trackingTime >= 600 
        else return false;
    }
}

int offboard::run()
{
    switch(drone_state_)
    {
        case ARM :
        iris_arming();
        break;
        
        case OFFBOARD :
        iris_offb_start();
        break;

        case GPS_TRACKING :
        iris_GPS_tracking();
        break;

        case VISION_TRACKING :
        iris_VISION_tracking();
        break;

        case LAND :
        iris_landing();
        break;

        case OFF :
        exit(0);   // system off
        break;

        default :
        break;
    }

    if(drone_state_!=LAND) Pub_targetpose();

    if(check_trackingtime())
    {
        ROS_ERROR("TRACKING TIME IS TOO LONG");
        drone_state_ = LAND;
    }

    dataWrite(globalpose_.drone_x, globalpose_.drone_y, globalpose_.drone_z,
              globalpose_.mlp_x  , globalpose_.mlp_y  , globalpose_.mlp_z,
              init_target_x, init_target_y);
}

//////////////////////////////main///////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "offb_node");
    offboard offb;
    mavros_msgs::State current_state;    

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(100);

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
