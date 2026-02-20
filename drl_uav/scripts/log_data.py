#!/usr/bin/env python
import rospy
from drl_uav.msg import log_data
import numpy

log_dataset = log_data()
get_data = False

default_path = '/home/baek/traj_data/'
traj_path = 'traj/'
error_path = 'error/'
return_path = 'return/'
landingstate_path = 'landingstate/'
landingstate_file_name = 'landingstate.txt'
traj_file_name = 'traj.txt'
error_file_name = 'error.txt'
return_file_name = 'return.txt'

def callback(log_data):
    global get_data, log_dataset
    log_dataset.uav_x = log_data.uav_x
    log_dataset.uav_y = log_data.uav_y
    log_dataset.uav_z = log_data.uav_z
    log_dataset.target_x = log_data.target_x
    log_dataset.target_y = log_data.target_y
    log_dataset.target_z = log_data.target_z 
    log_dataset.landingstate = log_data.landingstate     
    log_dataset.restart = log_data.restart
    log_dataset.episode = log_data.episode
    log_dataset.step = log_data.step
    get_data = True

def main_loop():
    global get_data, log_dataset
    rospy.init_node('log_data', anonymous=True)

    rospy.Subscriber("/getLogdata", log_data, callback)

    r = rospy.Rate(30)
    past_episode = 0
    drone_traj  = []
    target_traj = []

    while not rospy.is_shutdown():
        if get_data:
            drone_traj.append((log_dataset.uav_x,log_dataset.uav_y,log_dataset.uav_z))
            target_traj.append((log_dataset.target_x,log_dataset.target_y,log_dataset.target_z))
            get_data = False
        print("landingstate : ",log_dataset.landingstate," restart : ",log_dataset.restart)
        if log_dataset.restart==1 and log_dataset.episode-past_episode > 0:
            if log_dataset.landingstate==1:
                with open(default_path+traj_path+traj_file_name+str(episode), 'a') as f:
                    for i in range(log_dataset.step):
                        f.write('uav x : ' + str(drone_traj[i][1])+ ' uav y : ' + str(drone_traj[i][2]) + ' uav z : ' + str(drone_traj[i][3])\
                        + 'target x : ' + str(target_traj[i][1])+ ' target y : ' + str(target_traj[i][2]) + ' target z : ' + str(target_traj[i][3])+ '\n')
            numpy.empty_like(drone_traj)
            numpy.empty_like(target_traj)
            with open(default_path+landingstate_path+landingstate_file_name, 'a') as landing:
                landing.write('episode : ' + str(log_dataset.episode) + ' landing state : ' + str(log_dataset.landingstate) + '\n')
            past_episode = log_dataset.episode
        r.sleep()


if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print ('Interrupted')
        sys.exit(0)