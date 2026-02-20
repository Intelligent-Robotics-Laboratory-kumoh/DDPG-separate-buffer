#!/usr/bin/env python
# An implementation of UAV DRL

# Func:
# 1) kepp the pos1 fixed in 6+-0.3  D!
#
# Implementation:
# 1) Work with player_test.py   D!
#
# Subscribe: game(Environment) status
# Publish: action: only sent when game status is received
#
# author: bingbing li 07.02.2018

from tensorflow.python.ops.numpy_ops.np_math_ops import negative
from tensorflow.python.ops.summary_ops_v2 import initialize
import rospy

from drl_uav.srv import getEnvinfo, getEnvinfoResponse	         # Get state 
from drl_uav.srv import getEnvinfo_att, getEnvinfo_attResponse

#from keras.models import Sequential
#from keras.layers import *
#from keras.optimizers import *
import random, numpy, math, gym, time
import tensorflow as tf

from functools import partial
from collections import deque

tf.debugging.set_log_device_placement(True)

tf.compat.v1.disable_eager_execution()

import sys, os

mu = 5e-4

theta = 0.1
sigma = 0.1

actor_layer1_size = 400
actor_layer2_size = 300

normalize_mean = 0
normalize_val  = 1
scale  = 1
shift  = 0
val_epsilon = 1e-5

class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = numpy.ones(1) * mu

    def sample(self):
        dx = theta * (mu - self.X)
        dx += sigma * numpy.random.randn(len(self.X))
        self.X += dx
        return self.X

class Actor:
    def __init__(self, state_size, action_size, name):
        
        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, state_size])
            self.bn0 = tf.nn.elu(tf.nn.batch_normalization(self.state,normalize_mean,normalize_val,shift,scale,val_epsilon))            
            self.fc1 = tf.compat.v1.layers.dense(self.state, 900, activation=None)
            self.bn1 = tf.nn.elu(tf.nn.batch_normalization(self.fc1,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc2 = tf.compat.v1.layers.dense(self.bn1, 800, activation=None)
            self.bn2 = tf.nn.elu(tf.nn.batch_normalization(self.fc2,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc3 = tf.compat.v1.layers.dense(self.bn2, 700, activation=None)
            self.bn3 = tf.nn.elu(tf.nn.batch_normalization(self.fc3,normalize_mean,normalize_val,shift,scale,val_epsilon))        
#            self.fc4 = tf.compat.v1.layers.dense(self.bn3, 900, activation=None)
#            self.bn4 = tf.nn.elu(tf.nn.batch_normalization(self.fc4,normalize_mean,normalize_val,shift,scale,val_epsilon))            
#            self.fc5 = tf.compat.v1.layers.dense(self.bn4, 400, activation=tf.nn.elu)
#            self.bn5 = tf.nn.elu(tf.nn.batch_normalization(self.fc5,normalize_mean,normalize_val,shift,scale,val_epsilon))            
            self.action = tf.compat.v1.layers.dense(self.bn3, action_size, activation=tf.nn.tanh)
        self.trainable_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, name)


class Critic:
    def __init__(self, state_size, action_size, name):
        
        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, state_size])
            self.action = tf.compat.v1.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.state, self.action], axis=-1)
            self.bn0 = tf.nn.elu(tf.nn.batch_normalization(self.concat,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc1 = tf.compat.v1.layers.dense(self.concat, 900, activation=None)
            self.bn1 = tf.nn.elu(tf.nn.batch_normalization(self.fc1,normalize_mean,normalize_val,shift,scale,val_epsilon))         
            self.fc2 = tf.compat.v1.layers.dense(self.bn1, 800, activation=None)
            self.bn2 = tf.nn.elu(tf.nn.batch_normalization(self.fc2,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc3 = tf.compat.v1.layers.dense(self.bn2, 700, activation=None)
            self.bn3 = tf.nn.elu(tf.nn.batch_normalization(self.fc3,normalize_mean,normalize_val,shift,scale,val_epsilon))
#            self.fc4 = tf.compat.v1.layers.dense(self.bn3, 900, activation=None)
#            self.bn4 = tf.nn.elu(tf.nn.batch_normalization(self.fc4,normalize_mean,normalize_val,shift,scale,val_epsilon))
#            self.fc5 = tf.compat.v1.layers.dense(self.bn4, 400, activation=tf.nn.elu)
#            self.bn5 = tf.nn.elu(tf.nn.batch_normalization(self.fc5,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.predict_q = tf.compat.v1.layers.dense(self.bn3, 1, activation=None)
        self.trainable_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, name)

class Agent:
    steps = 0
    def __init__(self, num_state, num_action):
        # parameters of External Environment:
        self.num_state = num_state
        self.num_action = num_action

        # parameters of Internal DRL algorithm:
        ## Memory:
        self.MEMORY_CAPACITY = 1000000 # 1000000
        self.MEMORY_CAPACITY_HALF = int(self.MEMORY_CAPACITY/2)
        ## RL algorithm:
        self.GAMMA = 0.99
        ## Deep network: 
        self.MEMORY_BATCH_SIZE = 256 # number of data for one training! ?(Maybe we can set MEMORY_BATCH_SIZE = MEMORY_CAPACITY)
        ## Random selection proportion:
        self.MAX_EPSILON = 1
        self.EPSILON_DECAY = 0.999
        self.MIN_EPSILON = 0.001
        self.tau = 5e-5 # 5e-3 #1E-5cv                 #21.10.18 tau : 3e-5 alr : 7e-9 clr : 7e-8
        self.actor_lr = 1e-5 # 1e-3 #normal1 ==> 1e-4 hidden layer 2 2E-6
        self.critic_lr = 1e-4  # 5e-3 #normal1 ==> 1e-3 hidden layer 2 2E-5
        self.policy_noise = 0.2
        self.clip_value = 0.5    
   

        self.epsilon = self.MAX_EPSILON

        self.actor_main = Actor(self.num_state, self.num_action, "main_actor")
        self.actor_target = Actor(self.num_state, self.num_action, "target_actor")
        self.critic_main_1 = Critic(self.num_state, self.num_action, "main_critic_1")
        self.critic_target_1 = Critic(self.num_state, self.num_action, "traget_critic_1")
        self.critic_main_2 = Critic(self.num_state, self.num_action, "main_critic_2")
        self.critic_target_2 = Critic(self.num_state, self.num_action, "traget_critic_2")

        self.target_q = tf.compat.v1.placeholder(tf.float32, [None, 1])

        critic_loss = tf.compat.v1.losses.mean_squared_error(self.target_q, self.critic_main.predict_q)
        with tf.control_dependencies(self.critic_main.trainable_var):
            self.train_critic = tf.compat.v1.train.AdamOptimizer(self.critic_lr).minimize(critic_loss)


        action_grad = tf.clip_by_value(tf.gradients(ys = tf.squeeze(self.critic_main.predict_q), xs = self.critic_main.action), -10, 10) # default -10 10

        policy_grad = tf.gradients(ys=self.actor_main.action, xs=self.actor_main.trainable_var, grad_ys=action_grad)
        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads / self.MEMORY_BATCH_SIZE
        with tf.control_dependencies(self.actor_main.trainable_var):
            self.train_actor = tf.compat.v1.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(policy_grad, self.actor_main.trainable_var))

        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.Noise = OU_noise()
        self.memory = deque(maxlen=self.MEMORY_CAPACITY)
        self.positive_threshold = 0
        self.positive_memory = deque(maxlen=self.MEMORY_CAPACITY_HALF)
        self.negative_memory = deque(maxlen=self.MEMORY_CAPACITY_HALF)

        self.soft_update_target = []
        for idx in range(len(self.actor_main.trainable_var)):
            self.soft_update_target.append(self.actor_target.trainable_var[idx].assign(((1 - self.tau) * self.actor_target.trainable_var[idx].value()) + (self.tau * self.actor_main.trainable_var[idx].value())))
        for idx in range(len(self.critic_main.trainable_var)):
            self.soft_update_target.append(self.critic_target.trainable_var[idx].assign(((1 - self.tau) * self.critic_target.trainable_var[idx].value()) + (self.tau * self.critic_main.trainable_var[idx].value())))

        init_update_target = []
        for idx in range(len(self.actor_main.trainable_var)):
            init_update_target.append(self.actor_target.trainable_var[idx].assign(self.actor_main.trainable_var[idx]))
        for idx in range(len(self.critic_main.trainable_var)):
            init_update_target.append(self.critic_target.trainable_var[idx].assign(self.critic_main.trainable_var[idx]))
        self.sess.run(init_update_target)

    def act(self, state):   # action:[0,1,2,...,num_action-1]

        # Limit: 3) forced input in Emergency: Vz is out of [-3,3]
        action = self.sess.run(self.actor_main.action, feed_dict={self.actor_main.state: [state]})

        action_ = numpy.zeros((3))
        noise = self.Noise.sample()

#        for i in range(3):
#            action_[i] = action[0][i] + (self.epsilon * noise)
#            if(action_[i] > 1):
#                action_[i] = 1
#            elif(action_[i]<-1):
#                action_[i] = -1
#        print(action_)
        for i in range(3):
            action_[i] = action[0][i] + (self.epsilon * noise)
            if(action_[i] > 1):
                action_[i] = 1
            elif(action_[i]<-1):
                action_[i] = -1
        print(action_)

        return action_

    def observe(self, state, action, reward, next_state, done, step):
        if(step==1):
            self.positive_threshold = reward*0.75
        if(reward<self.positive_threshold):
            self.negative_memory.append((state, action, reward, next_state, done))
        else:
            self.positive_memory.append((state, action, reward, next_state, done))
        print("threshold : ",self.positive_threshold," negative len: ",len(self.negative_memory)," positive len: ",len(self.positive_memory))

        return len(self.positive_memory)
        # decrease Epsilon to reduce random action and trust more in greedy algorithm
        #self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.steps)
        #if self.epsilon > self.MIN_EPSILON:
        #    self.epsilon *= self.EPSILON_DECAY

    def save_replay(self, state, action, reward, next_state, done, step):
        self.memory.append((state, action, reward, next_state, done))
        return len(self.memory)

    def deletestate(self, step):
        del_idx = 1
        for del_idx in range(step):
            del self.memory[-1]    

    def replay(self):   # get knowledge from experience!
        global Envinfo, episode_R,  past_episode_R
        if min(len(self.negative_memory),len(self.positive_memory)) > self.MEMORY_BATCH_SIZE*2:
#        if len(self.memory) > 5000:
            print("RL-ing")
            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
            elif self.epsilon < self.MIN_EPSILON:
                self.epsilon = self.MIN_EPSILON

#            mini_batch = random.sample(self.memory, self.MEMORY_BATCH_SIZE)
            positive_batch = random.sample(self.positive_memory, round(self.MEMORY_BATCH_SIZE*max(len(self.negative_memory),len(self.positive_memory))/(len(self.positive_memory)+len(self.negative_memory))))
            negative_batch = random.sample(self.negative_memory, round(self.MEMORY_BATCH_SIZE*min(len(self.negative_memory),len(self.positive_memory))/(len(self.positive_memory)+len(self.negative_memory))))
#            positive_batch = random.sample(self.positive_memory, round(self.MEMORY_BATCH_SIZE*0.5))
#            negative_batch = random.sample(self.negative_memory, round(self.MEMORY_BATCH_SIZE*0.5))
            mini_batch = positive_batch+negative_batch
            # batch = self.memory.sample(self.memory.num_experience())  # the training data size is too big!
            start = time.time()
            states = numpy.asarray([sample[0] for sample in mini_batch])
            actions = numpy.asarray([sample[1] for sample in mini_batch])
            rewards = numpy.asarray([sample[2] for sample in mini_batch])
            next_states = numpy.asarray([sample[3] for sample in mini_batch])
            dones = numpy.asarray([sample[4] for sample in mini_batch])

            states = normalize(states)
            actions = normalize(actions)
            rewards = normalize(rewards)
            next_states = normalize(next_states)
            dones = numpy.asarray([sample[4] for sample in mini_batch])

            start = time.time()
            nor_dist = tf.clip_by_value(tf.random_normal([1]),-self.clip_value,self.clip_value) * self.policy_noise
            actor_target_actions = self.sess.run(self.actor_target.action, feed_dict={self.actor_target.state: next_states}) + nor_dist
            critic_target_predict_qs_1 = self.sess.run(self.critic_target_1.predict_q, feed_dict={self.critic_target_1.state: next_states, self.critic_target.action: actor_target_actions})
            critic_target_predict_qs_2 = self.sess.run(self.critic_target_2.predict_q, feed_dict={self.critic_target_2.state: next_states, self.critic_target.action: actor_target_actions})

            critic_target_predict_qs = tf.math.minimum(critic_target_predict_qs_1,critic_target_predict_qs_2)            
  
            target_qs = numpy.asarray([reward + self.GAMMA * (1 - done) * critic_target_predict_qs for reward, critic_target_predict_qs, done in zip(rewards, critic_target_predict_qs, dones)])

            self.sess.run(self.train_critic, feed_dict={self.critic_main.state: states, self.critic_main.action: actions, self.target_q: target_qs})
            actions_for_train = self.sess.run(self.actor_main.action, feed_dict={self.actor_main.state: states})
            self.sess.run(self.train_actor, feed_dict={self.actor_main.state: states, self.critic_main.state: states, self.critic_main.action: actions_for_train})
            self.sess.run(self.soft_update_target)

        #    if Envinfo.episode % 100 == 0:
        #        self.saver.save(self.sess, 'saved_networks/' +'fixed/'+ 'network' + '-ddpg_fixedlanding', global_step = Envinfo.episode)

global episode_R

def normalize(value_batch):
    batch_ep = 1e-5
    batch_mean = numpy.sum(value_batch)/len(value_batch)
    batch_val  = numpy.sum((value_batch-batch_mean)**2)/len(value_batch)
    normalized_value = (value_batch-batch_mean)/math.sqrt(batch_val+batch_ep)
    return normalized_value

def get_Envinfo_callback(data):
    global Envinfo, env_action, agent, state, next_state, episode_R, update_start, landingstate
    start_time=time.time()
    Envinfo.rel_x     = data.rel_x
    Envinfo.rel_y     = data.rel_y
    Envinfo.rel_z     = data.rel_z
    Envinfo.rel_pose_xdot = data.rel_pose_xdot
    Envinfo.rel_pose_ydot = data.rel_pose_ydot
    Envinfo.rel_pose_zdot = data.rel_pose_zdot
    Envinfo.episode    = data.episode
    Envinfo.step       = data.step
    Envinfo.reward     = data.reward
    Envinfo.yaw        = data.yaw
    Envinfo.yaw_speed  = data.yaw_speed
    Envinfo.done       = data.done
    Envinfo.reset_error      = data.reset_error
    Envinfo.mark_recogn = data.mark_recogn

    current_memory_size = 0

    if Envinfo.reset_error == False:
        if(Envinfo.step != 0):
            next_state = interact()
            current_memory_size = agent.observe(state, env_action, Envinfo.reward, next_state, Envinfo.done, Envinfo.step)
#            current_memory_size = agent.save_replay(state, env_action, Envinfo.reward, next_state, Envinfo.done, Envinfo.step)
            state = next_state

        else:
            state = interact()

        env_action = agent.act(state)
        update_start = True
        episode_R += Envinfo.reward

    else:
        env_action[0] = 0
        env_action[1] = 0
        env_action[2] = 0
#        env_action[3] = 0
        episode_R = 0
#        agent.deletestate(Envinfo.step)
        update_start = False
    
    print("landingstate : ",landingstate)
    print("total_time : ",time.time()-start_time)
    return getEnvinfoResponse(uav_vx = 1 * env_action[0], uav_vy = 1 * env_action[1], uav_vz = 1 * env_action[2])#, uav_yaw_sp = 1 * env_action[3])

def interact():

    # publish env_input(action):
    global Envinfo, done

    MAXUAVVEL_XY = 5
    MAXUAVVEL_Z  = 1
    MAXUAVPOSE_Z = 8
    MINUAVPOSE_Z = 1.25

    camera_fov   = 80*math.pi/180
    marker_size = 0.3556
    MAX_ARUCO_POSE = 8*math.tan(camera_fov/2)-marker_size/2
    image_size   = 400

#    normalized_vel_x = Envinfo.uav_vx / MAXUAVVEL_XY
#    normalized_vel_y = Envinfo.uav_vy / MAXUAVVEL_XY
#    normalized_vel_z = Envinfo.uav_vz / MAXUAVVEL_Z

    #using gazebo data
#    normalized_rel_x = Envinfo.rel_x/10  #/ (Envinfo.rel_z*math.tan(camera_fov/2)) 
#    normalized_rel_y = Envinfo.rel_y/10  #/ (Envinfo.rel_z*math.tan(camera_fov/2))
#    normalized_rel_pose_xdot = Envinfo.rel_pose_xdot / 20
#    normalized_rel_pose_ydot = Envinfo.rel_pose_ydot / 20
    
    #using camera data
    normalized_rel_x = Envinfo.rel_x/(MAXUAVPOSE_Z*math.tan(camera_fov/2))
    normalized_rel_y = Envinfo.rel_y/(MAXUAVPOSE_Z*math.tan(camera_fov/2))
    normalized_rel_z = 2*(Envinfo.rel_z-3.375)  / (MAXUAVPOSE_Z-MINUAVPOSE_Z)    
    normalized_rel_pose_xdot = Envinfo.rel_pose_xdot / (MAXUAVPOSE_Z*math.tan(camera_fov/2)*2)
    normalized_rel_pose_ydot = Envinfo.rel_pose_ydot / (MAXUAVPOSE_Z*math.tan(camera_fov/2)*2)
    normalized_rel_pose_zdot = Envinfo.rel_pose_zdot / (MAXUAVPOSE_Z-MINUAVPOSE_Z)
    normalized_yaw = Envinfo.yaw / math.pi
    normalized_yaw_speed = Envinfo.yaw_speed / 1
    normalized_mark_recogn = (Envinfo.mark_recogn - 0.5)*2

#    normalized_rel_x = Envinfo.rel_x  / (MAXUAVPOSE_Z*math.tan(camera_fov/2))
#    normalized_rel_y = Envinfo.rel_y  / (MAXUAVPOSE_Z*math.tan(camera_fov/2))
#    normalized_rel_z = Envinfo.rel_z  / MAXUAVPOSE_Z


#    state_ = numpy.array((normalized_vel_x,normalized_vel_y,normalized_vel_z,normalized_rel_x,normalized_rel_y,normalized_rel_z,normalized_yaw))
#    print(state_)
#    state_ = numpy.array((normalized_rel_x,normalized_rel_y,normalized_rel_z,normalized_rel_pose_xdot,normalized_rel_pose_ydot,normalized_yaw))
    state_ = numpy.array((normalized_rel_x,normalized_rel_y,normalized_rel_z,normalized_rel_pose_xdot,normalized_rel_pose_ydot,normalized_rel_pose_zdot,normalized_yaw))#,normalized_yaw_speed))
#    print(state_)

    return state_

def main_loop():

    global Envinfo, update_start, env_action, state, next_state, agent, episode_R, landingstate

    output_file_name = 'vel_result_output.txt'  # record the training result

    rospy.init_node('ddpg_agent', anonymous=True)
 
    # Service server -> tracking and landing
    rospy.Service('exchange_state_action',getEnvinfo,get_Envinfo_callback)

    # initialize
    done = 0
    num_state = 7
    num_action = 3#3
    agent = Agent(num_state, num_action)
    episode_R = 0
    landingstate = 0
    n = 0

    Envinfo = getEnvinfo()
    Envinfo.step = 0
    Envinfo.done = False
    env_action    = numpy.zeros((num_action))
#    env_action.uav_vx = 0    # initial action
#    env_action.uav_vy = 0
#    env_action.uav_vz = 0
#    env_action.thrust = 0
    writed = False
    update_start = False
    r = rospy.Rate(30)  # 20Hz

    while not rospy.is_shutdown():
        if(Envinfo.step != 0 and update_start==True):
            agent.replay() 
            update_start = False
        if Envinfo.done and Envinfo.reset_error == False and Envinfo.step > 1:
            if abs(Envinfo.rel_x<=0.75) and abs(Envinfo.rel_y)<=0.75 and abs(Envinfo.rel_z)<=1:
                landingstate = 1
            else:
                landingstate = 0

            if writed:
                print("-----------------------------")
                print("Total reward:", episode_R)
                with open(output_file_name, 'a') as f:
                    f.write(str(Envinfo.episode+1) + ' episode ' + str(Envinfo.step) + ' Steps' + ' Total_reward: ' + str(episode_R)\
                           + ' landing_state ' + str(landingstate) + '\n')
                numpy.empty_like(state)
                numpy.empty_like(next_state)
                print("episode : ", Envinfo.episode)
                print("total step :", Envinfo.step)
                print("-----------------------------")
                episode_R = 0.0
                writed = False

        elif Envinfo.done and Envinfo.reset_error == False and Envinfo.step == 1:
            episode_R = 0
            writed = True
        else:
            writed = True

if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print ('Interrupted')
        sys.exit(0)  

           
