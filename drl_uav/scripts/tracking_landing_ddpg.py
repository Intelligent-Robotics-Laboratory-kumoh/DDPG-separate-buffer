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


import rospy

from drl_uav.srv import getEnvinfo, getEnvinfoResponse	         # Get state 
from drl_uav.srv import getEnvinfo_att, getEnvinfo_attResponse

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import random, numpy, math, gym, time
import tensorflow as tf

from collections import deque

tf.debugging.set_log_device_placement(True)

tf.compat.v1.disable_eager_execution()

import sys

mu = 5e-4

theta = 0.1
sigma = 0.1

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
            self.fc1 = tf.compat.v1.layers.dense(self.state, 600, activation=tf.nn.elu)
            self.bn1 = tf.nn.elu(tf.nn.batch_normalization(self.fc1,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc2 = tf.compat.v1.layers.dense(self.bn1, 500, activation=tf.nn.elu)
            self.bn2 = tf.nn.elu(tf.nn.batch_normalization(self.fc2,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc3 = tf.compat.v1.layers.dense(self.fc2, 400, activation=tf.nn.elu)
            self.bn3 = tf.nn.elu(tf.nn.batch_normalization(self.fc3,normalize_mean,normalize_val,shift,scale,val_epsilon))
#            self.fc4 = tf.compat.v1.layers.dense(self.fc3, 64, activation=tf.nn.relu)
            self.action = tf.compat.v1.layers.dense(self.bn3, action_size, activation=tf.nn.tanh)
        self.trainable_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, name)

class Critic:
    def __init__(self, state_size, action_size, name):
        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, state_size])
            self.action = tf.compat.v1.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.state, self.action], axis=-1)
            self.fc1 = tf.compat.v1.layers.dense(self.concat, 600, activation=tf.nn.elu)
            self.bn1 = tf.nn.elu(tf.nn.batch_normalization(self.fc1,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc2 = tf.compat.v1.layers.dense(self.bn1, 500, activation=tf.nn.elu)
            self.bn2 = tf.nn.elu(tf.nn.batch_normalization(self.fc2,normalize_mean,normalize_val,shift,scale,val_epsilon))
            self.fc3 = tf.compat.v1.layers.dense(self.bn2, 400, activation=tf.nn.elu)
            self.bn3 = tf.nn.elu(tf.nn.batch_normalization(self.fc3,normalize_mean,normalize_val,shift,scale,val_epsilon))
 #           self.fc4 = tf.compat.v1.layers.dense(self.fc3, 64, activation=tf.nn.relu)
            self.predict_q = tf.compat.v1.layers.dense(self.fc3, 1, activation=None)
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
        ## RL algorithm:
        self.GAMMA = 0.99
        ## Deep network: 
        self.MEMORY_BATCH_SIZE = 128 # number of data for one training! ?(Maybe we can set MEMORY_BATCH_SIZE = MEMORY_CAPACITY)
        ## Random selection proportion:
        self.MAX_EPSILON = 1
        self.EPSILON_DECAY = 0.999
        self.MIN_EPSILON = 0.001
#        self.tau = 0.001  # 5e-3
#        self.actor_lr = 1e-2  # 1e-3
#        self.critic_lr = 1e-2  # 5e-3
        self.tau = 1e-3#0.001  # 5e-3
        self.actor_lr = 5e-6  # 1e-3
        self.critic_lr = 5e-5  # 5e-3
        #self.LAMBDA = 0.0015  # speed of decay

        self.epsilon = self.MAX_EPSILON

        self.actor_main = Actor(self.num_state, self.num_action, "main_actor")
        self.actor_target = Actor(self.num_state, self.num_action, "target_actor")
        self.critic_main = Critic(self.num_state, self.num_action, "main_critic")
        self.critic_target = Critic(self.num_state, self.num_action, "traget_critic")

        self.target_q = tf.compat.v1.placeholder(tf.float32, [None, 1])

        critic_loss = tf.compat.v1.losses.mean_squared_error(self.target_q, self.critic_main.predict_q)
        with tf.control_dependencies(self.critic_main.trainable_var):
            self.train_critic = tf.compat.v1.train.AdamOptimizer(self.critic_lr).minimize(critic_loss)
    # take first action:
        #env_action = agent.act(state)
    #print(env_input.action)amOptimizer(self.critic_lr).minimize(critic_loss)

        action_grad = tf.clip_by_value(tf.gradients(ys = tf.squeeze(self.critic_main.predict_q), xs = self.critic_main.action), -10, 10)

        policy_grad = tf.gradients(ys=self.actor_main.action, xs=self.actor_main.trainable_var, grad_ys=action_grad)
        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads / self.MEMORY_BATCH_SIZE
        with tf.control_dependencies(self.actor_main.trainable_var):
            self.train_actor = tf.compat.v1.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(policy_grad, self.actor_main.trainable_var))

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.Noise = OU_noise()
        self.memory = deque(maxlen=self.MEMORY_CAPACITY)

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
        #print(action)
        action_ = numpy.zeros((4))

        noise = self.Noise.sample()

        for i in range(4):
            action_[i] = action[0][i] + (self.epsilon * noise)
            if(action_[i] > 1):
                action_[i] = 1
            elif(action_[i]<-1):
                action_[i] = -1
        print(action_)
        return action_

    def observe(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return len(self.memory)
        # decrease Epsilon to reduce random action and trust more in greedy algorithm
        #self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.steps)
        #if self.epsilon > self.MIN_EPSILON:
        #    self.epsilon *= self.EPSILON_DECAY

    def replay(self):   # get knowledge from experience!
        
        if len(self.memory) > 5000:
            print("RL-ing")
            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
            mini_batch = random.sample(self.memory, self.MEMORY_BATCH_SIZE)
            # batch = self.memory.sample(self.memory.num_experience())  # the training data size is too big!
            #start = time.time()
            states = numpy.asarray([sample[0] for sample in mini_batch])
            actions = numpy.asarray([sample[1] for sample in mini_batch])
            rewards = numpy.asarray([sample[2] for sample in mini_batch])
            next_states = numpy.asarray([sample[3] for sample in mini_batch])
            dones = numpy.asarray([sample[4] for sample in mini_batch])

            states = self.normalize(states)
            actions = self.normalize(actions)
            rewards = self.normalize(rewards)
            next_states = self.normalize(next_states)
            
            #print("SAMPLING time : ",time.time()-start)
            #start = time.time()
            actor_target_actions = self.sess.run(self.actor_target.action, feed_dict={self.actor_target.state: next_states})
            critic_target_predict_qs = self.sess.run(self.critic_target.predict_q, feed_dict={self.critic_target.state: next_states, self.critic_target.action: actor_target_actions})
            str_done = str(dones)
#            target_qs = numpy.zeros((64,1))

#            for i in range(self.MEMORY_BATCH_SIZE):
#                if str_done[i]==True:
#                    target_qs[i] = rewards[i]
#                else:
#                    target_qs[i] = rewards[i] + self.GAMMA * critic_target_predict_qs[i]

            target_qs = numpy.asarray([reward + self.GAMMA * (1 - done) * critic_target_predict_qs for reward, critic_target_predict_qs, done in zip(rewards, critic_target_predict_qs, dones)])

            self.sess.run(self.train_critic, feed_dict={self.critic_main.state: states, self.critic_main.action: actions, self.target_q: target_qs})
            actions_for_train = self.sess.run(self.actor_main.action, feed_dict={self.actor_main.state: states})
            self.sess.run(self.train_actor, feed_dict={self.actor_main.state: states, self.critic_main.state: states, self.critic_main.action: actions_for_train})
            self.sess.run(self.soft_update_target)
            #print("UPDATE time : ",time.time()-start)

    def normalize(self,value_batch):
        batch_ep = 1e-5
        batch_mean = numpy.sum(value_batch)/len(value_batch)
        batch_val  = numpy.sum((value_batch-batch_mean)**2)/len(value_batch)
        normalized_value = (value_batch-batch_mean)/math.sqrt(batch_val+batch_ep)
        return normalized_value

global episode_R

def get_Envinfo_att_callback(data):
    global Envinfo, env_action, agent, state, next_state, episode_R, done

    Envinfo.rel_pose_xdot     = data.rel_pose_xdot
    Envinfo.rel_pose_ydot     = data.rel_pose_ydot
    Envinfo.rel_x      = data.rel_x
    Envinfo.rel_y      = data.rel_y
    Envinfo.rel_z      = data.rel_z
    Envinfo.episode    = data.episode
    Envinfo.step       = data.step
    Envinfo.reward     = data.reward
    Envinfo.yaw        = data.yaw
    Envinfo.done       = data.done

    episode_R += Envinfo.reward

    current_memory_size = 0 

    if(Envinfo.step != 0):
        next_state = interact()
        current_memory_size = agent.observe(state, env_action, Envinfo.reward, next_state, Envinfo.done)
        state = next_state
    else:
        state = interact()

    if(current_memory_size > 5000):
        print("TRAINING")
        env_action = agent.act(state)
    else:
        print("RANDOM ACTION")
        for i in range(4):
            env_action[i] = random.uniform(-1.0,1.0)  
    env_action = agent.act(state) 


    return getEnvinfo_attResponse(roll = env_action[0], pitch = env_action[1], yaw_speed = env_action[2], thrust = env_action[3])


def interact():

    # publish env_input(action):
    global Envinfo, done

    MAXUAVVEL_XY = 12
    MAXUAVVEL_Z  = 3
    MAXUAVPOSE_Z = 14

    camera_fov   = 80*math.pi/180
    image_size   = 400

    normalized_rel_x = Envinfo.rel_x  / 10
    normalized_rel_y = Envinfo.rel_y  / 10
    normalized_rel_z = (Envinfo.rel_z-7.5)  / 15
    normalized_rel_pose_xdot = Envinfo.rel_pose_xdot / 20
    normalized_rel_pose_ydot = Envinfo.rel_pose_ydot / 20

    normalized_yaw = Envinfo.yaw / math.pi

#    state_ = numpy.array((normalized_vel_x,normalized_vel_y,normalized_vel_z,normalized_rel_x,normalized_rel_y,normalized_rel_z,normalized_yaw))
#    print(state_)
    state_ = numpy.array((normalized_rel_x,normalized_rel_y,normalized_rel_z,normalized_rel_pose_xdot,normalized_rel_pose_ydot,normalized_yaw))

    return state_

def env_restore():
    # 1) publish pos destination: [0,0,3]
    # 2) judge if pos arrived?
    # 3) hover for 1 second -> break!
    # sleep for 1 seconds
    rospy.sleep(1.)


def main_loop():

    global Envinfo, done, env_action, state, next_state, agent, episode_R

    output_file_name = 'result_output.txt'  # record the training result

    rospy.init_node('ddpg_agent', anonymous=True)
 
    # Service server -> tracking and landing
    rospy.Service('exchange_state_action',getEnvinfo_att,get_Envinfo_att_callback)

    # initialize
    done = 0
    num_state = 6#8
    num_action = 4#3
    agent = Agent(num_state, num_action)
    episode_R = 0
    n = 0

    Envinfo = getEnvinfo_att()
    Envinfo.step = 0
    Envinfo.done = False
    env_action    = numpy.zeros((4))
    writed = True
#    env_action.roll = 0    # initial action
#    env_action.pitch = 0
#    env_action.yaw_speed = 0
#    env_action.thrust = 0
    r = rospy.Rate(30)  # 20Hz

    while not rospy.is_shutdown():
        if(Envinfo.step != 0):
            agent.replay()
        if Envinfo.done:
            if writed:
                print("-----------------------------")
                print("Total reward:", episode_R)
                with open(output_file_name, 'a') as f:
                    f.write(str(Envinfo.episode+1) + ' episode ' + str(Envinfo.step) + ' Steps' + ' Total_reward: ' + str(episode_R) + '\n')
                numpy.empty_like(state)
                numpy.empty_like(next_state)
                print("episode : ", Envinfo.episode)
                print("total step :", Envinfo.step)
                episode_R = 0.0
                writed = False
        else:
            writed = True;
 
        r.sleep()

if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print ('Interrupted')
        sys.exit(0)  

           
