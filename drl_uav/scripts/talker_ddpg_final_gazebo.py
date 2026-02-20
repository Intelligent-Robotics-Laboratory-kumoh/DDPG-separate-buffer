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
from drl_uav.msg import Num
from drl_uav.msg import Input_Game
from drl_uav.msg import Output_Game

# get UAV status:
from geometry_msgs.msg import PoseStamped	# UAV pos status
from geometry_msgs.msg import TwistStamped	# UAV vel status
from drl_uav.msg import Restart_Finished # UAV restart finished
from drl_uav.msg import AttControlRunning    # UAV att_control running: ready for Memory::observe().
from drl_uav.msg import AttitudeTarget       # UAV att setpoint(thrust is used)

from drl_uav.msg import getEnvinfo,getEnvinfoResponse	         # Get state 
from drl_uav.msg import getRLflag,getRLflagResponse           # push action

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import random, numpy, math, gym
import tensorflow as tf
from collections import deque

from gazebo_msgs.msg import ModelStates

tf.debugging.set_log_device_placement(True)

tf.compat.v1.disable_eager_execution()

import sys

mu = 5e-4

theta = 0.1
sigma = 0.1

class env_action

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
            self.fc1 = tf.compat.v1.layers.dense(self.state, 128, activation=tf.nn.relu)
            #self.fc2 = tf.compat.v1.layers.dense(self.fc1, 32, activation=tf.nn.relu)
            #self.fc3 = tf.compat.v1.layers.dense(self.fc2, 64, activation=tf.nn.relu)
            self.action = tf.compat.v1.layers.dense(self.fc1, action_size, activation=tf.nn.sigmoid)
        self.trainable_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, name)

class Critic:
    def __init__(self, state_size, action_size, name):
        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, state_size])
            self.action = tf.compat.v1.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.state, self.action], axis=-1)
            self.fc1 = tf.compat.v1.layers.dense(self.concat, 128, activation=tf.nn.relu)
            #self.fc2 = tf.compat.v1.layers.dense(self.fc1, 32, activation=tf.nn.relu)
            #self.fc3 = tf.compat.v1.layers.dense(self.fc2, 64, activation=tf.nn.relu)
            self.predict_q = tf.compat.v1.layers.dense(self.fc1, 1, activation=None)
        self.trainable_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, name)

class Agent:
    steps = 0
    def __init__(self, num_state, num_action):
        # parameters of External Environment:
        self.num_state = num_state
        self.num_action = num_action

        # parameters of Internal DRL algorithm:
        ## Memory:
        self.MEMORY_CAPACITY = 500000
        ## RL algorithm:
        self.GAMMA = 0.99
        ## Deep network: 
        self.MEMORY_BATCH_SIZE = 128 # number of data for one training! ?(Maybe we can set MEMORY_BATCH_SIZE = MEMORY_CAPACITY)
        ## Random selection proportion:
        self.MAX_EPSILON = 1
        self.EPSILON_DECAY = 0.999
        self.MIN_EPSILON = 0.001
        self.tau = 5e-3  # 5e-3
        self.actor_lr = 1e-3  # 1e-3
        self.critic_lr = 5e-3  # 5e-3
        #self.LAMBDA = 0.0015  # speed of decay

        self.epsilon = self.MAX_EPSILON

        self.actor_main = Actor(self.num_state, self.num_action, "main_actor")
        self.actor_target = Actor(self.num_state, self.num_action, "target_actor")
        self.critic_main = Critic(self.num_state, self.num_action, "main_critic")
        self.critic_target = Critic(self.num_state, self.num_action, "traget_critic")
        #self.target_model = self.model
        #self.target_update_rate = 10
        #self.initial_update_target()
        #self.memory = Memory(self.MEMORY_CAPACITY)
        #self.Noise = OU_noise()

        self.target_q = tf.compat.v1.placeholder(tf.float32, [None, 1])
        critic_loss = tf.compat.v1.losses.mean_squared_error(self.target_q, self.critic_main.predict_q)
        with tf.control_dependencies(self.critic_main.trainable_var):
            self.train_critic = tf.compat.v1.train.AdamOptimizer(self.critic_lr).minimize(critic_loss)

        action_grad = tf.clip_by_value(tf.gradients(ys=tf.squeeze(self.critic_main.predict_q), xs=self.critic_main.action), -10, 10)
        policy_grad = tf.gradients(ys=self.actor_main.action, xs=self.actor_main.trainable_var, grad_ys=action_grad)
        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads / self.MEMORY_BATCH_SIZE
        with tf.control_dependencies(self.actor_main.trainable_var):
            self.train_actor = tf.compat.v1.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(policy_grad, self.actor_main.trainable_var))

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        #self.memory = Memory(self.MEMORY_CAPACITY)
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

        #print(self.actor_main.trainable_var)


    def act(self, state):   # action:[0,1,2,...,num_action-1]

        # Limit: 3) forced input in Emergency: Vz is out of [-3,3].
        global UAV_Vel

        action = self.sess.run(self.actor_main.action, feed_dict={self.actor_main.state: [state]})
        noise = self.Noise.sample()
        action = action + (self.epsilon * noise)

        if action > 1:
            action = 1
        if action < 0:
            action = 0
        return action

    def observe(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        # decrease Epsilon to reduce random action and trust more in greedy algorithm
        self.steps += 1
        #self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.steps)
        #if self.epsilon > self.MIN_EPSILON:
        #    self.epsilon *= self.EPSILON_DECAY

    def replay(self):   # get knowledge from experience!
        if len(self.memory) > 1000:
            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
            mini_batch = random.sample(self.memory, self.MEMORY_BATCH_SIZE)
            # batch = self.memory.sample(self.memory.num_experience())  # the training data size is too big!

            states = numpy.asarray([sample[0] for sample in mini_batch])
            actions = numpy.asarray([[sample[1]] for sample in mini_batch])
            rewards = numpy.asarray([sample[2] for sample in mini_batch])
            next_states = numpy.asarray([sample[3] for sample in mini_batch])
            dones = numpy.asarray([sample[4] for sample in mini_batch])
           
            actor_target_actions = self.sess.run(self.actor_target.action, feed_dict={self.actor_target.state: next_states})
            critic_target_predict_qs = self.sess.run(self.critic_target.predict_q, feed_dict={self.critic_target.state: next_states, self.critic_target.action: actor_target_actions})

            target_qs = numpy.asarray([reward + self.GAMMA * (1 - done) * critic_target_predict_q for reward, critic_target_predict_q, done in zip(rewards, critic_target_predict_qs, dones)])

            self.sess.run(self.train_critic, feed_dict={self.critic_main.state: states, self.critic_main.action: actions, self.target_q: target_qs})
            actions_for_train = self.sess.run(self.actor_main.action, feed_dict={self.actor_main.state: states})
            self.sess.run(self.train_actor, feed_dict={self.actor_main.state: states, self.critic_main.state: states, self.critic_main.action: actions_for_train})
            self.sess.run(self.soft_update_target)

# get UAV status:
'''UAV_Vel = TwistStamped()
UAV_Pos = PoseStamped()
att_running = AttControlRunning()
UAV_Att_Setpoint = AttitudeTarget()
gziris = ModelStates()'''



def UAV_pos_callback(data):
    global UAV_Pos  
    # Subscribing:
    # rospy.loginfo('Receive UAV Pos Status: %f %f %f', data.pose.position.x, data.pose.position.y, data.pose.position.z)
    UAV_Pos = data

def UAV_vel_callback(data):
    global UAV_Vel
    # Subscribing:
    # rospy.loginfo('Receive UAV Vel Status: %f %f %f', data.twist.linear.x, data.twist.linear.y, data.twist.linear.z)  
    UAV_Vel = data

def restart_finished_callback(data):
    # Subscribing:
    # rospy.loginfo('UAV restart finished: %d', data.finished)
    pass

def get_gzmodel_callback(data):
    global gziris
    gziris = data
    #print("%f",gziris.pose[1].position.z)

def att_running_callback(data):
    global att_running
    # Subscribing:
    # rospy.loginfo('UAV att running!: %d', data.running)
    att_running = data
    # rospy.loginfo('att_running!:  ~~~~~~~~~~~~~~~~~~~~~~ %d ~~~~~~~~~~~~~~~~~~~~', att_running.running)

def local_attitude_setpoint__callback(data):
    global UAV_Att_Setpoint
    # Subscribing:
    # rospy.loginfo('thrust: %f', data.thrust)
    UAV_Att_Setpoint = data

def get_Envinfo_callback(data):
    global Envinfo
    Envinfo.UAV_vx = data.uav_vx
    Envinfo.UAV_vy = data.uav_vy
    Envinfo.UAV_vz = data.uav_vz
    Envinfo.rel_x  = data.rel_x
    Envinfo.rel_y  = data.rel_y
    Envinfo.rel_z  = data.rel_z
    Envinfo.episode = data.episode
    Envinfo.steps   = data.step
    Envinfo.reward  = data.reward
    Envinfo.markrecogn = data.markrecogn

    return getEnvinfoResponse
    (
        uav_vx  = env_action.vx
        uav_vy  = env_action.vy
        uav_vz  = env_action.vz
        landing = env_action.landing
    )

def get_RLflag_callback(data)
    global done
    done = ~data.flag
    return getRLflagResponse(True)

# Publisher:
'''
pub = rospy.Publisher('game_input', Input_Game, queue_size=10)
env_input = Input_Game()
env_input.action = 0.5    # initial action
'''
env_action    = getEnvinfoResponse()
env_action.vx = 0    # initial action
env_action.vy = 0
env_action.vz = 0
env_action.landing = 0

current_status = Output_Game()

def status_update(data):
    # Subscribing:
    # rospy.loginfo(rospy.get_caller_id() + 'Receive Game Status: %f %f %f %f',
    # data.vel1, data.vel2, data.pos1, data.pos2)
    # rospy.loginfo('Receive Game Status: %f %f %f %f', data.vel1, data.vel2, data.pos1, data.pos2)
    global current_status
    current_status = data
    # rospy.loginfo('Receive Game Status: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %d', data.num)
    

   
def interact():
    # # current_status.pos1:[2, 10] => failed = False
    # # current_status.pos:[5.7, 6.3] => done = True
    # global current_status
    # # 1) get pre_status = current_status
    # pre_status = current_status
    # # rospy.loginfo('pre_status.pos1 = %f', pre_status.pos1)
    # # 2) publish action
    # # rospy.loginfo('Publishing action: %f', env_input.action)
    # pub.publish(env_input)
    # # 3) judge from current_status: calculate: r, done, failed
    # # 4) return current_status, reward, done, failed(NOT Used!)
    # state_ = numpy.array(current_status.pos1)
    # if (current_status.pos1 > 10.0 or current_status.pos1 < 2.0):
    #     done = True
    #     return state_, -0.5, done, True
    # # reward = 10.0 / (numpy.square(current_status.pos1 - 6.0) + 1.0)
    # done = False
    # reward = 0.0
    # if (math.fabs(current_status.pos1 - 6.0) < 0.3):
    #     reward = 1.0
    # return state_, reward, done, False

    # publish env_input(action):
    global pub, env_input
    # get UAV status:
    global UAV_Vel, UAV_Pos

    # 1) publish action
    # rospy.loginfo('Publishing action: %f', env_input.action)
    pub.publish(env_action)


    # 2) judge from current_status: calculate: r, done, failed
    # 3) return current_status, reward, done, failed(NOT Used!)
    normalized_pos_z = (gziris.pose[1].position.z - 20.0) / 10.0      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
    normalized_vel_z = UAV_Vel.twist.linear.z / 3.0                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
    normalized_thrust = (UAV_Att_Setpoint.thrust - 0.59) / 0.19     # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]
    state_ = numpy.array((normalized_pos_z, normalized_vel_z, normalized_thrust))
    
    done = False
    #reward = 0.0

    if((gziris.pose[1].position.z > 25.0) or (gziris.pose[1].position.z < 15.0)): # allowed trial height:[8m,30m], release_height=restart_height=15m
        done = True                                               # Restart the env:
        #rospy.loginfo("Let's restart!")
   
    #if (math.fabs(UAV_Pos.pose.position.z - 20.0) < 0.3):
    #    reward = 1.0

    if (math.fabs(gziris.pose[1].position.z - 20.0) < 3.0):
        reward = 1.0
    else:
        reward = 0.0
    return state_, reward, done, True

def env_restore():
    # 1) publish pos destination: [0,0,3]
    # 2) judge if pos arrived?
    # 3) hover for 1 second -> break!
    # sleep for 1 seconds
    rospy.sleep(1.)


def main_loop():

    global current_status, pub, env_input

    # get UAV status:
    global UAV_Vel, UAV_Pos, att_running, UAV_Att_Setpoint, gziris

    rospy.init_node('custom_talker', anonymous=True)

    # 1) get current status:
    # Subscriber:
    rospy.Subscriber('game_status', Output_Game, status_update) 
    # rospy.loginfo('current_status: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)

    # Subscriber:
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, UAV_pos_callback)
    # Subscriber:
    rospy.Subscriber('mavros/local_position/velocity', TwistStamped, UAV_vel_callback)

    # Subscriber:
    rospy.Subscriber('restart_finished_msg', Restart_Finished, restart_finished_callback)

    # Subscriber:
    rospy.Subscriber('att_running_msg', AttControlRunning, att_running_callback)

    # Subscriber:
    rospy.Subscriber('/mavros/setpoint_raw/attitude', AttitudeTarget, local_attitude_setpoint__callback)

    rospy.Subscriber('/gazebo/model_states', ModelStates, get_gzmodel_callback)

    # Service server -> tracking and landing
    rospy.Service('exchange_state_action',getEnvinfo,get_Envinfo_callback)
    rospy.Service('RLflag',getRLflag,get_RLflag_callback)

    # 2) take action
    num_state = 3   # state=[UAV_height, UAV_vertical_vel, , UAV_Att_Setpoint.thrust]
    num_action = 1  # action=[0,1]
    agent = Agent(num_state, num_action)
    R = 0
    n = 0
    #model_saved = 0
    num_trial = 0   # the number of current trial
    new_trial = False

    r = rospy.Rate(20)  # 20Hz

    # get states:
    normalized_pos_z = (gziris.pose[1].position.z - 20.0) / 10.0      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
    normalized_vel_z = UAV_Vel.twist.linear.z / 3.0                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
    normalized_thrust = (UAV_Att_Setpoint.thrust - 0.59) / 0.19     # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]
    state = numpy.array((normalized_pos_z, normalized_vel_z, normalized_thrust))
    # take action:
    env_input.action = agent.act(state)
    #print(env_input.action)

    output_file_name = 'result_output.txt'  # record the training result


    while not rospy.is_shutdown():
        # rospy.loginfo('UAV Vel Status: %f %f %f', UAV_Vel.twist.linear.x, UAV_Vel.twist.linear.y, UAV_Vel.twist.linear.z)  
        # rospy.loginfo('UAV Pos Status: %f %f %f', UAV_Pos.pose.position.x, UAV_Pos.pose.position.y, UAV_Pos.pose.position.z)
        # rospy.loginfo('main loop: att_running.running: %d', att_running.running)

        #print("att_running.running: %d", att_running.running)
        
        if(att_running.running):

            
            n += 1

            state_, reward, done, failed = interact()
            if done:
                state_ = None
                # record the memory:
                #rospy.loginfo('Memory: state(Pos, Vel, thrust): %f, %f, %f  action: %f  reward: %f state_: %f, %f, %f', state[0], state[1], state[2], env_input.action, reward, 0.0, 0.0, 0.0)      
                if (n > 10):                                  
                    rospy.loginfo('%d th trial: n: %d current state(Pos, Vel, thrust): %f, %f, %f  current action: %f  current reward: %f Total reward: %f', num_trial, n, gziris.pose[1].position.z, UAV_Vel.twist.linear.z, UAV_Att_Setpoint.thrust, env_input.action, reward, R)
            else:
                # record the memory:
                #rospy.loginfo('Memory: state(Pos, Vel, thrust): %f, %f, %f  action: %f  reward: %f state_: %f, %f, %f', state[0], state[1], state[2], env_input.action, reward, state_[0], state_[1], state_[2])

                # ignore final experience(state_ = None) for action = -1 then, and will lead to no value of v_[action] in RL.
                try:
                    if (env_input.action != -1):
                        agent.observe(state, env_input.action, reward, state_, done)
                        agent.replay()                                
                except KeyboardInterrupt:
                    print('Interrupted')
                    # break
                    sys.exit(0)



            # agent.observe((state, env_input.action, reward, state_))
            # agent.replay()

            R += reward
            # rospy.loginfo('current action: %f', env_input.action)
            # rospy.loginfo('current reward: %f', reward)

            # # display the result in star figure:
            # state_scale = int(state*5.0)
            # # rospy.loginfo('state = %f', state)
            # for i in range(50):
            #     if(i == state_scale):
            #         print 'x',
            #     else:
            #         print '_',
            # print('Total reward:', R)

            # prepare for the next loop:
            # get states:
            normalized_pos_z = (gziris.pose[1].position.z - 20.0) / 10.0      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
            normalized_vel_z = UAV_Vel.twist.linear.z / 3.0                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
            normalized_thrust = (UAV_Att_Setpoint.thrust - 0.59) / 0.19     # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]
            state = numpy.array((normalized_pos_z, normalized_vel_z, normalized_thrust))
            # take action:
            env_input.action = agent.act(state)
            #print(env_input.action)

            if(n > 999):
                rospy.loginfo('%d th trial: n: %d current state(Pos, Vel, thrust): %f, %f, %f  current action: %f  current reward: %f Total reward: %f', num_trial, n, gziris.pose[1].position.z, UAV_Vel.twist.linear.z, UAV_Att_Setpoint.thrust, env_input.action, reward, R)
                done = True
                n = 0

            if done:    # restart(these code may run several times because of the time delay)!
                env_input.action = -1.0    # Restart the game!
                pub.publish(env_input)
                #new_trial = True
                #rospy.loginfo('Restarting!')

            if((new_trial == True) and done):       # to make sure this loop runs only once!
                num_trial += 1
                new_trial = False

                # record the trial result:  # stored in $HOME folder!
                with open(output_file_name, 'a') as f:
                    f.write(str(num_trial) + 'th_trial: ' + str(n) + ' Steps' + ' Total_reward: ' + str(R) + '\n')

                rospy.sleep(0.1)

                # save model:

                n = 0
                R = 0.0
            
            #rospy.loginfo('%d th trial: n: %d current state(Pos, Vel, thrust): %f, %f, %f  current action: %f  current reward: %f Total reward: %f', num_trial, n, UAV_Pos.pose.position.z, UAV_Vel.twist.linear.z, UAV_Att_Setpoint.thrust, env_input.action, reward, R)
            
        else:   # restarting!
            
            new_trial = True
            # publish random action(0/1) to stop Env-restart(-1) commander!
            env_input.action =  random.randint(0, 1)    # Restart the game!
            # rospy.loginfo('Random action: %f', env_input.action)
            pub.publish(env_input)

        r.sleep()

        # sleep for 10 seconds
        # rospy.sleep(10.)

  
    #agent.actor_main.model.save("DRL_UAV_latest.h5")
    print("Running: Total reward:", R)
            

if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print ('Interrupted')
        sys.exit(0)  

# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt:
#         print 'Interrupted'
#         sys.exit(0)        sample(self.memory, self.MEMORY_BATCH_SIZE)
           
