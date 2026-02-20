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

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import random, numpy, math, gym, time
import tensorflow as tf
from collections import deque
import copy
from drl_uav import replay_msg
from drl_uav import critic_msg
from drl_uav import agent_msg

import sys

normalize_mean = 0
normalize_val  = 1
scale  = 1
shift  = 0
val_epsilon = 1e-5

class Critic:
    def __init__(self,state_size,action_size,critic_lr,tau,states,actions):

        self.num_state  = state_size
        self.num_action = action_size
        self.critic_lr = critic_lr
        self.tau       = tau

        self.critic_main, self.critic_target = self.create_network()
        self.critic_target.set_weights(self.critic_main.set_weights)
        self.critic_opti = tf.keras.optimizers.Adam(self.critic_lr)

    def create_network(self):
        '''
        layer1_size = 400
        layer2_size = 300
        layer3_size = 200

        state = [None,self.num_state]
        action =[None,self.num_action]
        critic_input_layer = tf.concat([state,action],axis=-1)
        input_layer_size = [None,self.num_state+self.num_action]
        output_layer_size = 1

        W1 = self.variable([input_layer_size,layer1_size],input_layer_size)
        b1 = self.variable([layer1_size],input_layer_size)
        W2 = self.variable([layer1_size,layer2_size],layer1_size)
        b2 = self.variable([layer2_size],layer1_size)
        W3 = self.variable([layer2_size,layer3_size],layer2_size)
        b3 = self.variable([layer3_size],layer2_size)
        W4 = tf.Variable(tf.random_uniform([layer3_size,output_layer_size],-3e-3,3e-3))
        b4 = tf.Variable(tf.random_uniform([output_layer_size],-3e-3,3e-3))

        @tf.function
        def network(x,w,b):
            return w*x+b

        critic_input_layer = tf.nn.batch_normalization(critic_input_layer,normalize_mean,normalize_val,shift,scale,val_epsilon)
        layer1 = network(critic_input_layer,W1,b1)
        bn1 = tf.nn.relu(tf.nn.batch_normalization(layer1,normalize_mean,normalize_val,shift,scale,val_epsilon))
        layer2 = network(bn1,W2,b2)
        bn2 = tf.nn.relu(tf.nn.batch_normalization(layer2,normalize_mean,normalize_val,shift,scale,val_epsilon))        
        layer3 = network(bn2,W3,b3)
        bn3 = tf.nn.relu(tf.nn.batch_normalization(layer3,normalize_mean,normalize_val,shift,scale,val_epsilon))        
        output = network(bn3,W4,b4)

        model = tf.keras.Model(critic_input_layer,output)
        '''

        state_input = tf.keras.layers.Input(shape=(self.num_state))
        state_out = tf.keras.layers.BatchNormalization()(state_input)
        state_out = tf.keras.layers.Dense(16, activation="relu")(state_out)
        state_out = tf.keras.layers.BatchNormalization()(state_input)
        state_out = tf.keras.layers.Dense(32, activation="relu")(state_out)
        state_out = tf.keras.layers.BatchNormalization()(state_input)

        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_action))
        action_out = tf.keras.layers.BatchNormalization()(action_input)
        action_out = tf.keras.layers.Dense(32, activation="relu")(action_out)
        action_out = tf.keras.layers.BatchNormalization()(action_out)

        # Both are passed through seperate layer before concatenating
        concat = tf.keras.layers.Concatenate()([state_out, action_out])
        out = tf.keras.layers.BatchNormalization()(concat)
        out = tf.keras.layers.Dense(400, activation="relu")(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Dense(300, activation="relu")(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Dense(200, activation="relu")(out)
        out = tf.keras.layers.BatchNormalization()(out)
        outputs = tf.keras.layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def ciritic_update(self,states,actions,next_states,rewards,dones):
        
        GAMMA = 0.999
        with tf.GradientTape as tape:
            critic_target = self.critic_target([next_states, actions],training = True)
            target_value_batch = rewards + (1-dones) * GAMMA * critic_target
            critic_loss = tf.math.reduce_mean(tf.math.square(target_value_batch - self.critic_main([states,actions],training=True)))
        critic_grad = tape.gradient(critic_loss, self.critic_main.trainable_variables)
        self.critic_opti.apply_gradient(critic_grad,zip(self.critic_main.variables))
        self.soft_update(self.critic_main.variables,critic_target.variables,self.tau)

        self.pub_critic_batch(states,actions,next_states,rewards,dones,tf.squeeze(critic_grad))

        self.critic_target = critic_target

    def soft_update(self,true_weight,target_weight,tau):
        return target_weight.assign(true_weight * tau + target_weight * (1-tau))

    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt))

    def get_agent_batch(self,data):
        states = data.state
        states = numpy.reshape(states,(self.num_state,64))
        actions = data.action
        actions = numpy.reshape(actions,(self.num_action,64))
        rewards = data.reward
        next_states = data.next_state
        next_states = numpy.reshape(next_states,(self.num_state,64))
        dones  = data.done

        self.ciritic_update(states,actions,rewards,next_states,dones)

    def pub_critic_batch(self,states,actions,next_states,rewards,dones,q_value):
        critic_msg.states = numpy.reshape(states,(1,self.num_state*64))
        critic_msg.actions = numpy.reshape(actions,(1,self.num_action*64))
        critic_msg.next_states = numpy.reshape(next_states,(1,self.num_state*64))
        critic_msg.reward = rewards
        critic_msg.done = dones
        critic_msg.q_value = q_value
        self.critic_pub.publish(critic_msg)

def main_loop():

    rospy.init_node('ddpg_critic', anonymous=True)
    Critic.ciritic_pub = rospy.Publisher('critic',critic_msg,queue_size=10)
    rospy.Subscriber('agent', agent_msg, Critic.get_agent_batch)

    num_state  = rospy.get_param("state_num")
    num_action = rospy.get_param("action_num")
    critic_lr = rospy.get_param("critic_learning_rate")
    tau       = rospy.get_param("tau")

    Critic(num_state,num_action,critic_lr,tau)

    r = rospy.Rate(30)  # 20Hz

    while not rospy.is_shutdown():
        r.sleep()

if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print ('Interrupted')
        sys.exit(0)
           
