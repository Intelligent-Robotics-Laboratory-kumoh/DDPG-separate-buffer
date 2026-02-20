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
from drl_uav import status_save

import sys

class Agent:
    def __init__(self,num_state,num_action,actor_lr,tau,mu,theta,sigma):
        self.num_state = num_state
        self.num_action = num_action
        self.actor_lr = actor_lr
        self.tau = tau

        self.main_actor, self.target_actor = self.actor_create_network()
        self.target_actor.set_weights(self.main_actor.set_weights)
        self.actor_opti = tf.keras.optimizers.Adam(self.actor_lr)

        self.state,self.next_state = numpy.zeros((6))
        self.done = False
        self.episode_R = 0
        self.episode = 0
        self.step = 0

        action_pub = rospy.Publisher('agent',agent_msg,queue_size=10)
        status_save_pub = rospy.Publisher("RL_status_save",status_save,queue_size=10)
        rospy.Subscriber('minibatch', replay_msg, self.get_replay_batch)
        rospy.Subscriber('critic', critic_msg, self.get_critic_batch)
        
    def actor_create_network(self):
    
        state_input = tf.keras.layers.Input(shape=(self.num_state,))
        out = tf.keras.layers.BatchNormalization()(state_input)
        out = tf.keras.layers.Dense(400, activation="relu")(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Dense(300, activation="relu")(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Dense(200, activation="relu")(out)
        out = tf.keras.layers.BatchNormalization()(out)
        outputs = tf.keras.layers.Dense(self.num_action, activation="tanh")(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input], outputs)

        return model

    def make_action(self,states):
        noise = OU_noise(mu,theta,sigma)
        action = tf.squeeze(self.main_actor([states]) + noise.sample)
        action = tf.clip_by_value(action,-1,1)
        return action

    def actor_update(self,q_value):
        with tf.GradientTape() as tape:
            actor_loss = -tf.math.reduce_mean(q_value)
        actor_grad = tape.gradient(actor_loss, self.main_actor.trainable_variables)
        self.actor_opti.apply_gradients(zip(actor_grad, self.main_actor.trainable_variables))    
        self.soft_update(self.main_actor.variables,self.target_actor.variables,self.tau)

    def soft_update(self,true_weight,target_weight,tau):
        return target_weight.assign(true_weight * tau + target_weight * (1-tau))

    def RL_sna_change_callback(self,data):
        start_time=time.time()
        rel_x    = data.rel_x
        rel_y     = data.rel_y
        rel_z     = data.rel_z
        rel_xdot     = data.rel_pose_xdot
        rel_ydot     = data.rel_pose_ydot
        self.episode    = data.episode
        self.step       = data.step
        reward     = data.reward
        yaw        = data.yaw

        self.episode_R += reward

        if(step != 0):
            self.next_state = self.mapping(rel_x,rel_y,rel_z,rel_xdot,rel_ydot)
            self.state = self.next_state
        else:
            self.state = self.mapping(rel_x,rel_y,rel_z,rel_xdot,rel_ydot)

        action = self.make_action(self.state)

        if(step != 0):
            self.pub_env_info(self.state,action,self.next_state,reward,self.done)

        return getEnvinfoResponse(uav_vx = 3 * action[0], uav_vy = 3 * action[1] , uav_vz = 1 * action[2])

    def mapping(self,rel_x,rel_y,rel_z,rel_xdot,rel_ydot,yaw):
        rel_x = rel_x/10
        rel_y = rel_y/10
        rel_z = (rel_z-7.5)/7.5
        rel_xdot = rel_xdot/20
        rel_ydot = rel_ydot/20
        yaw = yaw/math.pi

        state = numpy.array(rel_x,rel_y,rel_z,rel_xdot,rel_ydot,yaw)
        return state
    
    def pub_env_info(self,state,action,next_state,reward,done):
        status_save.state = state
        status_save.action = action
        status_save.next_state = next_state
        status_save.reward = reward
        status_save.done = done

        self.status_save_pub.publish(status_save)

    def get_replay_batch(self,data):
        agent_msg.states = numpy.reshape(data.state,(self.num_state,64))
        agent_msg.actions = data.action
        agent_msg.rewards = data.reward
        agent_msg.next_states = data.next_state
        agent_msg.dones  = data.done
        agent_msg.next_action = self.make_action(states)
        agent_msg.states = numpy.reshape(agent_msg.states,(1,self.num_state*64))
        self.action_pub.publish(agent_msg)

    def get_critic_batch(self,data):
        states = data.state
        actions = data.action
        rewards = data.rewards
        next_states = data.next_state
        dones = data.done
        q_value = data.q_value

        self.actor_update(q_value)

class OU_noise:
    def __init__(self,mu,theta,sigma):
        self.reset()

    def reset(self):
        self.X = numpy.ones(1) * mu

    def sample(self):
        dx = theta * (mu - self.X)
        dx += sigma * numpy.random.randn(len(self.X))
        self.X += dx
        return self.X


def main_loop():
    rospy.init_node('ddpg_agent', anonymous=True)

    output_file_name = 'result_output.txt'

    num_state  = rospy.get_param("state_num")
    num_action = rospy.get_param("action_num")
    actor_lr = rospy.get_param("critic_learning_rate")
    tau       = rospy.get_param("tau")
    mu = rospy.get_param("mu")
    theta = rospy.get_param("theta")
    sigma = rospy.get_param("sigma")

    agent = Agent(num_state,num_action,actor_lr,tau,mu,theta,sigma)

    r = rospy.Rate(30)  # 20Hz

    while not rospy.is_shutdown():
        if agent.done:
            print("-----------------------------")
            print("Total reward:", episode_R)
            with open(output_file_name, 'a') as f:
                f.write(str(agent.episode) + ' episode ' + str(agent.step) + ' Steps' + ' Total_reward: ' + str(agent.episode_R) + '\n')
            numpy.empty_like(agent.state)
            numpy.empty_like(agent.next_state)
            print("episode : ", agent.episode)
            print("total step :", agent.step)
            agent.episode_R = 0.0
            agent.done = False        
        r.sleep()

if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print ('Interrupted')
        sys.exit(0)