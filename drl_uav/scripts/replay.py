#!/usr/bin/env python

import rospy
import random as rd
import numpy, math
import sys
from collections import deque
from drl_uav.msg import replay_msg
from drl_uav.msg import status_save

class replay_mem:
    def __init__(self):
        self.mini_batch_size = 128
        self.memory_size = 500000
        self.buffer = deque(maxlen=self.memory_size)

    def sampling(self):
        if(len(self.buffer) > 1000):
            mini_batch = rd.sample(self.buffer,self.mini_batch_size)

            state = self.normalize(numpy.asarray([sample[0] for sample in mini_batch]))
            state = numpy.reshape(state,(1,self.mini_batch_size*6))
            action = self.normalize(numpy.asarray([sample[1] for sample in mini_batch]))
            action = numpy.reshape(state,(1,self.mini_batch_size*3))
            reward = self.normalize(numpy.asarray([sample[2] for sample in mini_batch]))
            next_state = self.normalize(numpy.asarray([sample[3] for sample in mini_batch]))
            state = numpy.reshape(state,(1,self.mini_batch_size*6))
            done = self.normalize(numpy.asarray([sample[4] for sample in mini_batch]))   
	
            self.batch_pub(state,action,reward,next_state,done)

    def batch_pub(self,state,action,reward,next_state,done):
        replay_msg.state  = state
        replay_msg.action = action
        replay_msg.next_state = next_state
        replay_msg.reward = reward
        replay_msg.done = done
        batch_pub.publish(replay_msg)

    def memory_update(self,state,action,next_state,reward,done):
        self.buffer.append = (state,action,next_state,reward,done)

    def normalize(self,value_batch):
        batch_ep = 1e-5
        batch_mean = numpy.sum(value_batch)/len(value_batch)
        batch_val  = numpy.sum((value_batch-batch_mean)**2)/len(value_batch)
        normalized_value = (value_batch-batch_mean)/math.sqrt(batch_val+batch_ep)
        return normalized_value

def getEnvdata(data):
    replay_mem.memory_update(data.state,data.action,data.next_state,data.reward,data.done)

batch_pub = rospy.Publisher('minibatch',replay_msg,queue_size=10)
rospy.Subscriber('RL_status_save', status_save, getEnvdata)      

def main_loop():
    rospy.init_node('replay_memory', anonymous=True)

    replay = replay_mem()
    
    r = rospy.Rate(40)  # 20Hz

    while not rospy.is_shutdown():
	    replay.sampling()
	    r.sleep()

if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print ('Interrupted')
        sys.exit(0)  
       
