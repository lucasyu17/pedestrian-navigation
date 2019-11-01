#!/usr/bin/env python

import gym
import time
import numpy as np
import random
# import tensorflow.contrib.layers as layers

from gym import wrappers
from std_msgs.msg import Float64

# ROS packages required
import rospy
import rospkg
from gazebo_connection import GazeboConnection
# import our training environment
import navi_env
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from navi_ddpg import DDPG



if __name__ == '__main__':
    rospy.init_node('navigation_env', anonymous=True, log_level=rospy.INFO)
    env = gym.make('navigation_gym-v0')
    state_dim = 544
    action_dim = 2
    rate = rospy.Rate(50)
    pub_reset_flag = rospy.Publisher("/reset_flag", Float64, queue_size=1)
#####################################################################
    ddpg = DDPG(env)
#########################################################

    nepisodes = 200
    nsteps = 1000
    print ("start training...")
    # rate = rospy.Rate(50)
    reset_flag = Float64()
    while not rospy.is_shutdown():
        for episode in range(nepisodes):

            print "episode: ", episode+1
            # Initialize the environment and get first state of the robot
            print "env.reset..."
            # Now We return directly the stringuified observations called state

            current_state = env.reset()

            print "pauseSim outer"
            env.gazebo.pauseSim()

            reset_flag.data = 1.0
            pub_reset_flag.publish(reset_flag)

            for step in range(nsteps):
                print "step: ", step
                actions = ddpg.noise_action(current_state)
                # actions = np.random.rand(1, 2)
                # print "actions: ", actions
                next_state, reward, done = env.step(actions)
                ddpg.perceive(current_state,actions,reward,next_state,done)
                current_state = next_state
                if done:
                    print('Done!!!!!!!!!!!! at epoch{} , reward:{}'.format(episode,reward))
                    # maddpg.summary(episode)
                    break
                # rate.sleep()

            print "unpauseSim outer"
            env.gazebo.unpauseSim()

    env.close()




    
