#!/usr/bin/env python

import gym
import rospy
import numpy as np

from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register

from gazebo_connection import GazeboConnection
import rosparam

from navi_state import NavigationState

import sys
import os
import time

import tensorflow as tf
import tf as transform
from rosgraph_msgs.msg import Log
from geometry_msgs.msg import Twist

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState 
import random


#register the training environment in the gym as an available one
reg = register(
    id='navigation_gym-v0',
    entry_point='navi_env:Navigation_Env',
    # timestep_limit=50,
    )


state_dim = 512 + 32 + 32

class Navigation_Env(gym.Env):

    def __init__(self):
        # self.sess = tf.InteractiveSession()
        # action publisher
        self.cmd_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=2)

        self.running_step = 1/50.0

        self.num_agents = 4
        self.state_dim = state_dim
        self.action_dim = 2

        self.rate = rospy.Rate(40.0)

        high = np.array([1. for _ in range(self.action_dim)])
        low = np.array([0. for _ in range(self.action_dim)])
        self.action_space = spaces.Box(low=low, high=high)

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()

        # state object
        self.state_object = NavigationState(self.state_dim)

        self.state_msg = ModelState()
        
        self._seed()

        print ("end of init...")
        self.gazebo.unpauseSim()

    # A function to initialize the random generator

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Resets the state of the environment and returns an initial observation.
    def _reset(self):
        # 0st: We pause the Simulator
        # rospy.logdebug("Pausing SIM...")
        # self.gazebo.pauseSim()

        # 1st: resets the simulation to initial values
        # print "unpauseSim 1"
        self.gazebo.unpauseSim()


        rospy.logdebug("Reset SIM...")
        self.gazebo.resetSim()

        # random initial positions and orientations
        self.generate_random_init_pose()

        # Get the State Discrete Stringuified version of the observations
        while not self.state_object.new_msg_received:
            print "waiting for new images to arrive!"
            time.sleep(0.4)
        state = self.get_step_state()
        # print "got"
        # 7th: pauses simulation
        rospy.logdebug("Pausing SIM...")
        # print "pauseSim 1"
        self.gazebo.pauseSim()

        return state

    def _step(self, actions):
        # action to the gazebo environment
        self.gazebo.unpauseSim()

        # st = rospy.Time.now()

        action_fw = np.clip(actions, 0, 1)
        move_cmd = Twist()
        move_cmd.linear.x = action_fw[0]
        move_cmd.angular.z = action_fw[1]
        self.cmd_pub.publish(move_cmd)

        self.rate.sleep()
        time.sleep(0.025)

        self.gazebo.pauseSim()

        # end = rospy.Time.now()
        # print "time step: ", (end - st).to_sec()

        state = self.get_step_state()

        # finally we get an evaluation based on what happened in the sim
        reward, done = self.state_object.process_data()

        return state, reward, done


    def generate_random_init_pose(self):
        init_positions = {  
                            13:[-15.0, 11.0, 0.0], 14:[-6.0, 11.0, 0.0], 15:[5.0, 11.0, 0.0], 16:[14.0, 11.0, 0.0],
                            9:[-15.0, 5.0, 0.0], 10:[-6.0, 5.0, 0.0], 11:[5.0, 3.0, 0.0], 12:[14.0, 3.0, 0.0],
                            5:[-15.0, -3.0, 0.0], 6:[-6.0, -3.0, 0.0], 7:[5.0, -3.0, 0.0], 8:[14.0, -3.0, 0.0],
                            1:[-15, -10.5, 0.0], 2:[-6.0, -3.0, 0.0], 3:[5.0, -10.5, 0.0], 4:[14.0, -10.5, 0.0],
                        }
        init_eulers = {
                        13:[[0.0, 0.0, 0.0], [0.0, 0.0, -1.57]],

                        14:[[0.0, 0.0, 0.0], [0.0, 0.0, 3.14], [0.0, 0.0, -1.57]], 

                        15:[[0.0, 0.0, 0.0], [0.0, 0.0, 3.14], [0.0, 0.0, -1.57]], 

                        16:[[0.0, 0.0, 3.14], [0.0, 0.0, -1.57]],

                        9:[[0.0, 0.0, 1.57],[0.0, 0.0, 0.0], [0.0, 0.0, -1.57]], 

                        10:[[0.0, 0.0, 1.57], [0.0, 0.0, 3.14], [0.0, 0.0, -1.57]],

                        11:[[0.0, 0.0, 1.57],[0.0, 0.0, 0.0], [0.0, 0.0, -1.57]], 

                        12:[[0.0, 0.0, 1.57], [0.0, 0.0, 3.14], [0.0, 0.0, -1.57]],

                        5:[[0.0, 0.0, 1.57],[0.0, 0.0, 0.0], [0.0, 0.0, -1.57]], 

                        6:[[0.0, 0.0, 0.0],[0.0, 0.0, 1.57],[0.0, 0.0, 3.14],[0.0, 0.0, -1.57]],

                        7:[[0.0, 0.0, 1.57],[0.0, 0.0, 3.14],[0.0, 0.0, -1.57]], 

                        8:[[0.0, 0.0, 1.57], [0.0, 0.0, 3.14], [0.0, 0.0, -1.57]],

                        1:[[0.0, 0.0, 1.57], [0.0, 0.0, 0.0]],

                        2:[[0.0, 0.0, 0.0], [0.0, 0.0, 1.57], [0.0, 0.0, 3.14]], 

                        3:[[0.0, 0.0, 1.57], [0.0, 0.0, 0.0], [0.0, 0.0, 3.14]], 

                        4:[[0.0, 0.0, 1.57], [0.0, 0.0, 3.14]]
                        }


        rand_index = random.randint(1, 16)
        print "randint: ", rand_index

        position = init_positions[rand_index]
        euler_list = init_eulers[rand_index]
        rand_index_euler = random.randint(0, len(euler_list)-1)

        print "rand_index_euler: ", rand_index_euler
        euler = euler_list[rand_index_euler]
        print position
        print euler
        quat_ = transform.transformations.quaternion_from_euler(euler[0], euler[1], euler[2]) 
        self.state_msg.model_name = 'mobile_base'
        self.state_msg.pose.position.x = position[0]
        self.state_msg.pose.position.y = position[1]
        self.state_msg.pose.position.z = position[2]
        self.state_msg.pose.orientation.x = quat_[0]
        self.state_msg.pose.orientation.y = quat_[1]
        self.state_msg.pose.orientation.z = quat_[2]
        self.state_msg.pose.orientation.w = quat_[3]

        self.set_turtlebot_init_pose()


    def set_turtlebot_init_pose(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( self.state_msg )

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def get_init_state(self):
        return self.state_object.get_init_states()

    def get_step_state(self):
        return self.state_object.get_step_states()
