#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from darknet_ros_msgs.msg import BoundingBoxes
import tensorflow as tf
import numpy
import math
import numpy as np

from gym import spaces

# encoder network related
from depth_semantic_two_nets_ros_black_bcg import *


scope = 'states'
original_name_list = ['relu_encoder_1', 'relu_encoder_2', 'relu_commands_1', 'relu_commands_2']
variable_list = ['weights', 'bias']

keys = [scope + '/' + i_origin_name + '/' + i_variable for i_origin_name in original_name_list for i_variable in variable_list]
values = [i_origin_name + '/' + i_variable for i_origin_name in original_name_list for i_variable in variable_list]
variable_map = {}
for i, key in enumerate(keys):
	variable_map[key] = values[i]

class NavigationState(object):

	def __init__(self, state_dim):
		# rospy.init_node('navigation_state', anonymous=True, log_level=rospy.INFO)
		self.state = 0.0
		
		self.image_data = np.zeros([1, img_height, img_wid, img_channel])
		
		self.semantic_image_data = np.zeros([1, img_height, img_wid, 1])
		self.semantic_origin_size = np.zeros([semantic_img_height, semantic_img_width])

		self.bridge = CvBridge()
		self.new_msg_received = False
		self.state_dim = state_dim

		self.sub_img = rospy.Subscriber("/camera/depth/image_raw", Image, self.callBackDepth)
		self.sub_delta_yaw = rospy.Subscriber("/radar/delt_yaw", Float64, callBackDeltYaw)
		self.sub_current_yaw = rospy.Subscriber("/radar/current_yaw", Float64, callBackCurrentYaw)
		self.sub_odom = rospy.Subscriber("/odom", Odometry, callBackOdom)
		self.sub_sem = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.callBackSemantic)

		# reward related
		self.delt_yaw_abs = 0.0
		print "check point<<<<<<<<<<<<<<<<<<<<<<"

		self.depth_img_input, self.semantic_img_input, self.global_direction_input, self.state_output, self.graph, self.scope = self.create_latent_network()
		self.state_value = None

		# self.sess = sess
		self.sess = tf.InteractiveSession(graph=self.graph)
		# self.sess = tf.Session(graph=self.graph)

		print "keyssssssssssssssssssss: ", keys

		# self.graph = tf.Graph()

		self.initiate_net()

		self.num_action_space = 2

		self.rate = rospy.Rate(100.0)

		high = np.array([1. for _ in range(self.num_action_space)])
		low = np.array([0. for _ in range(self.num_action_space)])
		self.action_space = spaces.Box(low=low, high=high)

	def create_latent_network(self):
		depth_data_input, semantic_data_input, line2_liput, state_output, graph, scope = build_front_graph()
		print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
		print [v.name for v in tf.trainable_variables()] 
		return depth_data_input, semantic_data_input, line2_liput, state_output, graph, scope

	def initiate_net(self):
		''' Predicting '''
		variables_to_restore = tf.contrib.framework.get_variables_to_restore()
		restorer = tf.train.Saver(variables_to_restore)

		self.sess.run(tf.initialize_all_variables())

		# print "sssssssssssafafgagawsgargagaesw"
		# graph = tf.get_default_graph()
		# fc1 = graph.get_tensor_by_name("relu_commands_1/weights:0")
		# print self.sess.run(fc1)

		restorer.restore(self.sess, model_path)

		# print "sssssssssssafafgagawsgargagaesw"

		# graph = tf.get_default_graph()
		# fc1 = graph.get_tensor_by_name("relu_commands_1/weights:0")
		# print self.sess.run(fc1)

		# # print "model path: ", model_path
		# print "sssssssssssafafgagawsgargagaesw"

		# result = self.sess.run( self.sess.graph.get_tensor_by_name( self.scope + "encoder/conv1/Relu:0" ) )
		# print result

		# graph = self.graph
		# fc1 = graph.get_tensor_by_name(self.scope + "encoder/conv1/Relu:0")
		# print self.sess.run(fc1)

		
	def callBackDepth(self, img):
		try:
			cv_image = bridge.imgmsg_to_cv2(img)
		except CvBridgeError as e:
			print(e)

		depth_image_list = cv2.resize(cv_image, (img_wid, img_height), interpolation=cv2.INTER_AREA)
		depth_image = np.array(depth_image_list).reshape(1, img_height, img_wid, 1)
		depth_image[depth_image > 4.5] = 0.0
		depth_image[np.isnan(depth_image)] = 0.0
		depth_image = depth_image * 56
		self.image_data = np.trunc(depth_image[:, :, :, :])
		self.new_msg_received = True


	def callBackSemantic(self, objects):
		self.semantic_image_data[:, :, :, :] = 0
		self.semantic_origin_size[:, :] = 0

		# print "objects.bounding_boxes: ", len(objects.bounding_boxes)
		for box in objects.bounding_boxes:
			label = 0
			# print box.Class
			if box.Class == 'person':
				label = 7
			elif box.Class == 'toothbrush':
				label = 7
			elif box.Class == 'cat':
				label = 6
			elif box.Class == 'dog':
				label = 6
			elif box.Class == 'laptop':
				label = 5
			elif box.Class == 'bed':
				label = 5
			elif box.Class == 'chair':
				label = 5
			elif box.Class == 'diningtable':
				label = 5
			elif box.Class == 'sofa':
				label = 5
			elif box.Class == 'traffic light':
				label = 5
			else:
				label = 3
			if label < 4:
				continue
			range_x_min = box.xmin
			range_x_max = box.xmax
			range_y_min = box.ymin
			range_y_max = box.ymax

			if range_x_min > semantic_img_width - 1:
				range_x_min = semantic_img_width - 1
			if range_x_min < 0:
				range_x_min = 0
			if range_x_max > semantic_img_width - 1:
				range_x_max = semantic_img_width - 1
			if range_x_max < 0:
				range_x_max = 0

			# for i in range(range_y_min - 1, range_y_max):
			#     for j in range(range_x_min - 1, range_x_max):
			#         semantic_origin_size[i, j] = np.trunc(label * 32)

			self.semantic_origin_size[range_y_min - 1:range_y_max, range_x_min - 1:range_x_max] = np.trunc(label * 32)

			# semantic_list = np.array(semantic_origin_size).reshape(1, img_height, img_wid, 1)

			semantic_image_list = cv2.resize(self.semantic_origin_size, (img_wid, img_height), interpolation=cv2.INTER_AREA)
			self.semantic_image_data = np.array(semantic_image_list).reshape(1, img_height, img_wid, 1)
		
			# cv2.imshow("semantic", self.semantic_image_data)
			# cv2.waitKey(5)


	def update_step_state(self):
		if self.new_msg_received:
			# print "Yes"
			# cv2.imshow("rgb2", self.image_data[0,:,:,:])
			# cv2.waitKey(0)
			data3_yaw_forward = np.ones([1, commands_compose_each]) * yaw_forward
			data3_yaw_backward = np.ones([1, commands_compose_each]) * yaw_backward
			data3_yaw_leftward = np.ones([1, commands_compose_each]) * yaw_leftward
			data3_yaw_rightward = np.ones([1, commands_compose_each]) * yaw_rightward
			data3_to_feed = np.concatenate([data3_yaw_forward, data3_yaw_backward, data3_yaw_leftward, data3_yaw_rightward], axis=1)
			
			concate_vector_value = self.sess.run(self.state_output,feed_dict={
				self.depth_img_input:self.image_data, self.semantic_img_input:self.semantic_image_data, self.global_direction_input: data3_to_feed
				})[0]
			# shape: (544, )
			self.state_value = concate_vector_value
		# else:
			# print "No new img data received!"
		self.new_msg_received = False

	def get_step_states(self):
		self.update_step_state()
		return self.state_value

	def update_delt_yaw_abs(self):
		self.delt_yaw_abs = get_yaw_delt_abs()

	def process_data(self):
		# TODO: specify the reasonable reward and finishing sign
		# print "rgb_image: ", self.image_data
		self.update_delt_yaw_abs()
		reward = -math.fabs(self.delt_yaw_abs)
		print "reward: ", reward
		done = False
		if reward > 3.14:
			print "done, reward: ", reward
			done = True
		return reward, done

	def check_all_systems_ready(self):
		"""
		We check that all systems are ready
		:return:
		"""
		imu_data = None
		while imu_data is None and not rospy.is_shutdown():
			try:
				for i in range(self.num_agents):
					print "self.uav_imu_names[i]: ", self.uav_imu_names[i]
					imu_data = rospy.wait_for_message(self.uav_imu_names[i], Imu, timeout=0.1)
					self.imus[i] = imu_data
				print("Current imu_data READY")
			except:
				print("Current imu_data not ready yet, retrying for getting robot base_orientation, and base_linear_acceleration")

		data_pose = None
		while data_pose is None and not rospy.is_shutdown():
			try:
				for i in range(self.num_agents):
					print "self.uav_odom_names[i]: ", self.uav_odom_names[i]
					data_pose = rospy.wait_for_message(self.uav_odom_names[i], Odometry, timeout=0.1)
					self.odoms[i] = data_pose
				print("Current odom READY")
			except:
				print("Current odom pose not ready yet, retrying for getting robot pose")

		
		print("ALL SYSTEMS READY")


	def get_init_states(self):
		#self.concatenate_states()
		return self.states_concatenate

	def calculate_total_reward(self):
		return

	def calculate_reward_payload_orientation(self):
		uav_orientation_ok = self.uav_orientation_ok()
		done = not uav_orientation_ok
		if done:
			rospy.logdebug("It fell, so the reward has to be very low")
			total_reward = self._done_reward
		else:
			rospy.logdebug("Calculate normal reward because it didn't fall.")
			euler = tf.transformations.euler_from_quaternion(
				[self.payload_state.x, self.payload_state.y, self.payload_state.z, self.payload_state.w])
			reward = np.linalg.norm(euler)
		return reward, done

	def testing_loop(self):

		rate = rospy.Rate(50)
		while not rospy.is_shutdown():
			self.calculate_total_reward()
			rate.sleep()


if __name__ == "__main__":
	rospy.init_node('monoped_state_node', anonymous=True)
	monoped_state = MonopedState(max_height=3.0,
								 min_height=0.6,
								 abs_max_roll=0.7,
								 abs_max_pitch=0.7)
	monoped_state.testing_loop()
