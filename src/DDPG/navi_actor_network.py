import tensorflow as tf 
import numpy as np
import math
from depth_semantic_two_nets_ros_black_bcg import *

# Hyper Parameters
LAYER1_SIZE = fully_paras["input_len"]
LAYER2_SIZE = fully_paras["layer1_len"]
LAYER3_SIZE = fully_paras["layer2_len"]
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64





class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create actor network
		self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

		# create target actor network
		self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

		self.initiate()

		# define training rules
		self.create_training_method()
		self.update_target()
		#self.load_network()

	def initiate(self):
		# variables_to_restore = tf.contrib.framework.get_variables_to_restore()
		keys = ['relu_all_1/weights', 'relu_all_1/bias', 'relu_all_2/weights', 'relu_all_2/bias', 'relu_all_3/weights',
							  'relu_all_3/bias','relu_all_4/weights', 'relu_all_4/bias']
		values = [self.net[0], self.net[1], self.net[2], self.net[3], self.net[4], self.net[5], self.net[6], self.net[7]]

		variable_map = {}
		for i, key in enumerate(keys):
			variable_map[key] = values[i]

		print "variable_map: ", variable_map
		restorer = tf.train.Saver(var_list = variable_map)

		self.sess.run(tf.initialize_all_variables())

		graph = tf.get_default_graph()
		fc1 = graph.get_tensor_by_name("relu_all_1/weights:0")
		print self.sess.run(fc1)

		restorer.restore(self.sess, model_path)
		print "sssssssssssafafgagawsgargagaesw"
		graph = tf.get_default_graph()
		fc1 = graph.get_tensor_by_name("relu_all_1/weights:0")
		print self.sess.run(fc1)


	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

	# def create_network(self):
	#   state_input, action_output, [W1,b1,W2,b2,W3,b3,W4,b4] = custom_nn.build_graph()
	#   return state_input, action_output, [W1,b1,W2,b2,W3,b3,W4,b4]

	def create_network(self,state_dim,action_dim):
		# state_input, action_output,[W1,b1,W2,b2,W3,b3,W4,b4] = build_rear_graph
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
		layer3_size = LAYER3_SIZE

		# g2 = tf.Graph()
		# with g2.as_default() as g:
		#     with g.name_scope( "" ) as g1_scope:

		state_input = tf.placeholder("float",[None,state_dim])

		with tf.variable_scope("relu_all_1"):
			W1 = self.variable([state_dim,layer1_size],state_dim, name='weights')
			b1 = self.variable([layer1_size],state_dim, name='bias')
		with tf.variable_scope("relu_all_2"):
			W2 = self.variable([layer1_size,layer2_size],layer1_size, name='weights')
			b2 = self.variable([layer2_size],layer1_size, name='bias')
		with tf.variable_scope("relu_all_3"):
			W3 = self.variable([layer2_size,layer3_size],layer2_size, name='weights')
			b3 = self.variable([layer3_size],layer2_size, name='bias')
		with tf.variable_scope("relu_all_4"):
			W4 = tf.Variable(tf.random_uniform([layer3_size,action_dim],-3e-3,3e-3), name='weights')
			b4 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3), name='bias')

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
		layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
		action_output = tf.tanh(tf.matmul(layer3,W4) + b4)

		return state_input,action_output,[W1,b1,W2,b2,W3,b3,W4,b4]


	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
		layer3 = tf.nn.relu(tf.matmul(layer2,target_net[4]) + target_net[5])
		action_output = tf.tanh(tf.matmul(layer3,target_net[6]) + target_net[7])

		return state_input,action_output,target_update,target_net

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})

	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]


	def target_actions(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})

	# f fan-in size
	def variable(self,shape,f, name):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)), name=name)
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

		
