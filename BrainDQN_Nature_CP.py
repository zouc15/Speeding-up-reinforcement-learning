import tensorflow as tf 
import numpy as np 
import random
import time
from collections import deque
from tensor_train import TensorTrain

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100
RANK=10

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply

class BrainDQN:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		self.Loss = []
		# init Q network
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv21,self.W_conv22,self.W_conv23,self.W_conv24,self.b_conv2,self.W_conv31,self.W_conv32,self.W_conv33,self.W_conv34,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

		# init Target Q Network
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv21T,self.W_conv22T,self.W_conv23T,self.W_conv24T,self.b_conv2T,self.W_conv31T,self.W_conv32T,self.W_conv33T,self.W_conv34T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv21T.assign(self.W_conv21),self.W_conv22T.assign(self.W_conv22),self.W_conv23T.assign(self.W_conv23),self.W_conv24T.assign(self.W_conv24),self.b_conv2T.assign(self.b_conv2),self.W_conv31T.assign(self.W_conv31),self.W_conv32T.assign(self.W_conv32),self.W_conv33T.assign(self.W_conv33),self.W_conv34T.assign(self.W_conv34),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
		
		self.createTrainingMethod()

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print ("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
				print ("Could not find old network weights")


	def createQNetwork(self):
		# network weights
		W_conv1 = self.weight_variable([8,8,4,32])
		b_conv1 = self.bias_variable([32])

		W_conv21 = self.weight_variable([1,1,32,RANK])
		W_conv22 = self.weight_variable([4,1,1,RANK])
		W_conv23 = self.weight_variable([1,4,1,RANK])
		W_conv24 = self.weight_variable([1,1,RANK,64])
		b_conv2 = self.bias_variable([64])

		W_conv31 = self.weight_variable([1,1,64,RANK])
		W_conv32 = self.weight_variable([3,1,1,RANK])
		W_conv33 = self.weight_variable([1,3,1,RANK])
		W_conv34 = self.weight_variable([1,1,RANK,64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([1600,512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512,self.actions])
		b_fc2 = self.bias_variable([self.actions])

		# input layer

		stateInput = tf.placeholder("float",[None,80,80,4])

		# hidden layers
		h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4,4,1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv21 = self.conv2d(h_pool1,W_conv21,1,1,1)
		h_conv22 = self.conv2d(h_conv21,W_conv22,2,1,RANK)
		h_conv23 = self.conv2d(h_conv22,W_conv23,1,2,RANK)
		h_conv24 = tf.nn.relu(self.conv2d(h_conv23,W_conv24,1,1,1) + b_conv2)

		h_conv31 = self.conv2d(h_conv24,W_conv31,1,1,1)
		h_conv32 = self.conv2d(h_conv31,W_conv32,1,1,RANK)
		h_conv33 = self.conv2d(h_conv32,W_conv33,1,1,RANK)
		h_conv34 = tf.nn.relu(self.conv2d(h_conv33,W_conv34,1,1,1) + b_conv3)

		h_conv3_flat = tf.reshape(h_conv34,[-1,1600])

		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1)+b_fc1)
		
		# Q Value layer
		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

		return stateInput,QValue,W_conv1,b_conv1,W_conv21,W_conv22,W_conv23,W_conv24,b_conv2,W_conv31,W_conv32,W_conv33,W_conv34,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

	def copyTargetQNetwork(self):
                self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


	def trainQNetwork(self):

		
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		_,self.loss=self.session.run([self.trainStep,self.cost],feed_dict={self.yInput : y_batch, self.actionInput : action_batch, self.stateInput : state_batch})
		self.Loss.append(self.loss)
		#self.trainStep.run(feed_dict={self.yInput : y_batch,self.actionInput : action_batch,self.stateInput : state_batch})

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

		
	def setPerception(self,nextObservation,action,reward,terminal):
		#newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
			start=time.time()
			self.trainQNetwork()
			Time=time.time()-start
			# print info
			state = ""
			if self.timeStep <= OBSERVE:
				state = "observe"
			elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
				state = "explore"
			else:
				state = "train"
			
			print("TIMESTEP", self.timeStep, "/ STATE", state, \
				"/ TIME", Time, "/ ACTION", np.array(action).nonzero()[0][0], "/COST", self.loss)

		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
		if self.timeStep==0:
			self.start=time.time()
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		if self.timeStep==100:
			print(time.time()-self.start)
		action = np.zeros(self.actions)
		action_index = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action[action_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1 # do nothing
		# change episilon
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
		return action

	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self,x, W, stride_h, stride_w, group):
		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride_h, stride_w, 1], padding="SAME")
		if group==1:
			conv = convolve(x, W)
		else:
			#group means we split the input  into 'group' groups along the third demention
			input_groups = tf.split(x, group, 3)
			kernel_groups = tf.split(W, group, 3)
			output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
			conv = tf.concat(output_groups,3)
		return conv

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		
	def plot_cost(self):
		import matplotlib.pyplot as plt
		plt.plot(np.arange(len(self.Loss)), self.Loss)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()
