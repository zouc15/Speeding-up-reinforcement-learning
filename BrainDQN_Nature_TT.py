import tensorflow as tf 
import numpy as np 
import random
import time
from collections import deque
from tensor_train import TensorTrain

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0.1#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100

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
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

		# init Target Q Network
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

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

		W_conv2 = self.weight_variable([4,4,32,64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3,3,64,64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.ttweight_variable([[5,5,4,4,4,4], [5,5,4,4,2,2]], rank=2)
		b_fc1 = self.bias_variable([1600])

		W_fc2 = self.weight_variable([1600,self.actions])
		b_fc2 = self.bias_variable([self.actions])

		# input layer

		stateInput = tf.placeholder("float",[None,80,80,4])

		# hidden layers
		h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,2) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

		h_conv3_flat = tf.reshape(h_conv3,[-1,6400])
		
		h_conv3_tt = self.to_tt_matrix(h_conv3_flat,5)

		h_fc1 = tf.nn.relu(self.full(self.ttmul(h_conv3_tt,W_fc1)) + b_fc1)

		# Q Value layer
		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)
		tt_cores=[]
		for i in range(self.W_fc1.ndims()):
			tt_cores.append(tf.assign(self.W_fc1T._tt_cores[i], self.W_fc1._tt_cores[i]))
		self.W_fc1T=TensorTrain(tt_cores, self.W_fc1.get_raw_shape(),self.W_fc1.get_tt_ranks(),convert_to_tensor=False)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		#self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


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
		if self.timeStep==10000:
			Time=time.time()-self.start
			print(Time)
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

	def ttvariable(self,name,initializer=None,collections=None,validate_shape=True):
		variable_cores = []
		# Create new variable.
		with tf.variable_scope(name):
			num_dims = initializer.ndims()
			for i in range(num_dims):
				curr_core_var = tf.Variable(initializer._tt_cores[i],
                                                            collections=collections,
                                                            name='core_%d' % i)
				variable_cores.append(curr_core_var)
		v = TensorTrain(variable_cores, initializer.get_raw_shape(),
						initializer.get_tt_ranks(),
						convert_to_tensor=False)
		tf.add_to_collection('TensorTrainVariables', v)
		return v

	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def ttweight_variable(self,shape,rank):
            shape=np.array(shape)
            tt_rank=np.array(rank)
            num_dim=shape[0].size
            if tt_rank.size==1:
                tt_rank=tt_rank*np.ones(num_dim-1)
                tt_rank=np.concatenate([[1],tt_rank,[1]])
            tt_rank=tt_rank.astype(int)
            var=np.prod(tt_rank)
            cr_exponent=-1/(2*num_dim)
            var=np.prod(tt_rank**cr_exponent)
            stddev=np.sqrt(2/(np.prod(shape[0])+np.prod(shape[1])))
            core_stddev=stddev**(1/num_dim)*var
            tt_cores=[None]*num_dim
            for i in range(num_dim):
                curr_core_shape=(tt_rank[i],shape[0][i],shape[1][i],tt_rank[i+1])
                tt_cores[i]=tf.random_normal(curr_core_shape,mean=0,stddev=core_stddev)
            initial=TensorTrain(tt_cores,shape,tt_rank)
            return self.ttvariable('Weight',initializer=initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	def transpose(self,tt_matrix):
            transposed_tt_cores=[]
            for core_idx in range(tt_matrix.ndims()):
                curr_core=tt_matrix._tt_cores[core_idx]
                transposed_tt_cores.append(tf.transpose(curr_core,(0,2,1,3)))
            tt_matrix_shape=tt_matrix.get_raw_shape()
            transposed_shape=tt_matrix_shape[1],tt_matrix_shape[0]
            return TensorTrain(transposed_tt_cores,transposed_shape,tt_matrix.get_tt_ranks())
            
	def to_tt_matrix(self,h,max_rank):
		raw_shape=h.shape.as_list()[0]
		if raw_shape==32:
			shape=[[2,2,2,2,2,1],[5,5,4,4,4,4]]
		else:
			shape=[[1,1,1,1,1,1],[5,5,4,4,4,4]]
		shape=np.array(shape)
		tens = tf.reshape(h,shape.flatten())
		d = len(shape[0])
		# transpose_idx = 0, d, 1, d+1 ...
		transpose_idx = np.arange(2 * d).reshape(2, d).T.flatten()
		transpose_idx = transpose_idx.astype(int)
		tens = tf.transpose(tens, transpose_idx)
		new_shape = np.prod(shape, axis=0)
		tens = tf.reshape(tens, new_shape)
		max_rank=(max_rank*np.ones(d+1)).astype(np.int32)
		ranks=[1]*(d+1)
		tt_cores=[]
		for core_idx in range(d - 1):
			curr_mode = new_shape[core_idx]
			rows = ranks[core_idx] * curr_mode
			tens = tf.reshape(tens, [rows, -1])
			columns = tens.get_shape()[1].value
			s, u, v = tf.svd(tens, full_matrices=False)
			if max_rank[core_idx + 1] == 1:
				ranks[core_idx + 1] = 1
			else:
				ranks[core_idx + 1] = min(max_rank[core_idx + 1], rows, columns)
			u = u[:, 0:ranks[core_idx + 1]]
			s = s[0:ranks[core_idx + 1]]
			v = v[:, 0:ranks[core_idx + 1]]
			core_shape = (ranks[core_idx], shape[0, core_idx], shape[1, core_idx], ranks[core_idx + 1])
			tt_cores.append(tf.reshape(u, core_shape))
			tens = tf.matmul(tf.diag(s), tf.transpose(v))
		core_shape = (ranks[d - 1], shape[0, -1],shape[1, -1], ranks[d])
		tt_cores.append(tf.reshape(tens, core_shape))
		return TensorTrain(tt_cores,shape,ranks)        

	def dtmul(self,h,W):
            h=tf.transpose(h)
            W=self.transpose(W)
            ndims=W.ndims()
            W_columns=W.get_shape()[1].value
            h_rows=h.get_shape()[0].value
            W_shape=np.array(W.get_shape().as_list())
            W_raw_shape=np.array([s.as_list() for s in W.get_raw_shape()])
            h_shape=h.get_shape().as_list()
            W_ranks=np.array(W.get_tt_ranks().as_list())
            data=tf.transpose(h)
            data=tf.reshape(data,(-1,W_raw_shape[1][-1],1))
            for core_idx in reversed(range(ndims)):
                curr_core=W._tt_cores[core_idx]
                data=tf.einsum('aijb,rjb->ira',curr_core,data)
                if core_idx>0:
                    new_data_shape=(-1,W_raw_shape[1][core_idx-1],W_ranks[core_idx])
                    data=tf.reshape(data,new_data_shape)
            return tf.transpose(tf.reshape(data,(W_shape[0],-1)))
            
	def ttmul(self,tt_matrix_a,tt_matrix_b):
		ndims = tt_matrix_a.ndims()
		result_cores=[]
		# TODO: name the operation and the resulting tensor.
		a_shape = np.array([s.as_list() for s in tt_matrix_a.get_raw_shape()])
		a_ranks = np.array(tt_matrix_a.get_tt_ranks().as_list())
		b_shape = np.array([s.as_list() for s in tt_matrix_b.get_raw_shape()])
		b_ranks = np.array(tt_matrix_b.get_tt_ranks().as_list())
		for core_idx in range(ndims):
			a_core = tt_matrix_a._tt_cores[core_idx]
			b_core = tt_matrix_b._tt_cores[core_idx]
			curr_res_core = tf.einsum('aijb,cjkd->acikbd', a_core, b_core)
			res_left_rank = a_ranks[core_idx] * b_ranks[core_idx]
			res_right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
			left_mode = a_shape[0][core_idx]
			right_mode = b_shape[1][core_idx]
			core_shape = (res_left_rank, left_mode, right_mode, res_right_rank)
			curr_res_core = tf.reshape(curr_res_core, core_shape)
			result_cores.append(curr_res_core)
		res_shape = (tt_matrix_a.get_raw_shape()[0], tt_matrix_b.get_raw_shape()[1])
		static_a_ranks = tt_matrix_a.get_tt_ranks()
		static_b_ranks = tt_matrix_b.get_tt_ranks()
		out_ranks = [a_r * b_r for a_r, b_r in zip(static_a_ranks, static_b_ranks)]
		return TensorTrain(result_cores, res_shape, out_ranks)
	
	def full(self,tt):
		num_dims = tt.ndims()
		ranks = np.array(tt.get_tt_ranks().as_list())
		shape = np.array(tt.get_shape().as_list())
		raw_shape = np.array([s.as_list() for s in tt.get_raw_shape()])
		res = tt._tt_cores[0]
		for i in range(1, num_dims):
			res = tf.reshape(res, (-1, ranks[i]))
			curr_core = tf.reshape(tt._tt_cores[i], (ranks[i], -1))
			res = tf.matmul(res, curr_core)
		intermediate_shape = []
		for i in range(num_dims):
			intermediate_shape.append(raw_shape[0][i])
			intermediate_shape.append(raw_shape[1][i])
		res = tf.reshape(res, intermediate_shape)
		transpose = []
		for i in range(0, 2 * num_dims, 2):
			transpose.append(i)
		for i in range(1, 2 * num_dims, 2):
			transpose.append(i)
		res = tf.transpose(res, transpose)
		return tf.reshape(res,shape)

	def plot_cost(self):
		import matplotlib.pyplot as plt
		plt.plot(np.arange(len(self.Loss)), self.Loss)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()        
