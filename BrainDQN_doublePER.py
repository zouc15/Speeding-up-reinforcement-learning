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
EXPLORE = 50000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001#0.001 # final value of epsilon
INITIAL_EPSILON = 0.1#0.01 # starting value of epsilon
REPLAY_MEMORY = 10000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply

class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        #self.tree = tf.Variable(tf.zeros([2 * capacity - 1]))
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        #self.data = tf.Variable(tf.zeros([capacity],dtype=tf.tuple))
        
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        #self.data=tf.scatter_nd_update(self.data,[self.data_pointer],data)
        self.update(tree_idx, p)  # update tree_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:    
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), deque(), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class BrainDQN:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = Memory(capacity = REPLAY_MEMORY) 
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		self.Loss = []
		# init Q network
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

		# init Target Q Network
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
		
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

		W_fc1 = self.weight_variable([1600,512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512,self.actions])
		b_fc2 = self.bias_variable([self.actions])

		# input layer

		stateInput = tf.placeholder("float",[None,80,80,4])

		# hidden layers
		h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

		h_conv3_flat = tf.reshape(h_conv3,[-1,1600])

		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1)+b_fc1)
		
		# Q Value layer
		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.ISWeight = tf.placeholder("float",[None,1])
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
		self.abs_errors = tf.abs(self.yInput - Q_Action)
		self.cost = tf.reduce_mean(self.ISWeight * tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


	def trainQNetwork(self):

		
		# Step 1: obtain random minibatch from replay memory
		tree_idx,minibatch,ISWeight = self.replayMemory.sample(BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		Qeval4next = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})
		act4next = np.argmax(Qeval4next, axis=1) 
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * QValue_batch[i][act4next[i]])

		_,abs_errors,self.loss=self.session.run([self.trainStep,self.abs_errors,self.cost],feed_dict={self.yInput : y_batch, self.actionInput : action_batch, self.stateInput : state_batch, self.ISWeight : ISWeight})
		self.replayMemory.batch_update(tree_idx,abs_errors)
		self.Loss.append(self.loss)

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

		
	def setPerception(self,nextObservation,action,reward,terminal):
		newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
		tmp = np.zeros(2)
		tmp[action] = 1
		action = tmp
		self.replayMemory.store((self.currentState,action,reward,newState,terminal))
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
				"/ TIME", Time, "/ ACTION", action, "/COST", self.loss)

		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		action = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action = action_index
			else:
				action_index = np.argmax(QValue)
				action = action_index
		else:
			action = 0 # do nothing
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

	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		
	def plot_cost(self):
		import matplotlib.pyplot as plt
		plt.plot(np.arange(len(self.Loss)), self.Loss)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()
