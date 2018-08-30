import multiprocessing
import threading
import queue
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import cv2
import sys
import time
from collections import deque
sys.path.append("game/")
import wrapped_flappy_bird as game

N_WORKERS = multiprocessing.cpu_count()
EP_MAX = 10000
MIN_BATCH_SIZE = 32
UPDATE_STEP = 15
GAMMA = 0.9
EPLISON = 0.2
LR_A = 1e-6   # learning rate for actor
LR_C = 1e-6    # learning rate for critic
GLOBAL_EP = 1
TIME=60000

flappyBird = game.GameState()

class PPONet(object):
	def __init__(self):
		self.SESS = tf.Session()
		self.s = tf.placeholder(tf.float32, [None,80,80,4], 'S')

		self.a_prob, self.v, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_a, self.b_fc2_a, self.W_fc2_c, self.b_fc2_c = self._build_net()
		self.a_params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_a, self.b_fc2_a]
		self.c_params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_c, self.b_fc2_c]
		oldprob, oldpar = self.build_old_net()
		#critic
		self.disr = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
		self.advantage = self.disr - self.v
		self.c_loss = tf.reduce_mean(tf.square(self.advantage))
		self.c_train = tf.train.AdamOptimizer(LR_C).minimize(self.c_loss)

		#actor
		self.update_oldnet_op = [old.assign(new) for old, new in zip(oldpar, self.a_params)]
		
		self.a = tf.placeholder(tf.int32, [None, ], 'A')
		self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
		
		a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype = tf.int32), self.a], axis = 1)
		new_prob = tf.gather_nd(params = self.a_prob, indices = a_indices)
		old_prob = tf.gather_nd(params = oldprob, indices = a_indices)
		ratio = new_prob/old_prob
		surr = ratio * self.adv
		
		self.a_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - EPLISON, 1. + EPLISON) * self.adv))
		self.a_train = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss)

		self.saver = tf.train.Saver()
		self.SESS.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(SESS, checkpoint.model_checkpoint_path)
			print ("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print ("Could not find old network weights")

	def _build_net(self):
		with tf.variable_scope('actor-critic'):
			# network weights
			W_conv1 = self.weight_variable([8,8,4,32], True)
			b_conv1 = self.bias_variable([32],True)
			W_conv2 = self.weight_variable([4,4,32,64], True)
			b_conv2 = self.bias_variable([64],True)
			W_conv3 = self.weight_variable([3,3,64,64], True)
			b_conv3 = self.bias_variable([64],True)
			W_fc1 = self.weight_variable([1600,512], True)
			b_fc1 = self.bias_variable([512],True)
			W_fc2_a = self.weight_variable([512,2], True)
			b_fc2_a = self.bias_variable([2],True)
			W_fc2_c = self.weight_variable([512,1], True)
			b_fc2_c = self.bias_variable([1],True)
			# hidden layers
			h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)
			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
			h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
			h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
			h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1)+b_fc1) 
			a_prob = tf.nn.softmax(tf.matmul(h_fc1,W_fc2_a) + b_fc2_a)
			v = tf.matmul(h_fc1,W_fc2_c) + b_fc2_c
		return a_prob, v, W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2_a,b_fc2_a,W_fc2_c,b_fc2_c

	def build_old_net(self):
		with tf.variable_scope('old_network'):
			W_conv1 = self.weight_variable([8,8,4,32], False)
			b_conv1 = self.bias_variable([32],False)
			W_conv2 = self.weight_variable([4,4,32,64], False)
			b_conv2 = self.bias_variable([64],False)
			W_conv3 = self.weight_variable([3,3,64,64], False)
			b_conv3 = self.bias_variable([64],False)
			W_fc1 = self.weight_variable([1600,512], False)
			b_fc1 = self.bias_variable([512],False)
			W_fc2 = self.weight_variable([512,2], False)
			b_fc2 = self.bias_variable([2],False)
			# hidden layers
			h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)
			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
			h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
			h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
			h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1)+b_fc1) 
			a_prob = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)
			params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'old_network')
		return a_prob, params
		
	def update(self): 
		global GLOBAL_UPDATE_COUNTER
		while not COORD.should_stop():
			if GLOBAL_EP < EP_MAX:
				UPDATE_EVENT.wait()
				self.SESS.run(self.update_oldnet_op)
				state_batch = [data[0] for data in replayMemory]
				action_batch = [data[1] for data in replayMemory]
				reward_batch = [data[2] for data in replayMemory]
				replayMemory.clear()
				reward_batch = np.array(reward_batch)[:, None]
				adv = self.SESS.run(self.advantage, {self.s: state_batch, self.disr: reward_batch})
				[self.SESS.run(self.a_train, {self.s: state_batch, self.a: action_batch, self.adv:adv}) for _ in range(UPDATE_STEP)]
				[self.SESS.run(self.c_train, {self.s: state_batch, self.disr: reward_batch}) for _ in range(UPDATE_STEP)]
				UPDATE_EVENT.clear()
				GLOBAL_UPDATE_COUNTER = 0
				ROLLING_EVENT.set()

	def choose_action(self, s): 
		prob_weights = self.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action

	def weight_variable(self, shape, train):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial, trainable = train)

	def bias_variable(self, shape, train):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial, trainable = train)

	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

class Worker(object):
	def __init__(self, name):
		self.flappyBird = game.GameState()
		self.name = name
		self.ppo = GLOBAL_PPO

	def work(self):
		global GLOBAL_EP, GLOBAL_UPDATE_COUNTER
		action0 = 0  # do nothing
		observation0, reward0, terminal, score = self.flappyBird.frame_step(action0)
		observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
		start = time.time()
		while (not COORD.should_stop()) and (GLOBAL_EP < EP_MAX) and (time.time() - start < TIME):
			s = np.stack((observation0, observation0, observation0, observation0), axis = 2)
			state_batch, action_batch, reward_batch = [], [], []
			while True:
				if not ROLLING_EVENT.is_set():
					ROLLING_EVENT.wait()
					state_batch, action_batch, reward_batch = [], [], []
				a = self.ppo.choose_action(s)
				s_,r,done,score = self.flappyBird.frame_step(a)
				if score != 'n':
					SCORE.append(score)
				s_ = cv2.cvtColor(cv2.resize(s_, (80, 80)), cv2.COLOR_BGR2GRAY)
				ret, s_ = cv2.threshold(s_,1,255,cv2.THRESH_BINARY)
				s_ = np.reshape(s_,(80,80,1))
				s_ = np.append(s[:,:,1:],s_,axis = 2)
				state_batch.append(s)
				action_batch.append(a)
				reward_batch.append(r)
				s = s_
				GLOBAL_UPDATE_COUNTER += 1

				if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:   # update global and assign to local net
					v_s_, a_prob = self.ppo.SESS.run([self.ppo.v, self.ppo.a_prob], {self.ppo.s: s_[np.newaxis, :]})
					v_s_ = v_s_[0,0]
					if done:
						v_s_ = 0   # terminal
					discounted_r = []
					for r in reward_batch[::-1]:    # reverse buffer r
						v_s_ = r + GAMMA * v_s_
						discounted_r.append(v_s_)
					discounted_r.reverse()

					[replayMemory.append((s,a,r)) for s, a, r in zip(state_batch, action_batch, reward_batch)]
					state_batch, action_batch, reward_batch = [], [], []

					if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
						ROLLING_EVENT.clear()
						UPDATE_EVENT.set()

					if done:
						break

			sys.stdout.write("Time:"+str(time.time()-start)+"/ Work_Name:"+self.name+"/ TIMESTEP:"+str(GLOBAL_EP)+"/ ACTION:"+str(a)+"/ ACTION_Prob:"+str(a_prob)+'\n')
			if GLOBAL_EP % 1000 == 0:
				self.ppo.saver.save(self.ppo.SESS, 'saved_networks/' + 'network', global_step = GLOBAL_EP)
			GLOBAL_EP += 1


if __name__ == "__main__":
	SCORE = []
	replayMemory = deque()
	GLOBAL_PPO = PPONet()  # we only need its params
	UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
	UPDATE_EVENT.clear()
	ROLLING_EVENT.set()
	workers = []
	#Create worker
	for i in range(N_WORKERS):
		i_name = 'W_%i' % i   # worker name
		workers.append(Worker(i_name))

	GLOBAL_UPDATE_COUNTER = 0
	COORD = tf.train.Coordinator()

	worker_threads = []
	for worker in workers:
		job = lambda: worker.work()
		t = threading.Thread(target=job)
		t.start()
		worker_threads.append(t)
	worker_threads.append(threading.Thread(target=GLOBAL_PPO.update,))
	worker_threads[-1].start()
	COORD.join(worker_threads)
	result=set(SCORE)
	length=len(SCORE)
	y=[]
	for i in result:
		num = SCORE.count(i)
		y.append(num)
		print('the percent of score ' + str(i) + ' is ' + str(num/length))
	x=list(result)
	plt.bar(x,y)
	plt.show()
