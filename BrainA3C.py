import multiprocessing
import threading
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
MAX_GLOBAL_EP = 65000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 1e-6   # learning rate for actor
LR_C = 1e-6    # learning rate for critic
GLOBAL_EP = 1
TIME=60000

flappyBird = game.GameState()

class ACNet(object):
	def __init__(self, scope, globalAC=None):
		if scope == GLOBAL_NET_SCOPE:   # get global network
			with tf.variable_scope(scope):
				self.s = tf.placeholder(tf.float32, [None,80,80,4], 'S')
				_, _, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_a, self.b_fc2_a, self.W_fc2_c, self.b_fc2_c = self._build_net(scope)
				self.a_params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_a, self.b_fc2_a]
				self.c_params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_c, self.b_fc2_c]
		else:   # local net, calculate losses 
			with tf.variable_scope(scope):
				self.s = tf.placeholder(tf.float32, [None,80,80,4], 'S')
				self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
				self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

				self.a_prob, self.v, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_a, self.b_fc2_a, self.W_fc2_c, self.b_fc2_c = self._build_net(scope)
				self.a_params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_a, self.b_fc2_a]
				self.c_params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_c, self.b_fc2_c]
				td = tf.subtract(self.v_target, self.v, name='TD_error')
				with tf.name_scope('c_loss'):
					self.c_loss = tf.reduce_mean(tf.square(td))
					
				with tf.name_scope('a_loss'):
					log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, 2, dtype=tf.float32), axis=1, keep_dims=True)
					exp_v = log_prob * tf.stop_gradient(td)
					entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),axis=1, keep_dims=True)  # encourage exploration
					self.exp_v = ENTROPY_BETA * entropy + exp_v
					self.a_loss = tf.reduce_mean(-self.exp_v)

				with tf.name_scope('local_grad'):
					self.a_grads = tf.gradients(self.a_loss, self.a_params)
					self.c_grads = tf.gradients(self.c_loss, self.c_params)

			with tf.name_scope('sync'):
				with tf.name_scope('pull'):
					self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
					self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
				with tf.name_scope('push'):
					self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
					self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    
	def _build_net(self, scope):
		with tf.variable_scope('actor'):
			# network weights
			W_conv1 = self.weight_variable([8,8,4,32])
			b_conv1 = self.bias_variable([32])
			W_conv2 = self.weight_variable([4,4,32,64])
			b_conv2 = self.bias_variable([64])
			W_conv3 = self.weight_variable([3,3,64,64])
			b_conv3 = self.bias_variable([64])
			W_fc1 = self.weight_variable([1600,512])
			b_fc1 = self.bias_variable([512])
			W_fc2_a = self.weight_variable([512,2])
			b_fc2_a = self.bias_variable([2])
			W_fc2_c = self.weight_variable([512,1])
			b_fc2_c = self.bias_variable([1])
			# hidden layers
			h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)
			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
			h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
			h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
			h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1)+b_fc1) 
			a_prob = tf.nn.softmax(tf.matmul(h_fc1,W_fc2_a) + b_fc2_a)
			v = tf.matmul(h_fc1,W_fc2_c) + b_fc2_c
		#a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
		#c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
		return a_prob, v, W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2_a,b_fc2_a,W_fc2_c,b_fc2_c

	def update_global(self, feed_dict):  # run by a local
		SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

	def pull_global(self):  # run by a local
		SESS.run([self.pull_a_params_op, self.pull_c_params_op])

	def choose_action(self, s):  # run by a local
		prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action

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

class Worker(object):
	def __init__(self, name, globalAC):
		self.flappyBird = game.GameState()
		self.name = name
		self.AC = ACNet(name, globalAC)

	def work(self):
		global GLOBAL_EP
		replayMemory = deque()
		action0 = 0  # do nothing
		observation0, reward0, terminal, score = self.flappyBird.frame_step(action0)
		observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
		total_step = 1
		start1 = time.time()
		start = time.time()
		while (not COORD.should_stop()) and (GLOBAL_EP < MAX_GLOBAL_EP) and (time.time() - start1 < TIME):
			s = np.stack((observation0, observation0, observation0, observation0), axis = 2)
			while True:
				a = self.AC.choose_action(s)
				s_,r,done,score = self.flappyBird.frame_step(a)
				if score != 'n':
					SCORE.append(score)
				s_ = cv2.cvtColor(cv2.resize(s_, (80, 80)), cv2.COLOR_BGR2GRAY)
				ret, s_ = cv2.threshold(s_,1,255,cv2.THRESH_BINARY)
				s_ = np.reshape(s_,(80,80,1))
				s_ = np.append(s[:,:,1:],s_,axis = 2)
				replayMemory.append((s,a,r,s_,done))

				if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
					state_batch = [data[0] for data in replayMemory]
					action_batch = [data[1] for data in replayMemory]
					reward_batch = [data[2] for data in replayMemory]
					nextState_batch = [data[3] for data in replayMemory]
					v_s_, a_prob= SESS.run([self.AC.v, self.AC.a_prob], {self.AC.s: s_[np.newaxis, :]})
					v_s_ = v_s_[0,0]
					if done:
						v_s_ = 0   # terminal
					buffer_v_target = []
					for r in reward_batch[::-1]:    # reverse buffer r
						v_s_ = r + GAMMA * v_s_
						buffer_v_target.append(v_s_)
					buffer_v_target.reverse()
				
					buffer_s, buffer_a, buffer_v_target = state_batch, action_batch, np.vstack(buffer_v_target)
					feed_dict = {
					self.AC.s: buffer_s,
					self.AC.a_his: buffer_a,
					self.AC.v_target: buffer_v_target,
					}
					self.AC.update_global(feed_dict)
				
					replayMemory.clear()
					self.AC.pull_global()
					sys.stdout.write("Time:"+str(time.time()-start1)+"/ Work_Name:"+self.name+"/ TIMESTEP:"+str(GLOBAL_EP)+"/ ACTION:"+str(a)+"/ ACTION_Prob:"+str(a_prob)+'\n')
					if GLOBAL_EP % 10000 == 0:
						saver.save(SESS, 'saved_networks/' + 'network', global_step = GLOBAL_EP)
					GLOBAL_EP += 1
					start = time.time()

				s = s_
				total_step += 1
				if done:
					break

if __name__ == "__main__":
	SESS = tf.Session()
	SCORE = []

	with tf.device("/cpu:0"):
		OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
		OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
		GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
		workers = []
		#Create worker
		for i in range(N_WORKERS):
			i_name = 'W_%i' % i   # worker name
			workers.append(Worker(i_name, GLOBAL_AC))

	COORD = tf.train.Coordinator()
	saver = tf.train.Saver()
	SESS.run(tf.global_variables_initializer())
	checkpoint = tf.train.get_checkpoint_state("saved_networks")
	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(SESS, checkpoint.model_checkpoint_path)
		print ("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print ("Could not find old network weights")

	worker_threads = []
	for worker in workers:
		job = lambda: worker.work()
		t = threading.Thread(target=job)
		t.start()
		worker_threads.append(t)
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
