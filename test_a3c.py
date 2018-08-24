import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import cv2
import sys
from collections import deque
sys.path.append("game/")
import wrapped_flappy_bird as game

N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 10000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 1e-6   # learning rate for actor
LR_C = 1e-6    # learning rate for critic
GLOBAL_EP = 0

flappyBird = game.GameState()

class ACNet(object):
	def __init__(self, scope):
		with tf.variable_scope(scope):
			self.s = tf.placeholder(tf.float32, [None,80,80,4], 'S')
			self.a_prob, self.v, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2_a, self.b_fc2_a, self.W_fc2_c, self.b_fc2_c = self._build_net(scope)
                    
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
	def __init__(self, name):
		self.flappyBird = game.GameState()
		self.name = name
		self.AC = ACNet(name)

	def work(self):
		action0 = 0  # do nothing
		observation0, reward0, terminal, score = self.flappyBird.frame_step(action0)
		observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
		GLOBAL_EP = 1
		while GLOBAL_EP < MAX_GLOBAL_EP:
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
				s = s_
				GLOBAL_EP += 1
				if done:
					sys.stdout.write("TIMESTEP:"+str(GLOBAL_EP)+"/ ACTION:"+str(a)+'\n')#"/ Q:"+str(Q)+'\n')
					break

if __name__ == "__main__":
	SESS = tf.Session()
	SCORE = []
	Work = Worker(GLOBAL_NET_SCOPE)  # we only need its params
	saver = tf.train.Saver([Work.AC.W_conv1]+[Work.AC.b_conv1]+[Work.AC.W_conv2]+[Work.AC.b_conv2]+[Work.AC.W_conv3]+[Work.AC.b_conv3]+[Work.AC.W_fc1]+[Work.AC.b_fc1]+[Work.AC.W_fc2_a]+[Work.AC.b_fc2_a]+[Work.AC.W_conv1]+[Work.AC.b_conv1]+[Work.AC.W_conv2]+[Work.AC.b_conv2]+[Work.AC.W_conv3]+[Work.AC.b_conv3]+[Work.AC.W_fc1]+[Work.AC.b_fc1]+[Work.AC.W_fc2_c]+[Work.AC.b_fc2_c])
	SESS.run(tf.global_variables_initializer())
	checkpoint = tf.train.get_checkpoint_state("saved_networks")
	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(SESS, checkpoint.model_checkpoint_path)
		print ("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print ("Could not find old network weights")
	Work.work()
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

