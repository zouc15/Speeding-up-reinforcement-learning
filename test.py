import tensorflow as tf
import numpy as np
import time

RANK=2

def conv2d(x, W, stride_h, stride_w, group):
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


x=tf.Variable(tf.random_normal([32,200,200,64],stddev=1))

W1=tf.Variable(tf.random_normal([1,1,64,RANK],stddev=1))
W2=tf.Variable(tf.random_normal([8,1,1,RANK],stddev=1))
W3=tf.Variable(tf.random_normal([1,8,1,RANK],stddev=1))
W4=tf.Variable(tf.random_normal([1,1,RANK,128],stddev=1))
start=time.time()
h1 = conv2d(x,W1,1,1,1)
h2 = conv2d(h1,W2,1,1,RANK)
h3 = conv2d(h2,W3,1,1,RANK)
h4 = conv2d(h3,W4,1,1,1)
print('time with cp decomposition'+str(time.time()-start))

W=tf.Variable(tf.random_normal([8,8,64,128],stddev=1))
start=time.time()
h=conv2d(x,W,1,1,1)
print('time without cp decomposition'+str(time.time()-start))
