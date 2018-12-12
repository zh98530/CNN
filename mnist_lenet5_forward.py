#coding:utf-8

import tensorflow as tf

#输入图片为28*28*1，第一层卷积核为5*5*1*32，第二层卷积核为5*5*32*64
IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10

def get_weight(shape,regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: 
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01,shape=shape))
	return b

#卷积计算函数提取特征，x为输入，w为卷积核，步长为[横向:1,纵向:1]，padding选择全零填充
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化计算函数取max，x为输入，池化核为[2,2]，步长为[横向:1,纵向:1]，padding选择全零填充
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	
def forward(x,train,regularizer):
	#get_weight待优化参数conv1_w卷积核5*5,1通道，32核;32个核对应32个偏置conv1_b
	conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)
	conv1_b = get_bias([CONV1_KERNEL_NUM])
	#第一层卷积，并激活，池化
	conv1 = conv2d(x,conv1_w)
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
	pool1 = max_pool_2x2(relu1)
	
	#get_weight待优化参数conv2_w卷积核5*5,32通道，64核;64个核对应64个偏置conv2_b
	conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
	conv2_b = get_bias([CONV2_KERNEL_NUM])
	conv2 = conv2d(pool1,conv2_w)
	#第二层卷积，并激活，池化	
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
	pool2 = max_pool_2x2(relu2)
	
	#将第二层池化层的输出 pool2 矩阵转化为全连接层的输入格式即向量形式，从 list 中依次取出矩阵的长宽及深度，并求三者的乘积，得到矩阵被拉长后的长度nodes，转换成[BATCH_SIZE,nodes]
	pool_shape = pool2.get_shape().as_list()
	nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
	reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
	
	#输入至全链接网络，如果是训练阶段，则对该层输出使用dropout，也就是随机的将该层输出中的一半神经元置为无效，
	fc1_w = get_weight([nodes,FC_SIZE],regularizer)
	fc1_b = get_bias([FC_SIZE])
	fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
	if train:fc1 = tf.nn.dropout(fc1,0.5)
	
	fc2_w = get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
	fc2_b = get_bias([OUTPUT_NODE])
	y = tf.matmul(fc1,fc2_w)+fc2_b
	return y
