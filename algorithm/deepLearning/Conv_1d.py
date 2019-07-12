import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.python.framework import ops
from sklearn.preprocessing import OneHotEncoder

iris = datasets.load_iris()
#print(iris.data)
#print(iris.target)
ops.reset_default_graph()
conv_size  =2
stride_size = 1
maxpool_size = 2

# Create graph session 创建初始图结构
ops.reset_default_graph()
sess = tf.Session()
#placeholder

# --------Convolution--------
def conv_layer_1d(input_1d, my_filter, stride):
    # TensorFlow's 'conv2d()' function only works with 4D arrays:
    # [batch, height, width, channels], we have 1 batch, and
    # width = 1, but height = the length of the input, and 1 channel.
    # So next we create the 4D array by inserting dimension 1's.
    # 关于数据维度的处理十分关键，因为tensorflow中卷积操作只支持四维的张量，
    # 所以要人为的把数据补充为4维数据[1,1,25,1]
    #input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_1d, 1)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution with stride = 1, if we wanted to increase the stride,
    # to say '2', then strides=[1,1,2,1]
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 1, stride, 1], padding="VALID")
    # Get rid of extra dimensions 去掉多余的层数，只保留数字
    conv_output_1d = tf.squeeze(convolution_output)
    return (conv_output_1d)

# --------Activation--------
def activation(input_1d):
    return (tf.nn.relu(input_1d))


# --------Fully Connected--------
def fully_connected(input_layer,num_outputs):

    # First we find the needed shape of the multiplication weight matrix:
    # The dimension will be (length of input) by (num_outputs)

    weight_shape = tf.squeeze(tf.stack([[tf.shape(input_layer)[0]], [tf.shape(input_layer)[1]],[num_outputs]]))

    #weight_shape = [num_outputs]
    # squeeze函数用于去掉维度为1的维度。保留数据。
    # Initialize such weight
    # 初始化weight
    input_flat = tf.reshape(input_layer, [-1, 2])
    weight = tf.random_normal([2,3], stddev=0.1)
    #weight = tf.random_normal(weight_shape, stddev=0.1)
    #print(sess.run(weight))
    # Initialize the bias
    # 初始化bias
    bias = tf.random_normal(shape=[num_outputs])
    # Make the 1D input array into a 2D array for matrix multiplication
    # 将一维的数组添加一维成为2维数组
    input_layer_2d = tf.expand_dims(input_layer, 2)
    # Perform the matrix multiplication and add the bias

    full_output = tf.add(tf.matmul(input_flat, weight), bias)
    softmax_output = tf.nn.softmax(full_output)
    # Get rid of extra dimensions
    # 去掉多余的维度只保留数据
    full_output_1d = tf.squeeze(full_output)
    return (softmax_output)

# --------Max Pool--------
def max_pool(input_1d, width, stride):
    # Just like 'conv2d()' above, max_pool() works with 4D arrays.
    # [batch_size=1, width=1, height=num_input, channels=1]
    # 因为在处理卷积层的结果时，使用squeeze函数对结果输出进行降维，所以此处要将最大池化层的维度提升为4维
    #input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_1d, 1)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform the max pooling with strides = [1,1,1,1]
    # If we wanted to increase the stride on our data dimension, say by
    # a factor of '2', we put strides = [1, 1, 2, 1]
    # We will also need to specify the width of the max-window ('width')

    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
                                 strides=[1, 1, stride, 1],
                                 padding='VALID')
    # Get rid of extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return (pool_output_1d)

#输入层

tf_X = tf.placeholder(tf.float32,[None,4])
tf_Y = tf.placeholder(tf.float32,[None,3])

print("---xiaoyao--->")
ex_2d_y = iris.target.reshape(-1,1)

feed_Y = OneHotEncoder().fit_transform(ex_2d_y).todense()
feed_dict = {tf_X: iris.data,tf_Y:feed_Y}

conv_filter_w1 = tf.Variable(tf.random_normal([1, 3, 1, 10]))
conv_filter_b1 =  tf.Variable(tf.random_normal([10]))

my_filter = tf.Variable(tf.random_normal(shape=[1, conv_size, 1, 1]))
conv_filter_b1 = tf.Variable(tf.random_normal([1]))

my_convolution_output = conv_layer_1d(tf_X, my_filter, stride=stride_size)
my_activation_output = activation(my_convolution_output)
my_maxpool_output = max_pool(my_activation_output, width=maxpool_size, stride=stride_size)

my_full_output = fully_connected(my_maxpool_output,3)

init = tf.global_variables_initializer()




print('>>>> 1D Data <<<<')

# Convolution Output
#print('Input = array of length %d'%(tf_X.shape.as_list()[0]))  # 25
#print('Convolution w/ filter, length = %d, stride size = %d, results in an array of length %d:'%
      #(conv_size, stride_size, my_convolution_output.shape.as_list()[0]))  # 21
#print(sess.run(my_convolution_output, feed_dict=feed_dict))

#print(sess.run(my_full_output, feed_dict=feed_dict))
loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(my_full_output,1e-11,1.0)))
sess.run(init)
#train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
print(sess.run(loss, feed_dict=feed_dict))
"""
# Activation Output
print('\nInput = above array of length %d'%(my_convolution_output.shape.as_list()[0]))  # 21
print('ReLU element wise returns an array of length %d:'%(my_activation_output.shape.as_list()[0]))  # 21
print(sess.run(my_activation_output, feed_dict=feed_dict))

# Max Pool Output
print('\nInput = above array of length %d'%(my_activation_output.shape.as_list()[0]))  # 21
print('MaxPool, window length = %d, stride size = %d, results in the array of length %d'%
      (maxpool_size, stride_size, my_maxpool_output.shape.as_list()[0]))  # 17
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output
print('\nInput = above array of length %d'%(my_maxpool_output.shape.as_list()[0]))  # 17
print('Fully connected layer on all 4 rows with %d outputs:'%
      (my_full_output.shape.as_list()[0]))  # 5
print(sess.run(my_full_output, feed_dict=feed_dict))
"""












"""
# 定义向量
input = np.array(np.arange(1, 1+10*8*16).reshape([10, 8, 16]), dtype=np.float32)
print(input.shape)

# 卷积核
kernel = np.array(np.arange(1, 1+5*16*3), dtype=np.float32).reshape([5, 16, 3])
print(kernel.shape)

# 进行conv1d卷积
conv1out = tf.nn.conv1d(input, kernel, 1, 'VALID')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 输出卷积值
    print(sess.run(conv1out).shape)
    print(sess.run(conv1out))
"""