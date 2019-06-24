import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input = tf.Variable(tf.random_normal([1,3,3,1]))
filter = tf.Variable(tf.random_normal([1,1,1,1]))
conv_2d1 = tf.nn.conv2d(input,filter,[1,1,1,1],"VALID")
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(input))
    print("######################################")
    print(sess.run(filter))
    print("######################################")
    print(sess.run(conv_2d1))



