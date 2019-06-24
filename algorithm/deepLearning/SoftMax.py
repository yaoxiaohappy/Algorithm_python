import tensorflow as tf
sess = tf.InteractiveSession() # see the answers above :)
x = [[1.0,2.0,2.0],[1.0,1.0,1.0]]    # a 2D matrix as input to softmax
y = tf.nn.softmax(x)           # this is the softmax function
                               # you can have anything you like here
#u = y.eval()
print(sess.run(y))

"""
[[0.15536241 0.42231882 0.42231882]
 [0.33333334 0.33333334 0.33333334]]
"""