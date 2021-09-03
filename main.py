import tensorflow as tf
import numpy as np
import time, datetime
x=tf.Variable([[1,2,3],[4,5,6]])
print(x)
y=tf.expand_dims(x,-1)
print(y)
g=tf.square(y)
print(g)
w = tf.random.uniform(shape=[2, 3,3],minval=1, maxval=5, dtype=tf.int32)
print(w)
z=tf.cast(tf.greater_equal(w,tf.cast(2,tf.int32)),tf.int32)
print(z)
print(g*z)
print('********************************************************')
print(tf.linalg.trace(g*z))
