import tensorflow as tf 
import numpy as np
#c=a*b

#build graph
a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
c = tf.mul(a,b)


sess = tf.Session()

c_value = sess.run(c, feed_dict={
	a: [[1,2],[3,4]],
	b: [[3,4],[5,6]]
	})
print('\n >> {}'.format(c_value))