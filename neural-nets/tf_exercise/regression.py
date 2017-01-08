import numpy as np 
import tensorflow as tf

'''
    N training examples (X,Y)
    Y = f(X)
    Y = f(X; theta)
    # linear model
    Y = WX + b
    (x,y) => (1.43, 3.6)
    learnable parameters : W, b
'''

n = 1000
x_train = np.random.randn(n) 
y_train = (0.1*x_train)+ 0.3 # y=0.1x+0.3 -> W=0.1 , b=0.3
#add noise 
y_train = [ yi+ np.random.normal(0., 0.03) for yi in y_train]


#build a linear model 
x_ = tf.placeholder(shape=None, name='x', dtype=tf.float32)
y_ = tf.placeholder(shape=None, name='y', dtype=tf.float32)

w = tf.get_variable(name='w',shape=1,initializer= tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros(1),name='b')

#y=wx+b
y = tf.mul(w,x_)+b

loss = tf.square( y - y_)
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optim.minimize(loss) 

# session run
with tf.Session() as sess:
    # W, b
    sess.run(tf.initialize_all_variables())
    for i in range(10):
		for xi, yi in zip(x_train, y_train):
			_, loss_val, wval, bval = sess.run([train_op,loss,w,b],
				feed_dict ={
					x_ : xi,
					y_ : yi
				})
		print('Loss: {}; y = {}x + {}'.format(loss_val[0],wval[0],bval[0]))