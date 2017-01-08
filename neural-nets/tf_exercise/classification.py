import tensorflow as tf                                                                                                                                                         
import numpy as np 
'''                                                                                                                                                                             
Classification                                                                                                                                                                  
                                                                                                                                                                                
X - [ 0.2, 0.1, 0.6 ]                                                                                                                                                           
Y - 2 (argmax X)                                                                                                                                                                
                                                                                                                                                                                
'''                                                                                                                                                                             
                                                                                                                                                                                
n = 1000                                                                                                                                                                        
x_train = [ np.random.randn(10) for i in range(n) ] # [1000x10]                                                                                                                 
y_train = [ np.argmax(xi) for xi in x_train ] # [1000x1] 

x_ = tf.placeholder(shape=(1, 10), name='x', dtype= tf.float32)
y_ = tf.placeholder(shape=(1,), name='y', dtype= tf.int32)
w  = tf.get_variable(name='w', shape=1, initializer = 
        tf.contrib.layers.xavier_initializer())
b  = tf.Variable(tf.zeros(1), name='b')

y = tf.mul(w,x_) + b


# training
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

'''
y  -> logits => n (1, 10) = 10 => ndims = 1
y_ -> labels => n-1 (1,) = 1 => ndims = 1
'''

optim = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optim.minimize(loss)


# session run
with tf.Session() as sess:
    # W, b
    sess.run(tf.initialize_all_variables()) 
    for i in range(10):
        for xi, yi in zip(x_train, y_train):

            _, loss_val = sess.run( [train_op, loss], 
                    feed_dict = {
                        x_ : np.array(xi).reshape([1,10]),
                        y_ : np.array(yi).reshape([1,])
                        })
        print('At {}, Loss : {}'.format(i, loss_val))

    testx = [ 0.01 * item for item in  list(range(10)) ]
    from random import shuffle
    shuffle(testx)
    print('test x : ', testx)

    op = sess.run([y], feed_dict = { x_ : np.array(testx).reshape([1,10]) } )
print('max index : ', np.argmax(op))        