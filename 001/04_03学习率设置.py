#1.学习率为1的时候，x在5和-5之间震荡
import tensorflow as tf
TRAIN_STEPS = 10  #必须用大写
LEARNING_RATE = 1 #必须用大学
x = tf.Variable(tf.constant(5,dtype = tf.float32), name = "x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAIN_STEPS):
        sess.run(train_op)
        x_value = sess.run(x)
        print("第 %s 次迭代后 x%s 输出是: %f." %(i+1, i+1, x_value))

#2.学习率为0.001的时候，下降速度过慢，在1001轮时才收敛到0.673971
TRAINING_STEPS = 1001
LEARNING_RATE = 0.001
x = tf.Variable(tf.constant(5,dtype = tf.float32), name = "x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i %100 ==0:
            x_value = sess.run(x)
            print("第 %s 次迭代后 x%s 输出是: %f."%(i+1,i+1 , x_value))

#3.使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得不错的收敛程度
TRAINING_STEPS = 201
global_step = tf.Variable(0)
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase = True)

x = tf.Variable(tf.constant(5,dtype = tf.float32),name = "x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y,global_step = global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 10 == 0 :
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print("第 %s 次迭代后 x%s 是: %f,学习率是 %f." % (i + 1, i + 1, x_value, LEARNING_RATE_value))