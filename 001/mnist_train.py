# 《TensorFlow实战Google深度学习框架》05 minist数字识别问题
# win10 Tensorflow1.0.1 python3.5.3
# CUDA v8.0 cudnn-8.0-windows10-x64-v5.1
# filename:mnist_train.py # 训练程序

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os

# 1. 定义神经网络结构相关的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "MNIST_model/" # 在当前目录下存在MNIST_model子文件夹
MODEL_NAME = "mnist_model"

# 2. 定义训练过程
def train(mnist):
    # 定义输入输出placeholder。
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

# 3. 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
'''
After 1 training step(s), loss on training batch is 2.79889.
After 1001 training step(s), loss on training batch is 0.224816.
After 2001 training step(s), loss on training batch is 0.154148.
After 3001 training step(s), loss on training batch is 0.138829.
After 4001 training step(s), loss on training batch is 0.116191.
After 5001 training step(s), loss on training batch is 0.114107.
After 6001 training step(s), loss on training batch is 0.0982979.
After 7001 training step(s), loss on training batch is 0.0957554.
After 8001 training step(s), loss on training batch is 0.0807435.
After 9001 training step(s), loss on training batch is 0.073724.
After 10001 training step(s), loss on training batch is 0.0664153.
After 11001 training step(s), loss on training batch is 0.0645105.
After 12001 training step(s), loss on training batch is 0.0572586.
After 13001 training step(s), loss on training batch is 0.0571209.
After 14001 training step(s), loss on training batch is 0.0620913.
After 15001 training step(s), loss on training batch is 0.0530401.
After 16001 training step(s), loss on training batch is 0.0473576.
After 17001 training step(s), loss on training batch is 0.046784.
After 18001 training step(s), loss on training batch is 0.0434549.
After 19001 training step(s), loss on training batch is 0.0432269.
After 20001 training step(s), loss on training batch is 0.0423441.
After 21001 training step(s), loss on training batch is 0.0418092.
After 22001 training step(s), loss on training batch is 0.0371407.
After 23001 training step(s), loss on training batch is 0.0368375.
After 24001 training step(s), loss on training batch is 0.0379352.
After 25001 training step(s), loss on training batch is 0.0358277.
After 26001 training step(s), loss on training batch is 0.034691.
After 27001 training step(s), loss on training batch is 0.0370471.
After 28001 training step(s), loss on training batch is 0.0342903.
After 29001 training step(s), loss on training batch is 0.0363797.
'''