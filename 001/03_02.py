import tensorflow as tf
#通过numpy工具包生成模拟数据集
from numpy.random import RandomState

# 1. 定义神经网络的参数，输入和输出节点
#训练数据batch的大小(一次训练模型，投入的样例数,本该一次性投入所有样例，为了防止内存泄漏设定batch)
batch_size = 16
#产生随机变量，2行3列，方差为1，种子为1
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 数据是float32型，数据形状，行不定，列为2
#使用none作用是使用不大的batch大小。数据小时方便计算，大的话可能会导致内存溢出
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 使用placeholder的作用是把指定类型数据进行存储，在图计算时再把数据加入，
#因为使用常量的话，需要在图里添加节点，迭代百万上亿次任务效率会很低
#使用这种方法时，对输入要进行类型的约束
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 2. 定义前向传播过程，损失函数及反向传播算法
a = tf.matmul(x, w1) #把x与w1进行乘法运算
y = tf.matmul(a, w2)
#定义损失函数（交叉熵）来刻画预测值与真实值的差距
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
#定义反向传播的优化方法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 3. 通过随机数生成一个模拟数据集
rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 4. 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("\n")

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选择batch_size个样本进行训练
        start = (i * batch_size) % 16
        end = (i * batch_size) % 16 + batch_size
        #使用选择的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        #每隔1000次迭代计算所有数据上的交叉熵并输出
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的结果。
    #结果越小，说明预测的结果与实际值之间的差距越小
    print("\n")
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))