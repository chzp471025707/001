# 1. 生成模拟数据集
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = []
label = []
np.random.seed(0)

# 以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
for i in range(150):
    x1 = np.random.uniform(-1,1) #x1（横坐标）的取值范围，均匀分布
    x2 = np.random.uniform(0,2)  #x2（纵坐标）的取值范围，均匀分布
    if x1**2 + x2**2 <= 1:
        #在data数组里最后添加坐标（x1，x2）
        data.append([np.random.normal(x1, 0.1),np.random.normal(x2,0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)

# -1就是让系统根据元素数和已知行或列推算出剩下的列或行，-1就是模糊控制，
# （-1,2）就是固定两列，行不知道
data = np.hstack(data).reshape(-1,2) #-1表示系统自动识别行数，2表示有2列
label = np.hstack(label).reshape(-1, 1)
plt.scatter(data[:,0], data[:,1], c=label,
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.show()

# 2. 定义一个获取权重的函数，自动加入正则项到损失函数并返回权值大小
def get_weight(shape, lambda1):
    # 生成一个变量
    #tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值
    #var相代表权值，随机生成shape（输入与输出）个正态分布的数
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)

    # tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
    #add_to_collection函数将这个新生成变量的L2正则化损失项加入集合
    #这个函数的第一个参数‘losses’是集合的名字，第二个参数是要加入这个集合的内容
    #对var进行L2正则化处理，下文会提到lambdal=0.003
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    # 返回生成的变量
    return var

# 3. 定义神经网络
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
sample_size = len(data)

# 每层节点的个数
layer_dimension = [2,10,5,3,1]
#神经网路的层数
n_layers = len(layer_dimension)
#这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
#当前层节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    # tf.get_collection：从一个集合中取出全部变量，是一个列表
    weight = get_weight([in_dimension, out_dimension], 0.003)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用elu激活函数
    cur_layer = tf.nn.elu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]
#与cur_layer = x 对应，这个变量维护前向传播时最深层的节点，开始的时候就是输入层
y= cur_layer

# 损失函数的定义。
#在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了图上的集合，
#这里只需要计算刻画模型在训练数据上表现的损失函数
#mse_loss=（损失函数全部相加（（真实的分类值-预测的分类值）的平方））/样本个数
mse_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / sample_size
#将均方差损失函数加入到损失集合
tf.add_to_collection('losses', mse_loss)
# tf.add_n：把一个列表的东西都依次加起来
loss = tf.add_n(tf.get_collection('losses'))

# 4. 训练不带正则项的损失函数mse_loss
# 定义训练的目标函数mse_loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(mse_loss)
TRAINING_STEPS = 20000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        sess.run(train_op, feed_dict={x: data, y_: label})
        if i % 2000 == 0:
            print("After %d steps, mse_loss: %f" % (i,sess.run(mse_loss,
                                                               feed_dict={x: data, y_: label})))

    # 画出训练后的分割曲线
    # mgrid函数产生两个240×241的数组：-1.2到1.2每隔0.01取一个数共240个
    xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)

plt.scatter(data[:,0], data[:,1], c=label,
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()

# 5. 训练带正则项的损失函数loss
# 定义训练的目标函数loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
TRAINING_STEPS = 20000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        sess.run(train_op, feed_dict={x: data, y_: label})
        if i % 2000 == 0:
            print("After %d steps, loss: %f" % (i, sess.run(loss, feed_dict={x: data, y_: label})))

    # 画出训练后的分割曲线
    xx, yy = np.mgrid[-1:1:.01, 0:2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)

plt.scatter(data[:,0], data[:,1], c=label,
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()