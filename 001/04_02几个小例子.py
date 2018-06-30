import tensorflow as tf

sess = tf.Session();
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
# y_代表正确结果，y代表预测结果
# tf.clip_by_value确保不会出现log0或者大于1的概率出现
v = tf.constant([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
print(sess.run(tf.clip_by_value(v, 2.5, 4.5)))
# 输出[[2.5 2.5 3.][4. 4.5 4.5]]

v = tf.constant([1.0, 2.0, 3.0])
print(sess.run(tf.log(v)))
# 输出 [ 0.          0.69314718  1.09861231]

# 交叉熵中"*"代表对应位置的元素相乘，而不是矩阵相乘
v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print(sess.run((v1 * v2)))
print(sess.run(tf.matmul(v1, v2)))
# 输出：[[5. 12.][21. 32.]]
# [[19. 22.][43. 50.]]

# 交叉熵得到一个nXm的矩阵，n=batch_size，m=分类数量

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(sess.run(tf.reduce_mean(v)))
# 输出 3.5

# 带softmax的交叉熵损失函数，y 神经网络输出结果，y_标准答案
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
# 对于只有一个正确答案的分类问题
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)

# 均方差损失函数，解决回归问题，主要是对具体数值的预测
# mse = tf.reduce_mean(tf.square(y_ - y))

# 自定义损失函数
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * a, (y_ - y) * b))
v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])
print(sess.run(tf.greater(v1, v2))) # [False False  True  True]
print(sess.run(tf.where(tf.greater(v1, v2), v1, v2))) # [ 4.  3.  3.  4.]

sess.close();