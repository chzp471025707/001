# filename:ts05.09.py # ckpt文件保存方法

import tensorflow as tf

# 1. 申明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
#申明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 需要在本python脚本文件下存在Saved_model目录
    # 否则提示错误 ValueError: Parent directory of Saved_model/model.ckpt doesn't exist, can't save.
    saver.save(sess, "Saved_model/model.ckpt")

# 2. 加载保存了两个变量和的模型
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(result)) # [3.]

# 3. 直接加载持久化的图
saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))) # [3.]

# 4. 变量重命名
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
saver = tf.train.Saver({"v1": v1, "v2": v2})