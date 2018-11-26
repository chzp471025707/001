# 《TensorFlow实战Google深度学习框架》05 minist数字识别问题
# win10 Tensorflow1.0.1 python3.5.3
# CUDA v8.0 cudnn-8.0-windows10-x64-v5.1
# filename:ts05.10.py # 滑动平均类的保存

import tensorflow as tf

# 1. 使用滑动平均
v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables():
    print(variables.name)
'''
v:0
'''
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables.name)
'''
v:0
v/ExponentialMovingAverage:0
'''
# 2. 保存滑动平均模型
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
    saver.save(sess, "Saved_model/model2.ckpt")
    print(sess.run([v, ema.average(v)]))
'''
[10.0, 0.099999905]
'''
# 3. 加载滑动平均模型
v = tf.Variable(0, dtype=tf.float32, name="v")

# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print(sess.run(v))
'''
0.0999999
'''