# 《TensorFlow实战Google深度学习框架》03 TensorFlow入门
# win10 Tensorflow1.0.1 python3.5.3
# CUDA v8.0 cudnn-8.0-windows10-x64-v5.1
# filename:ts03.01.py

'''
tensorflow的计算模型：计算图--tf.Graph
tensorflow的数据模型：张量--tf.Tensor
tensorflow的运行模型：会话--tf.Session
tensorflow可视化工具：TensorBoard
通过集合管理资源：tf.add_to_collection、tf.get_collection
Tensor主要三个属性：名字(name)、维度(shape)、类型(type)
会话Session需要关闭才能释放资源，通过Python的上下文管理器 with ，可以自动释放资源
'''
import tensorflow as tf;
print("tensorflow version:",tf.VERSION);

def func00_def():

    a = tf.Variable([1.0,2.0],name = "a");
    b = tf.Variable([2.0,3.0],name = "b");
    #result = tf.placeholder(tf.float32, shape = (1,2)), name = "result");
    g = tf.Graph(); # 图
    with tf.Session() as sess:
        with g.device('cpu:0'):
            init_op = tf.global_variables_initializer();
            sess.run(init_op);
            result = a+b;
            print(sess.run(result));

def func01_constant():
    w1 = tf.Variable(tf.random_normal([2,3],stddev = 1));
    w2 = tf.Variable(tf.random_normal([3,1],stddev = 1));

    x = tf.constant([[0.7,0.9]]);
    a = tf.matmul(x,w1);
    y = tf.matmul(a,w2);

    sess = tf.Session();
    init_op = tf.global_variables_initializer();
    sess.run(init_op);
    print(sess.run(y));

def func02_placeholder():
    w1 = tf.Variable(tf.random_normal([2,3],stddev = 1));
    w2 = tf.Variable(tf.random_normal([3,1],stddev = 1));

    x = tf.placeholder(tf.float32,shape=(1,2), name = "input");
    a = tf.matmul(x,w1);
    y = tf.matmul(a,w2);

    sess = tf.Session();
    init_op = tf.global_variables_initializer();

    sess.run(init_op);

    #placeholder的赋值
    print(sess.run(y,feed_dict={x:[[0.7,0.9]]}));

func00_def();
func01_constant();
func02_placeholder();