# filename:ts05.08.py # 变量管理(命名空间)

import tensorflow as tf

# 1. 在名为foo的命名空间内创建名字为V的变量。
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
#因为在命名空间foo中已经存在名字为V的变量，所有下面的代码将会报错：
#with tf.variable_scope("foo"):
#    v1 = tf.get_variable("v", [1])
#print(v == v1)

#在生成上下文管理器时，将参数reuse设置为 True.这样tf.get_variable函数将直接获取
#已经申明的变量。
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
print(v == v1) # 输出为True，代表v，v1表带的是相同的tensorflow中变量

# 2. 嵌套上下文管理器中的reuse参数的使用

with tf.variable_scope("root"):
    #可以通过tf.get_variable_scppe().reuse函数来获取当前上下文管理器中reuse参数的取值
    print(tf.get_variable_scope().reuse)          # 输出False，即最外层reuse是False
    with tf.variable_scope("foo", reuse=True):    # 新建一个嵌套的上下文管理器，并指定reuse为True
        print(tf.get_variable_scope().reuse)      # 输出True
        with tf.variable_scope("bar"):            # 新建一个嵌套的上下文管理器但不指定reuse，此时
                                                  # reuse的取值会和外层一层保持一致
            print(tf.get_variable_scope().reuse)  # 输出True
    print(tf.get_variable_scope().reuse)          # 输出False。退出reuse设置

# 3. 通过variable_scope来管理变量命名空间
v1 = tf.get_variable("v", [1])
#输出  v:0   “v”为变量的名称，“：0”表示这个变量是生成变量这个运算的第一个结果
print(v1.name)

with tf.variable_scope("foo", reuse=True):
    v2 = tf.get_variable("v", [1])
    # 输出 foo/v:0   在tf.variable_scope中创建的变量，名称前面会加入
    #命名空间的名称，并通过/来分隔命名空间的名称和变量的名称
print(v2.name)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        # 输出  foo/bar/v:0  命名空间可以嵌套，同时变量的名称也会加入
        #所有命名空间的名称作为前缀
        print(v3.name)

v4 = tf.get_variable("v1", [1])
# 输出 v1:0
print(v4.name)

# 4. 我们可以通过变量的名称来获取变量
with tf.variable_scope("",reuse=True):
    # 可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量。比如这里
    # 通过指定名称foo/bar/v来获取在命名空间foo/bar/中创建的变量
    v5 = tf.get_variable("foo/bar/v", [1])
    print(v5 == v3)                 #输出 True
    v6 = tf.get_variable("v1", [1])
    print(v6 == v4)                 #输出 True