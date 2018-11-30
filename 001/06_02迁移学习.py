# 《TensorFlow实战Google深度学习框架》06 图像识别与卷积神经网络
# win10 Tensorflow1.0.1 python3.5.3
# CUDA v8.0 cudnn-8.0-windows10-x64-v5.1
# filename:ts06.03.py # 迁移学习

# 以下实验需要如下资源
# 源码及资源位置：git clone https://github.com/caicloud/tensorflow-tutorial.git
# 需要tensorflow-tutorial.git库中的flower_photos和inception_dec_2015
# tensorflow-tutorial\Deep_Learning_with_TensorFlow\datasets\flower_photos
# tensorflow-tutorial\Deep_Learning_with_TensorFlow\datasets\inception_dec_2015

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 1. 模型和样本路径的设置
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


MODEL_DIR = '../../datasets/inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

CACHE_DIR = '../../datasets/bottleneck'
INPUT_DATA = '../../datasets/flower_photos'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 2. 神经网络参数的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 3. 把样本中所有的图片列表并按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()

        # 初始化
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

# 4. 定义函数通过类别名称、所属数据集和图片编号获取一张图片的地址
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 5. 定义函数获取Inception-v3模型处理之后的特征向量的文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

# 6. 定义函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):

    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 7. 定义函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    if not os.path.exists(bottleneck_path):

        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)

        image_data = gfile.FastGFile(image_path, 'rb').read()

        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

# 8. 这个函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

# 9. 这个函数获取全部的测试数据，并计算正确率
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 10. 定义主函数
def main():
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    # 读取已经训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全链接层
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 训练过程。
        for i in range(STEPS):

            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, BATCH, validation_accuracy * 100))

        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    main()
'''
Step 0: Validation accuracy on random sampled 100 examples = 29.0%
Step 100: Validation accuracy on random sampled 100 examples = 83.0%
Step 200: Validation accuracy on random sampled 100 examples = 86.0%
Step 300: Validation accuracy on random sampled 100 examples = 89.0%
Step 400: Validation accuracy on random sampled 100 examples = 84.0%
Step 500: Validation accuracy on random sampled 100 examples = 91.0%
Step 600: Validation accuracy on random sampled 100 examples = 91.0%
Step 700: Validation accuracy on random sampled 100 examples = 91.0%
Step 800: Validation accuracy on random sampled 100 examples = 90.0%
Step 900: Validation accuracy on random sampled 100 examples = 84.0%
Step 1000: Validation accuracy on random sampled 100 examples = 87.0%
Step 1100: Validation accuracy on random sampled 100 examples = 84.0%
Step 1200: Validation accuracy on random sampled 100 examples = 94.0%
Step 1300: Validation accuracy on random sampled 100 examples = 87.0%
Step 1400: Validation accuracy on random sampled 100 examples = 95.0%
Step 1500: Validation accuracy on random sampled 100 examples = 90.0%
Step 1600: Validation accuracy on random sampled 100 examples = 94.0%
Step 1700: Validation accuracy on random sampled 100 examples = 91.0%
Step 1800: Validation accuracy on random sampled 100 examples = 88.0%
Step 1900: Validation accuracy on random sampled 100 examples = 91.0%
Step 2000: Validation accuracy on random sampled 100 examples = 86.0%
Step 2100: Validation accuracy on random sampled 100 examples = 91.0%
Step 2200: Validation accuracy on random sampled 100 examples = 93.0%
Step 2300: Validation accuracy on random sampled 100 examples = 95.0%
Step 2400: Validation accuracy on random sampled 100 examples = 91.0%
Step 2500: Validation accuracy on random sampled 100 examples = 95.0%
Step 2600: Validation accuracy on random sampled 100 examples = 87.0%
Step 2700: Validation accuracy on random sampled 100 examples = 95.0%
Step 2800: Validation accuracy on random sampled 100 examples = 93.0%
Step 2900: Validation accuracy on random sampled 100 examples = 95.0%
Step 3000: Validation accuracy on random sampled 100 examples = 97.0%
Step 3100: Validation accuracy on random sampled 100 examples = 90.0%
Step 3200: Validation accuracy on random sampled 100 examples = 95.0%
Step 3300: Validation accuracy on random sampled 100 examples = 97.0%
Step 3400: Validation accuracy on random sampled 100 examples = 91.0%
Step 3500: Validation accuracy on random sampled 100 examples = 98.0%
Step 3600: Validation accuracy on random sampled 100 examples = 92.0%
Step 3700: Validation accuracy on random sampled 100 examples = 92.0%
Step 3800: Validation accuracy on random sampled 100 examples = 96.0%
Step 3900: Validation accuracy on random sampled 100 examples = 95.0%
Step 3999: Validation accuracy on random sampled 100 examples = 99.0%
Final test accuracy = 92.7%
'''
