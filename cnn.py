import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 關閉不必要的 System Warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 關閉不必要的 Tensorflow Warning
tf.logging.set_verbosity(tf.logging.ERROR)

# 數字 1 ~ 10 資料集
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

### 顯示資料筆數 ###

# Train 55,000
print(len(mnist.train.labels))
# Validation
print(len(mnist.validation.labels))
# Test  10,000
print(len(mnist.test.labels))

### 建立常用函式 ###

# 計算準確率


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

# 計算權重


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

# 計算偏差


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

# 二維


def conv2d(x, W):
    print('conv2d', tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 最大池化


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


### 建立輸入型別 ###

xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

## conv1 layer ## ================== ##

# filter 的 size 是 5x5
# 因為是圖片是黑白色所以 channel 是 1
# 使用 32 個 filter 去抽取 feature map
W_conv1 = weight_variable([5, 5, 1, 32])

## pool1 layer ##

b_conv1 = bias_variable([32])

# Combine

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) +
                     b_conv1)  # output size 28x28x32

h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

## conv2 layer ## ================== ##

# patch 5x5, in channel size 32, out size 64
W_conv2 = weight_variable([5, 5, 32, 64])

## pool2 layer ##

b_conv2 = bias_variable([64])
# Combine

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
                     b_conv2)  # output size 14x14x64

h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

## fc1 layer ##

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7,7,64]  => [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## output layer ##

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

## 定義訓練方法 ##

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

## 開始訓練 ##

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={
             xs: batch_xs, ys: batch_ys, keep_prob: 0.5})  # 0.5 dropout
    if i % 100 == 0:
        print("Iter:" + str(i) + ", Accuracy:", compute_accuracy(
            mnist.test.images, mnist.test.labels))
