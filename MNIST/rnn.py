# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 獲取MNIST數據Ｆ
mnist = input_data.read_data_sets(r"./MNIST_data", one_hot=True)

# 設置RNN結構
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 150

# 輸入向量和輸出向量
_inputs = tf.placeholder(
    tf.float32, shape=[None, time_steps, element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='inputs')

# 利用TensorFlow的内置函數BasicRNNCell, dynamic_rnn來建構RNN的基本模型
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)
Wl = tf.Variable(tf.truncated_normal(
    [hidden_layer_size, num_classes], mean=0, stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))


def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl


# 取輸出的向量outputs中的最後一個向量為最终輸出
last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)

# 定義損失函數並用RMSPropOptimizer優化
softmax = tf.nn.softmax_cross_entropy_with_logits(
    logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# 統計準確率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# 測試集
test_data = mnist.test.images[:batch_size].reshape(
    (-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

# 每次訓練batch_size張圖片，一共訓練3000次
for i in range(3001):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, element_size))
    sess.run(train_step, feed_dict={_inputs: batch_x, y: batch_y})
    if i % 100 == 0:
        loss = sess.run(cross_entropy, feed_dict={
                        _inputs: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={_inputs: batch_x, y: batch_y})
        print("Iter " + str(i) + ", Minibatch Loss= " +
              "{:.6f}".format(loss) + ", Training Accuracy= " +
              "{:.5f}".format(acc))

# 在測試集上的準確率
print("Testing Accuracy:", sess.run(
    accuracy, feed_dict={_inputs: test_data, y: test_label}))
