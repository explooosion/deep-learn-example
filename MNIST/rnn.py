import tensorflow as tf
import tensorflow.keras.utils as np_utils
import os

# 關閉不必要的 System Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 關閉不必要的 Tensorflow Warning
tf.logging.set_verbosity(tf.logging.ERROR)

# 數字 1 ~ 10 資料集
mnist = tf.keras.datasets.mnist

# 取出訓練與測試集資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 資料轉換
x_train = x_train.reshape(x_train.shape[0], -1, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, -1, 1).astype('float32')

# 將圖片數值從原本的 0~255 正規化成 0~1
x_train, x_test = x_train / 255, x_test / 255

# 將 Features 進行標準化與 Label 的 Onehot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(100,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(
                                        stddev=0.001),
                                    recurrent_initializer=tf.keras.initializers.Identity(
                                        gain=1.0),
                                    activation='relu',
                                    input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))

rmsprop = tf.keras.optimizers.RMSprop(lr=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

# 開始訓練
model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=300)

# 驗證模型
loss, acc = model.evaluate(x_test, y_test)
print('Testing Accurakcy: ', str(acc))
