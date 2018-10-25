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
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# 將圖片數值從原本的 0~255 正規化成 0~1
x_train, x_test = x_train / 255, x_test / 255

# 將 Features 進行標準化與 Label 的 Onehot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 建立模型
model = tf.keras.models.Sequential()

# 建立卷積層與池化層
# 卷積層
model.add(tf.keras.layers.Conv2D(filters=16,
                                 kernel_size=(5, 5),
                                 padding='same',
                                 input_shape=(28, 28, 1),
                                 activation=tf.nn.relu))

# 池化層
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 卷積層
model.add(tf.keras.layers.Conv2D(filters=32,
                                 kernel_size=(5, 5),
                                 padding='same',
                                 input_shape=(28, 28, 1),
                                 activation='relu'))
# 池化層
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 丟失層
model.add(tf.keras.layers.Dropout(0.25))

# 建立神經網路
model.add(tf.keras.layers.Flatten())    # 扁平化(一維化)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    # 全連接層
model.add(tf.keras.layers.Dropout(0.5))  # 丟失率

# 最後建立輸出層, 共有 10 個神經元, 對應到 0~9 共 10 個數字.
# 並使用 softmax 激活函數 進行轉換
# (softmax 函數可以將神經元的輸出轉換成每一個數字的機率):
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 全連接層

# 定義模型訓練方式
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 開始訓練
model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=300)

# 驗證模型
loss, acc = model.evaluate(x_test, y_test)

print('Testing Accurakcy: ', str(acc))
