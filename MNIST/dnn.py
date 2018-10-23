import tensorflow as tf
import os

# 關閉不必要的系統警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 關閉不必要的 Tensorflow 警告
tf.logging.set_verbosity(tf.logging.ERROR)

# 載入資料集
mnist = tf.keras.datasets.mnist

# 取出訓練與測試集資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),                             # 扁平化(一維化)
    tf.keras.layers.Dense(512, activation=tf.nn.relu),     # 全連接層
    tf.keras.layers.Dropout(0.5),                          # 丟失率
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)    # 全連接層
])

# optimizer=tf.train.AdamOptimizer(0.001)
# optimizer=tf.train.RMSPropOptimizer(0.01)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 載入訓練完的檢查點
# model.load_weights('./checkpoints/my_checkpoint')

# 開始訓練
model.fit(x_train, y_train, epochs=5)
# model.save('./my_model.h5')

# 儲存訓練完的檢查點
# model.save_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(x_test, y_test)

print('Testing Accuracy: ', str(acc))
