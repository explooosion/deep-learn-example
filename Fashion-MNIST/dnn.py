import tensorflow as tf
import os

# 關閉不必要的系統警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 關閉不必要的 Tensorflow 警告
tf.logging.set_verbosity(tf.logging.ERROR)

# 載入資料集
fashion_mnist = tf.keras.datasets.fashion_mnist

# 取出訓練與測試集資料
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 將圖片數值從原本的 0~255 正規化成 0~1
x_train, x_test = x_train / 255.0, x_test / 255.0

# 建立模型: 獨立寫法
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    # 扁平化(一維化)
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))    # 全連接層
model.add(tf.keras.layers.Dropout(0.5))  # 丟失率
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 全連接層

# 建立模型: 一次性的寫法
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),                             # 扁平化(一維化)
#     tf.keras.layers.Dense(512, activation=tf.nn.relu),     # 全連接層
#     tf.keras.layers.Dropout(0.5),                          # 丟失率
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)    # 全連接層
# ])

# optimizer=tf.train.AdamOptimizer(0.001)
# optimizer=tf.train.RMSPropOptimizer(0.01)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 載入訓練完的檢查點
# model.load_weights('./checkpoints/my_checkpoint')

# 開始訓練
model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=300)

# model.save('./my_model.h5')

# 儲存訓練完的檢查點
# model.save_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(x_test, y_test)

print('Testing Accuracy: ', str(acc))
