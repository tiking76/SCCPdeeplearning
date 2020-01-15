from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# mnistの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# モデル構築
model = Sequential()
model.add(Dense(512, input_dim=(784)))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

# 学習のさせ方の指定
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# データのフォーマット修正
# 画像
X_train = X_train.reshape(60000, 784)/255
X_test = X_test.reshape(10000, 784)/255

# ラベル
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 学習させる
hist = model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=10, validation_split=0.1)

# 正答率
score = model.evaluate(X_test, y_test, verbose=1)
print("正解率(acc)：", score[1])

# モデルの保存
model.save("MNIST.h5")
