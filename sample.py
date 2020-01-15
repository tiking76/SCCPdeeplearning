#!/usr/bin/python
# coding: utf-8

import cv2
import numpy as np
from chainer import Chain, serializers
import chainer.functions  as F
import chainer.links as L

# 多層パーセプトロンモデルの設定
class MyMLP(Chain):
    # 入力784、中間層500、出力10次元
    def __init__(self, n_in=784, n_units=500, n_out=10):
        super(MyMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )
    # ニューラルネットの構造
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

def preprocessing(img):
    img = img[190:290,270:370]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (28, 28))
    res, img = cv2.threshold(img, 70 , 255, cv2.THRESH_BINARY)
    img = 255 - img
    img = img.astype(np.float32)
    cv2.imwrite("img.jpg",img)
    img /= 255
    img = np.array(img).reshape(1,784)
    return img

def main():
    # 学習済みモデルの読み込み
    net = MyMLP()
    serializers.load_npz('my.model2', net)
    #Webカメラの映像表示
    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
            raise("IO Error")
    while True:
        #Webカメラの映像とりこみ
        ret, image = capture.read()
        if ret == False:
            continue
        #Webカメラの映像表示
        cv2.rectangle(image,(270,190),(370,290),(0,0,255),3)
        cv2.imshow("Capture", image)
        k = cv2.waitKey(10)
        #Eキーで処理実行
        if k == 101:
            img = preprocessing(image)
            num = net(img)
            #cv2.imwrite("img.jpg",img)
            print(num.data)
            print(np.argmax(num.data))
        #ESCキーでキャプチャー画面を閉じる
        if  k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
