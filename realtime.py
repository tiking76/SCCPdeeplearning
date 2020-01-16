"""This is a test program."""
import cv2
import numpy as np
from keras.models import load_model


# ここの数字は0はwebカメラ使用時、それ以外は1オリジンでやっていく。失敗したときは-1が帰ってくる
cap = cv2.VideoCapture(0)
model = load_model("MNIST.h5")

while(True):

    Xt = []
    Yt = []

    ret, frame = cap.read()

    h, w, _ = frame.shape[:3]

    w_center = w//2
    h_center = h//2

    # 四角のレンダリング
    cv2.rectangle(frame, (int(w_center - 71), int(h_center - 71)),
                  (int(w_center + 71), int(h_center + 71)), (255, 0, 0))

    # カメラ画像の整形
    im = frame[h_center - 70:h_center + 70, w_center - 70:w_center + 70]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)
    th = cv2.GaussianBlur(th, (9, 9), 0)
    th = cv2.resize(th, (28, 28), cv2.INTER_CUBIC)

    Xt.append(th)
    Xt = np.array(Xt)/255

    result = model.predict_classes(Xt)
    for i in range(10):
        r = round(result[0, i], 2)
        Yt.append([i, r])
        Yt = sorted(Yt, key=lambda x: (x[1]))

    cv2.putText(frame, "1:"+str(Yt[9]), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "2:"+str(Yt[8]), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "3:"+str(Yt[7]), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xFF
    prop_val = cv2.getWindowProperty("frame", cv2.WND_PROP_ASPECT_RATIO)

    if k == ord("q") or (prop_val < 0):
        break

cap.release()
cv2.destroyAllWindows()
