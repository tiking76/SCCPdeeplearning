"""This is a test program."""
import cv2

IMG = cv2.imread("sample.jpg", 1)
cv2.rectangle(IMG, (50, 50), (100, 100), (255, 0, 0))
IMG = IMG[50:150, 50:150]
cv2.imshow("Color", IMG)
cv2.imwrite("result.png", IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()


