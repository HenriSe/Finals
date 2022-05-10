import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

# vajalik fail kausta lisamiseks
face_cascade = cv2.CascadeClassifier("venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

def leia(img):
    # etteantud fail otsitava leidmiseks läbi mudelite
    näod = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in näod:
        rgb = (255, 0, 0)
        joone_paksus = 2
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), rgb, joone_paksus)
        näod = img[y:y + h, x:x + w]  # joonistab ristküliku
        näod = cv2.resize(näod, (255, 255))  # oleneb mis suurust vaja parasjagu

    # piltide vaatamiseks
    #cv2.imshow('img', img)
    #cv2.waitKey()
    return näod

