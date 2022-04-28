import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

eye_cascade = cv2.CascadeClassifier("venv\Lib\site-packages\cv2\data\haarcascade_eye.xml")


def leia_silmad(img):
    # TODO siin vaja numbreid alla tõmmata, ntks 1.5 ja 5

    scaler = 1.5
    silmad = eye_cascade.detectMultiScale(img, scaler, 5)
    for (x, y, w, h) in silmad:
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)


    cv2.imshow('img', img)
    cv2.waitKey()
    return img
