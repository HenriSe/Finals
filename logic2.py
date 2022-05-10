# import keras.layers
# from keras.callbacks import TensorBoard
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# from keras.models import Sequential
import cv2 as cv2

import detect_eyes_2
import detect_faces_2
import detect_mouth_2
import load_images_from_folders

datadir = r'C:\Users\h.seppel\OneDrive - Playtech\Desktop\proge\python\loputoo\tester pildid'

img_size = 224
kaustad = ["fake", "real"]
andmed = load_images_from_folders.hangi_andmed(datadir, img_size, kaustad[0], kaustad[1])

faces_train = []
faces_val = []
suud_train = []
suud_val = []
eyes_train = []
eyes_val = []

for fn, sildid in andmed:
    fn = detect_faces_2.leia(fn)
    faces_train.append(fn)
    faces_val.append(sildid)


for fn, sildid in andmed:
    fn = detect_mouth_2.leia_suu(fn)
    suud_train.append(fn)
    suud_val.append(sildid)

for fn, sildid in andmed:
    fn = detect_eyes_2.leia_silmad(fn)
    eyes_train.append(fn)
    eyes_val.append(sildid)

print("\nTreening nägude len: ", len(andmed))
print("Nägude array pikkus: ", len(faces_train))
print("Suude array pikkus: ", len(suud_train))
print("Eyes array pikkus: ", len(eyes_train))

array = eyes_train

for pilt in range(len(array)):
    # cv2.imshow('img', faces_train[pilt])
    try:
        cv2.imshow('img', array[pilt])
        cv2.waitKey()
    except Exception as e:
        pass


# TODO tekitan kolm arrayd
# TODO igale ühele oma mudel ja oma saved model

