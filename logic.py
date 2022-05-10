import random

import keras.layers
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
import cv2 as cv2


import andmete_hankimine
import leia_näod
import leia_suu

datadir = r'C:\Henri\Desktop\loputoo\tester_pildid\smiles\train'

img_size = 224
kaustad = ["fake_smile", "real_smile"]
andmed = andmete_hankimine.hangi_andmed(datadir, img_size, kaustad[0], kaustad[1])

# mudime andmed mudeli jaoks valmis
random.shuffle(andmed)
faces_train = []
faces_val = []
suud_train = []
suud_val = []
silmad_train = []
silmnad_val = []

for fn, sildid in andmed:
    fn = leia_näod.leia(fn)
    faces_train.append(fn)
    faces_val.append(sildid)

"""
for fn, sildid in andmed:
    fn = leia_suu.leia_suu(fn)
    suud_train.append(fn)
    suud_val.append(sildid)
"""
#faces_train = np.array(faces_train).reshape(-1, img_size, img_size, 1)
#faces_val = np.array(faces_val)

print("\nTreening nägude len: ", len(andmed))
print("Nägude array pikkus: ", len(faces_train))
print("Suude array pikkus: ", len(suud_train))

# nägu
#cv2.imshow('img2', suud_train[0])
cv2.imshow('img', faces_train[0])
cv2.waitKey()



# TODO kas on ikka see vajalik? sama vist mis normalzie
#faces_train = faces_train / 255.0
#x = keras.utils.np_utils.normalize(x)

# MUDELI VÄÄRTUSED
dense_layers = 0
conv_layers = 1
layer_size = 64
dropout = 0.1

epochs = 20
validation = 0.1
batch_size = 32

# MUDEL NÄGUDE JAOKS
# mudeli tensorboard log
"""
NAME = "smile 1 : {}-dense {}-node {}-conv".format(dense_layers, layer_size, conv_layers)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
print(NAME)

model_faces = Sequential()

model_faces.add(Conv2D(layer_size, (3, 3), activation='relu', input_shape=faces_train.shape[1:]))
model_faces.add(MaxPooling2D(pool_size=(2, 2)))
model_faces.add(Conv2D(layer_size, (3, 3), activation='relu'))
model_faces.add(MaxPooling2D(pool_size=(2, 2)))
model_faces.add(Flatten())
model_faces.add(Dropout(dropout))
model_faces.add(Dense(1, activation='sigmoid'))

"""
# MUDEL SUUDE JAOKS



#model_faces.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation)
#model.fit(x, y, epochs=10, batch_size=32, validation_split=0.1, callbacks=[tensorboard])

#score = model_faces.evaluate(x, y, verbose=0)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

#model.save('smile_faces1.model')

















