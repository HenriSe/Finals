import random
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential

import andmete_hankimine

path = r"C:\Henri\Desktop\loputoo\tester_pildid\cats_vs_dogs_tester\train"

# cd C:\Henri\Desktop\loputoo\smilers > tensorboard --logdir=logs/

kaustad = ["dogs", "cats"]

img_size = 224
andmed = andmete_hankimine.hangi_andmed(path, img_size)

random.shuffle(andmed)

print("\nTreening andmete pikkus: ", len(andmed))

x = []
y = []

for fn, sildid in andmed:
    x.append(fn)
    y.append(sildid)

x = np.array(x).reshape(-1, img_size, img_size, 1)
y = np.array(y)

x = x / 255.0

# hetkene top 1
# TODO uurida miks ta dense ei taha
# TODO kernel size muudaks?
# TODO vajab veel kruttimist - error and trial
layer_size = 64
conv_layer = 1
dense_layer = 0

# mudel logisse
NAME = "TOP1 - with dropout : {}-dense {}-node {}-conv".format(dense_layer, layer_size, conv_layer)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
print(NAME)

model = Sequential()

# sätime layer size 64 peale
model.add(Conv2D(layer_size, (3, 3), activation='relu', input_shape=x.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

# lisatud 1 conv layer
model.add(Conv2D(layer_size, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
# lisades dropout layer, paraneb täpsus, väikese dataset mure
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x, y, epochs=10, batch_size=32, validation_split=0.1)

score = model.evaluate(x, y, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

model.summary()

# for predictions, salvestan mudeli
#model.save('TOP1.model')
