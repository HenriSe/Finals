import random
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import andmete_hankimine

# faili eesmärk on leida parim optimeeritud mudeli ülesehitus
# selleks loopin läbi erinevate väärtuste, et leida parim mudel parimate tulemustega
# võrdlen tensorboard logisid

path = r"C:\Henri\Desktop\loputoo\tester_pildid\smiles\train"

# cd C:\Henri\Desktop\loputoo\smilers > tensorboard --logdir=logs/

kaustad = ["real", "fake"]

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

# parameetrite väärtused
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# andmete loopid
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            # logide nimed
            NAME = "{}-dense {}-node {}-conv".format(dense_layer, layer_size, conv_layer)
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)

            # esimene kiht
            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), activation='relu', input_shape=x.shape[1:]))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # TODO lisada eraldi activation layer?

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # flatten kiht siia sujumiseks
            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))

            # lõpu dense layer
            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer="adam",
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.fit(x, y, epochs=20, batch_size=32, validation_split=0.1, callbacks=[tensorboard])

            score = model.evaluate(x, y, verbose=0)
            print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
