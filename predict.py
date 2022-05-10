import os
import cv2
import keras.utils.np_utils
from keras.models import load_model

# programm võtab varem salvestatud mudeli ja kasutab seda sisseantud failide peal
# sellega saan sisestada uusi pilte ja saada nende kohta pakkumise millega tegu

model_path = './saved_model'
model = load_model(model_path, compile=True)

path = r''

kaustad = ["real", "fake"]


def build(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = keras.models.load_model('.model')

for img in os.listdir(path):
    #print(path + img)
    prediction = model.predict([build(path + "\\" + img)])
    # prediction ehitatakse alati jada sisse, toon välja kausta nime
    print(img, " : ", kaustad[int(prediction[0][0])])

#prediction = model.predict([prepare(path)])
#print(prediction)

