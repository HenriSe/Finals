import cv2 as cv2

mouth_cascade = cv2.CascadeClassifier("venv\Lib\site-packages\cv2\data\haarcascade_smile.xml")


def leia_suu(img):
    # kui scaler 1.1, tekitab mitu kasti üle näo, kui scaler 1.9, tekitab ühe ja korrektse
    # sama ka vimase arvuga, kui see ntks 2, siis mitu kasti, kui see 20 siis üks korrektne
    scaler = 1.9
    suud = mouth_cascade.detectMultiScale(img, scaler, 20)
    for (x, y, w, h) in suud:
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

    # TODO kui ei leia ühtegi

    cv2.imshow('img', img)
    cv2.waitKey()
    return img
