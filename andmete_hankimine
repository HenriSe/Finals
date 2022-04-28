import os
import cv2

# datadir = r'C:\Users\Henri\Desktop\lõputöö\tester_pildid'
# siin asuvad kaustad, kus on pildid with fake\real
datadir = r"C:\Users\Henri\Desktop\loputoo\tester_pildid"
kaustad = ["fake_smile", "real_smile"]


def hangi_andmed(datadir, pildi_suurus):

    jada = []

    for valik in kaustad:
        # koostan tee kausta
        path = os.path.join(datadir, valik)

        # eristan kas pilt on kaustas 1 - real, või kaustas 0 - fake
        num = kaustad.index(valik)

        for img in os.listdir(path):
            try:  # juhul kui image on broken

                img_path = os.path.join(path, img)
                # loen pildi hallides toonides - tõmbab mahtu väiksemaks
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (pildi_suurus, pildi_suurus))
                # lükkan arraysse oma pildid ja nende class_num
                jada.append([new_array, num])
            except Exception as err:
                print(err)
                pass

    return jada
