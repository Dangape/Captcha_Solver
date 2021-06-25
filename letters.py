import os
import os.path
import cv2
import glob
import imutils
import numpy as np
import numpy as np
import imutils
import cv2
from PIL import Image


CAPTCHA_IMAGE_FOLDER = "Data/tratados/1/"
OUTPUT_FOLDER = 'Data/letras1'


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER,'*'))
print(captcha_image_files)
counts = {}



for (i, captcha) in enumerate(captcha_image_files):

    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    filename = os.path.basename(captcha)

    captcha_correct_text = os.path.splitext(filename)[0]
    img = cv2.imread(captcha)
    resized = cv2.resize(img, (140, 60), interpolation=cv2.INTER_AREA)
    # cinza
    img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.copyMakeBorder(img_gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # preto e branco
    _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)

    # encontrar contornos de letras
    contornos, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    regiao_letras = []

    # filtrar contornos que sao realmente letras
    max_w, max_h = img.shape[:2]

    for contorno in contornos:
        (x, y, l, a) = cv2.boundingRect(contorno)
        if l >= max_w:
            pass
        else:
            area = cv2.contourArea(contorno)
            # print('Area:',area)
            # print(l,a)
            if area > 200:
                if l / a > 1.1:
                    half_width = int(a / 2)
                    regiao_letras.append((x, y, half_width, a))
                    regiao_letras.append((x + half_width, y, half_width, a))
                else:
                    regiao_letras.append((x, y, l, a))

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])

    if len(regiao_letras) != 5:
            continue

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
    # print(regiao_letras)

    # desenhar contornos e separar em arquivos
    img_final = cv2.merge([img] * 3)
    lista_letras = []

    for retangulo,texto_letra in zip(regiao_letras,captcha_correct_text):
        x, y, l, a = retangulo
        img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

        cv2.rectangle(img_final, (x - 2, y - 2), (x + l + 2, y + a + 2), (0, 255, 0), 1)
        lista_letras.append(img_letra)

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, texto_letra)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(texto_letra, 1)
        p = os.path.join(save_path, "{}.png".format(str(count+5000).zfill(6)))
        cv2.rectangle(img_final,(x-2,y-2),(x+l+2,y+a+2),(0,255,0),1)

        cv2.imwrite(p, img_letra)



        # increment the count for the current key
        counts[texto_letra] = count + 1
