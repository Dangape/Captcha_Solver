import os
import os.path
import cv2
import glob
import imutils
import numpy as np
import imutils
import cv2.cv2 as cv2
from PIL import Image
import numpy as np
from imutils import paths
import base64
import API_KERAS.processing_lab as pl

#funciona para 1,2,4
CAPTCHA_IMAGE_FOLDER_1 = r"E:\Users\Daniel\OneDrive\CaptchaML\Data\Testes reais\7tgpd.png"
CAPTCHA_IMAGE_FOLDER_2 = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\test_set\1\0176E.png"
CAPTCHA_IMAGE_FOLDER = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\not_trained\8"

captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(1,), replace=False)

raw_img = cv2.imread(captcha_image_files[0])
# raw_img = cv2.imread(CAPTCHA_IMAGE_FOLDER_7)

img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
cv2.imshow("Output", thresh)
cv2.waitKey()

thresh = cv2.bitwise_not(thresh)
cv2.imshow("Output", thresh)
cv2.waitKey()

# resized = cv2.resize(thresh, (140, 60), interpolation=cv2.INTER_AREA)
# cv2.imshow("Output", resized)
# cv2.waitKey()

img = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
cv2.imshow("Output", img)
cv2.waitKey()

# blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
# cv2.imshow("Output", blur)
# cv2.waitKey()

# threshold
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
cv2.imshow("Output", img)
cv2.waitKey()

# find letter contours
contornos, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
regiao_letras = []

# filter contours that are letters
max_w, max_h = img.shape[:2]

for contorno in contornos:
    (x, y, l, a) = cv2.boundingRect(contorno)
    if l >= max_w:
        pass
    else:
        area = cv2.contourArea(contorno)
        print('Area:',area)
        print(l,a)
        if area >= 80:
            if l / a > 1.1:
                half_width = int(a / 2)
                regiao_letras.append((x, y, half_width, a))
                regiao_letras.append((x + half_width, y, half_width, a))
            else:
                regiao_letras.append((x, y, l, a))

regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
print(regiao_letras)
prod = []
if len(regiao_letras) > 5:
    for i in regiao_letras:
        prod.append(i[2] * i[3])
    min_index = prod.index(min(prod))
    del regiao_letras[min_index]

# desenhar contornos e separar em arquivos
img_final = cv2.merge([img] * 3)
lista_letras = []
for retangulo in regiao_letras:
    x, y, l, a = retangulo
    img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

    cv2.rectangle(img_final, (x - 2, y - 2), (x + l + 2, y + a + 2), (0, 255, 0), 1)
    lista_letras.append(img_letra)

    cv2.imshow("Output", img_letra)
    cv2.waitKey()
#
cv2.imshow("Output", img_final)
cv2.waitKey()
