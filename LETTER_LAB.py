import os
import os.path
import cv2
import glob
import imutils
import numpy as np
import imutils
import cv2
from PIL import Image
import numpy as np
import base64



CAPTCHA_IMAGE_FOLDER = r"E:\Users\Daniel\OneDrive\CaptchaML\Data\captcha_groups\1\1L8QM.png"


# Load image
img = cv2.imread(CAPTCHA_IMAGE_FOLDER)
# cv2.imshow("Output", img)
# cv2.waitKey()
# Grayscaling
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(img,240,255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
thresh = cv2.bitwise_not(thresh)
# rgb = Image.fromarray(thresh)
# print(type(thresh))
counts = {}
# img = cv2.imread(captcha)
resized = cv2.resize(thresh, (140, 60), interpolation=cv2.INTER_AREA)
# cinza
# img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
img_gray = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
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
        print('Area:',area)
        print(l,a)
        if area > 155:
            if l / a > 0.7:
                half_width = int(a / 2)
                regiao_letras.append((x, y, half_width, a))
                regiao_letras.append((x + half_width, y, half_width, a))
            else:
                regiao_letras.append((x, y, l, a))

regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
print(regiao_letras)

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

cv2.imshow("Output", img_final)
cv2.waitKey()

#
