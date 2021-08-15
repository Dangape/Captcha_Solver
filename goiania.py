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
from imutils import paths
import base64
import API_KERAS.processing_lab as p
import matplotlib.pyplot as plt


CAPTCHA_IMAGE_FOLDER = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\captcha_groups\7"
CAPTCHA_IMAGE_FOLDER_3 = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\not_trained\7\1LZF.png" #vermelha
CAPTCHA_IMAGE_FOLDER_5 = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\not_trained\7\1iL7.png" #azul
CAPTCHA_IMAGE_FOLDER_6 = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\not_trained\7\2ijQ.png" #preta
CAPTCHA_IMAGE_FOLDER_4 = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\not_trained\7\0RLf.png" #branca

captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
print(captcha_image_files)
captcha_image_files = np.random.choice(captcha_image_files, size=(1,), replace=False)

img = cv2.imread(captcha_image_files[0])
img = img[15:55, 10:100] #crop image to get dominant color

dominant = img.copy() #copy image

#get dominant color to get text color
unique, counts =np.unique(dominant.reshape(-1,dominant.shape[-1]), axis=0, return_counts=True)
dominant[:,:,0], dominant[:,:,1], dominant[:,:,2] = unique[np.argmax(counts)]
dominant = list(dominant[0,0])
print(dominant)

height, width, _ = img.shape

cv2.imshow('hough',img)
cv2.waitKey(0)

'''
colors in BGR code
[blue,green,red]
vermelho = [0,0,255]
branco = [255,255,255]
preto = [0,0,0]
azul_claro = [255,0,0]
azul_escuro = [139,0,0]
'''

bgr_codes = [[0,0,255],[255,255,255],[0,0,0],[255,0,0],[139,0,0]]

# print((bgr_codes.index(dominant)))
print(dominant==[255,255,255])
for i in range(height):
    for j in range(width):
        # img[i,j] is the RGB pixel at position (i, j)
        if dominant == [255,255,255]:
            if any(img[i,j] != bgr_codes[bgr_codes.index(dominant)]):
                img[i, j] = [255, 255, 255]
            elif any(img[i, j] == bgr_codes[bgr_codes.index(dominant)]):
                img[i, j] = [0, 0, 0]
        elif any(img[i,j] != bgr_codes[bgr_codes.index(dominant)]):
            img[i, j] = [255, 255, 255]


# cv2.imshow('Test', img)
# cv2.waitKey(0)


counts = {}
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Final", img)
cv2.waitKey()

#threshold
_,thresh = cv2.threshold(img,240,255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)

#resizing
# resized = cv2.resize(thresh, (140, 60), interpolation=cv2.INTER_AREA)
img_gray = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#
cv2.imshow("Final", img_gray)
cv2.waitKey()
print(img_gray.shape)

#reforcing image
blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
blur2 = cv2.GaussianBlur(blur,(5,5),0)
thresh2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 3)

# cv2.imshow("Final", thresh2)
# cv2.waitKey()

kernel = np.ones((2,1),np.uint8)
img = cv2.dilate(thresh2, kernel,1)
# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# img = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, rect_kernel)
# cv2.imshow("Final", img)
# cv2.waitKey()

#find letter contours

contornos, heirar= cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

regiao_letras = []

#filter contours that are letters
max_w, max_h = img.shape[:2]

for contorno in contornos:
    (x, y, l, a) = cv2.boundingRect(contorno)
    if l >= max_w:
        pass
    else:
        area = cv2.contourArea(contorno)
        print('Area:',area)
        print(l,a)
        if area >= 125:
            if l / a > 1.2:
            # if l>a:
                # pass
                half_width = int(a / 2)
                regiao_letras.append((x, y, half_width, a))
                regiao_letras.append((x + half_width, y, half_width, a))
            else:
                regiao_letras.append((x, y, l, a))

regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
print(regiao_letras)
prod = []
if len(regiao_letras) > 4:
    for i in regiao_letras:
        prod.append(i[2]*i[3])
    min_index = prod.index(min(prod))
    del regiao_letras[min_index]
print(regiao_letras)

# desenhar contornos e separar em arquivos
img_final = cv2.merge([img] * 3)

# cv2.imshow("Final", img_final)
# cv2.waitKey()

lista_letras = []
for retangulo in regiao_letras:
    x, y, l, a = retangulo
    img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

    cv2.rectangle(img_final, (x - 2, y - 2), (x + l + 2, y + a + 2), (0, 255, 0), 1)
    lista_letras.append(img_letra)
    # cv2.imshow("Letter", img_letra)
    # cv2.waitKey()
#
cv2.imshow("Contours", img_final)
cv2.waitKey()
