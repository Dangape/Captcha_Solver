from imutils import paths
import numpy as np
import imutils
import cv2
from PIL import Image

#Tirar comentario do imrad e trocar parametro path no grayscaling para rodar localmente
def process_1(path):
    # Load image
    # img = cv2.imread(path)
    # Grayscaling
    img = cv2.cvtColor(path, cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(img,240,255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
    thresh = cv2.bitwise_not(thresh)
    rgb = Image.fromarray(thresh)
    # print(type(thresh))
    return thresh

def process_2(path):

    # Load image
    img = cv2.imread(path)
    # Grayscaling
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _,thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img = Image.fromarray(img)
    img = img.convert("P")
    img2 = Image.new("P", img.size, 255)

    for x in range(img.size[1]): #iterate columns
        for y in range(img.size[0]): #iterate lines
            pixel_color = img.getpixel((y,x))
            if pixel_color < 127:
                img2.putpixel((y,x),0)

    # img2.save("treated/imagem_tratada_{}.png".format(i))
    img = np.asarray(img2)
    img = cv2.bitwise_not(img)

    # cv2.imshow("Output", img)
    # cv2.waitKey()

    # cv2.imwrite("treated/imagem_tratada_{}.png".format("group2"), img)
    return img


def process_3(path):
    # Load image
    img = cv2.imread(path)
    # Grayscaling
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _,thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY + cv2.THRESH_TRUNC)

    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 1), np.uint8))
    dilate = cv2.dilate(close, np.ones((4, 1), np.uint8), iterations=1)

    img = Image.fromarray(dilate)
    img = img.convert("P")
    img2 = Image.new("P", img.size, 255)

    for x in range(img.size[1]): #iterate columns
        for y in range(img.size[0]): #iterate lines
            pixel_color = img.getpixel((y,x))
            if pixel_color <127:
                img2.putpixel((y,x),0)

    img = np.asarray(img2)

    # cv2.imshow("Output", img)
    # cv2.waitKey()

    # cv2.imwrite("treated/imagem_tratada_{}.png".format("group3"), img)
    return img

#Tirar comentario do imgread se quiser rodar localmente
def get_letters(captcha):
    counts = {}
    resized = cv2.resize(captcha, (140,60), interpolation=cv2.INTER_AREA)
    # cinza
    # img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.copyMakeBorder(resized, 10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
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
            if area > 200:
                if l / a > 1.1:
                    half_width = int(a / 2)
                    regiao_letras.append((x, y, half_width, a))
                    regiao_letras.append((x + half_width, y, half_width, a))
                else:
                    regiao_letras.append((x, y, l, a))

    # if len(regiao_letras) != 5:
    #     continue

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])

    # desenhar contornos e separar em arquivos
    img_final = cv2.merge([img] * 3)
    lista_letras = []
    for retangulo in regiao_letras:
        x, y, l, a = retangulo
        img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

        cv2.rectangle(img_final, (x - 2, y - 2), (x + l + 2, y + a + 2), (0, 255, 0), 1)
        lista_letras.append(img_letra)

    return lista_letras