from imutils import paths
import numpy as np
import imutils
import cv2
from PIL import Image
import scipy.misc

# 1 = caxias
# 2 = barueri
# 3 = niteroi

#Tirar comentario do imread e trocar parametro path no grayscaling para rodar localmente
def process_1(path):
    # Load image
    img = cv2.imread(path)
    cv2.imshow("Output", img)
    cv2.waitKey(10)
    # Grayscaling
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    img = cv2.imread(captcha)
    resized = cv2.resize(img, (140,60), interpolation=cv2.INTER_AREA)
    # cinza
    img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.copyMakeBorder(img_gray, 10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
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

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])

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

    return lista_letras

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

# CAPTCHA_IMAGE_FOLDER = r"E:\Users\Daniel\OneDrive\CaptchaML\Data\tratados\1"

# captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
# captcha_image_files = np.random.choice(captcha_image_files, size=(1,), replace=False)
# # print(get_letters(captcha_image_files[0]))
# print(get_letters(captcha_image_files[0]))


