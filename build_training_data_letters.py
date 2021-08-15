import os
import os.path
import cv2
import glob
import imutils
import numpy as np


CAPTCHA_IMAGE_FOLDER = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\captcha_groups\7"
OUTPUT_FOLDER = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\letras_goiania"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}
skipped = 0
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)

    # crop image to get dominant color
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
    thresh = cv2.bitwise_not(thresh)

    img = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)

    # find letter contours
    contornos, heirar = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    regiao_letras = []

    # filter contours that are letters
    max_w, max_h = img.shape[:2]

    for contorno in contornos:
        (x, y, l, a) = cv2.boundingRect(contorno)
        if l >= max_w:
            pass
        else:
            area = cv2.contourArea(contorno)
            # print('Area:', area)
            # print(l, a)
            if area >= 80:
                if l / a > 1.1:
                    # if l>a:
                    # pass
                    half_width = int(a / 2)
                    regiao_letras.append((x, y, half_width, a))
                    regiao_letras.append((x + half_width, y, half_width, a))
                else:
                    regiao_letras.append((x, y, l, a))
                    
    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
    prod = []
    if len(regiao_letras) > 5:
        for i in regiao_letras:
            prod.append(i[2] * i[3])
        min_index = prod.index(min(prod))
        del regiao_letras[min_index]

    # desenhar contornos e separar em arquivos
    img_final = cv2.merge([img] * 3)
    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!

    if len(regiao_letras) != 5:
        print('SKIP',filename)
        skipped += 1
        continue

    # Save out each letter as a single image
    for retangulo, letter_text in zip(regiao_letras, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, l, a = retangulo

        # Extract the letter from the original image with a 2-pixel margin around the edge
        img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

        # cv2.imshow('hough', img_letra)
        # cv2.waitKey(0)

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, img_letra)

        # increment the count for the current key
        counts[letter_text] = count + 1
print(skipped)