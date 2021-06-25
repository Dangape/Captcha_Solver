
import cv2
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

CAPTCHA_IMAGE_FOLDER = "Data/captcha_groups/2"
OUTPUT_FOLDER = 'Data/tratados/2/{}'


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
print(captcha_image_files)
counts = {}

def process_1(path):
    # Load image
    img = cv2.imread(path)
    # Grayscaling
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(img,240,255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
    thresh = cv2.bitwise_not(thresh)
    # rgb = Image.fromarray(thresh)
    # print(type(thresh))
    return thresh


for (i, captcha) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
    filename = os.path.basename(captcha)

    captcha_correct_text = os.path.splitext(filename)[0]
    print(captcha_correct_text)
    img = process_1(captcha)
    cv2.imwrite(OUTPUT_FOLDER.format(filename), img)
