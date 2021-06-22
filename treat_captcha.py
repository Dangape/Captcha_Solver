from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import glob
import processing_lab
import os

CAPTCHA_IMAGE_FOLDER = "Data/captcha_groups/4"
OUTPUT_FOLDER = 'Data/tratados/4/{}'


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
print(captcha_image_files)
counts = {}

for (i, captcha) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
    filename = os.path.basename(captcha)

    captcha_correct_text = os.path.splitext(filename)[0]
    img = processing_lab.process_3(captcha)
    cv2.imwrite(OUTPUT_FOLDER.format(filename), img)
