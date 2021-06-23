import cv2
import pytesseract
from pytesseract import Output
import string as s

import processing_lab
from processing_lab import *

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CAPTCHA_IMAGE_FOLDER = r"E:\Users\Daniel\OneDrive\CaptchaML\Data\captcha_groups\1"

captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(1,), replace=False)

def predict_text(path):
    predictions = {}

    raw_img = processing_lab.process_1(path)
    img = processing_lab.get_letters(raw_img)
    for letter in img:
        rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)

        results = pytesseract.image_to_data(rgb, output_type=Output.DICT,config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789")

        # loop over each of the individual text localizations
        for i in range(0, len(results["text"])):
            # extract the bounding box coordinates of the text region from
            # the current result
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            # extract the OCR text itself along with the confidence of the
            # text localization
            text = results["text"][i]
            conf = int(float(results["conf"][i]))
            predictions.update({text:conf})

        # filter out weak confidence text localizations
            if conf > 50:
                # display the confidence and text to our terminal
                # print("Confidence: {}".format(conf))
                # print("Text: {}".format(text))
                # print("")
                # strip out non-ASCII text so we can draw the text on the image
                # using OpenCV, then draw a bounding box around the text along
                # with the text itself
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(letter, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(letter, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
    table = str.maketrans('', '', s.ascii_lowercase)
    predicted = ''.join(predictions.keys()).translate(table)

    # # Find real captcha name
    string = captcha_image_files[0]
    start = string.rfind('\\')  # last occurence of '\'
    end = string.rfind('.')  # last occurence of '.'
    real = string[start + 1:end]
    return real, predicted

teste = predict_text(captcha_image_files[0])
print("Real text is:",teste[0])
print("Predicted text is:",teste[1])

