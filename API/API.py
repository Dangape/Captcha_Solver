from flask import Flask, render_template, request, redirect, url_for

import processamentos
import cv2
import pytesseract
from pytesseract import Output
import base64
import string as s
from imageio import imread
import io



pytesseract.pytesseract.tesseract_cmd = r"C:\Program FIles\Tesseract-OCR\tesseract.exe"
app = Flask(__name__)
app.config['DEBUG'] = True
UPLOAD_FOLDER = r'E:\Users\Daniel\OneDrive\CaptchaML\templates'
ALLOWED_EXTENSIONS = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    text = 'Acessar /ocr para rodar a API'
    return text

@app.route('/ocr', methods=['POST'])
def predict_text():

    name1 = request.files['file']
    filename = name1.filename
    encoded_string = base64.b64encode(name1.read())
    # reconstruct image as an numpy array
    b64_string = encoded_string.decode()
    img = imread(io.BytesIO(base64.b64decode(b64_string)))
    predictions = {}
    raw_img = processamentos.process_1(img)
    img = processamentos.get_letters(raw_img)
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

    # Find real captcha name
    end = filename.rfind('.')  # last occurence of '.'
    real = filename[:end]
    return {'Predicted':predicted,"Real":real}

if __name__ == '__main__':
    app.run()

