from flask import Flask, request
import processing_lab
import base64
from imageio import imread
import io
import numpy as np
from tensorflow.keras.models import load_model
import pickle


app = Flask(__name__)

app.config['DEBUG'] = True
UPLOAD_FOLDER = r'E:\Users\Daniel\OneDrive\CaptchaML\templates'
ALLOWED_EXTENSIONS = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']
app.secret_key = "secret key"


# MODEL_FILENAME = "/app/result_model_letter.h5"
# MODEL_LABELS_FILENAME = "/app/model_labels.dat"

MODEL_FILENAME = "result_model_letter.h5"
MODEL_LABELS_FILENAME = "model_labels.dat"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

@app.route('/')
def home():
    text = 'Hello World'
    return text

@app.route('/ocr', methods=['POST'])
def predict_text():
    try:

        #FILE
        name1 = request.files['file']
        b64_string = base64.b64encode(name1.read())

        #B64 STRING
        # name1 = request.form['string']
        # b64_string = name1

        img = imread(io.BytesIO(base64.b64decode(b64_string)))
        predictions = []
        raw_img = processing_lab.process_1(img)
        img = processing_lab.get_letters(raw_img)
        for letter in img:
            # rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = processing_lab.resize_to_fit(letter, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            # Get captcha's text
            captcha_text = "".join(predictions)
            # filename = name1.filename
            # Find real captcha name
            # end = filename.rfind('.')  # last occurence of '.'
            # real = filename[:end]
        return {'Predicted':captcha_text}
    except:
        return 'Captcha error !!'

if __name__ == '__main__':
    # app.run(debug=True, use_debugger=False, use_reloader=False)
    app.run(host='0.0.0.0')