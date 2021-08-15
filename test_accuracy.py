import os
from imutils import paths
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import base64
from imageio import imread
import API_KERAS.processing_lab as processing_lab
from tqdm import tqdm
import cv2

CAPTCHA_IMAGE_FOLDER = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\test_set\7"
MODEL_FILENAME = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\API_KERAS\result_model_letter.h5"
MODEL_LABELS_FILENAME = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\API_KERAS\model_labels.dat"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))

def predict_text(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictions = []
    img = processing_lab.model2(img)
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
        captcha_text = captcha_text.replace("_","")
        # filename = name1.filename
        # Find real captcha name
        # end = filename.rfind('.')  # last occurence of '.'
        # real = filename[:end]
    return str(captcha_text)

# teste = predict_text(captcha_image_files[0]) == '010E3'
# print('Predicted:',teste)
# print('Real:',captcha_image_files[0])

'''
1 = caxias
2 = barueri
3 = niteroi
4 = aparecido de goiania
'''
correct = 0
folder = 7

path = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\test_set\{}".format(folder)
captcha_image_files = list(paths.list_images(path))
# print(captcha_image_files)
wrongs = []
for file in tqdm(captcha_image_files):
    name = os.path.basename(file).split('.')[0]
    predicted = predict_text(file)
    if predicted == name: #check right predictions
        correct += 1
    if predicted != name: #check wrong predictions
        wrongs.append(os.path.basename(file).split('.')[0])
print('Acc:',correct/len(captcha_image_files)*100)
print("wrong captchas:",len(wrongs))
print(wrongs)
