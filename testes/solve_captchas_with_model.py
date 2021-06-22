from tensorflow.keras.models import load_model
from processing_lab import resize_to_fit, get_letters
from imutils import paths
import numpy as np
import cv2
import pickle


MODEL_FILENAME = r"E:\Users\Daniel\OneDrive\CaptchaML\result_model_letter.h5"
MODEL_LABELS_FILENAME = r"E:\Users\Daniel\OneDrive\CaptchaML\model_labels.dat"
CAPTCHA_IMAGE_FOLDER = r"E:\Users\Daniel\OneDrive\CaptchaML\Data\tratados\1"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(1,), replace=False)
print(captcha_image_files)

def solve_captcha(image_file):
    img = cv2.imread(image_file)
    resized = cv2.resize(img, (140, 60), interpolation=cv2.INTER_AREA)
    # cinza
    img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.copyMakeBorder(img_gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
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

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
    predictions = []

    # loop over the lektters
    for letter_bounding_box in regiao_letras:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = img[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    # print("CAPTCHA text is: {}".format(captcha_text))
    # print("Real text is: {}".format(image_file[-8:])))

    #Find real captcha name
    string = image_file
    start = string.rfind('\\') #last occurence of '\'
    end = string.rfind('.') #last occurence of '.'

    return captcha_text,image_file[start+1:end]

image_file = r"E:\Users\Daniel\OneDrive\CaptchaML\Data\tratados\1\16FXY.png"
solution = solve_captcha(captcha_image_files[0])
print('Predicted text is: {}'.format(solution[0]))
print('Real text is: {}'.format(solution[1]))
