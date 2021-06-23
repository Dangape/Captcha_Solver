import string as s
import processing_lab
from processing_lab import *
import pickle
from tensorflow.keras.models import load_model


# MODEL_FILENAME = r"E:\Users\Daniel\OneDrive\CaptchaML\API_KERAS\result_model_letter.h5"
# MODEL_LABELS_FILENAME = r"E:\Users\Daniel\OneDrive\CaptchaML\API_KERAS\model_labels.dat"
# CAPTCHA_IMAGE_FOLDER = r"E:\Users\Daniel\OneDrive\CaptchaML\Data\captcha_groups\1"
#
# captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
# captcha_image_files = np.random.choice(captcha_image_files, size=(1,), replace=False)

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

def predict_text(path):
    raw_img = processing_lab.process_1(path)
    img = processing_lab.get_letters(raw_img)
    predictions = []
    for letter in img:
        # rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter, 20, 20)

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

        # Find real captcha name
        string = path
        start = string.rfind('\\')  # last occurence of '\'
        end = string.rfind('.')  # last occurence of '.'

    return captcha_text,path[start+1:end]

teste = predict_text(captcha_image_files[0])
print("Real text is:",teste[0])
print("Predicted text is:",teste[1])