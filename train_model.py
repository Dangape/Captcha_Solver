import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from API_KERAS.processing_lab import resize_to_fit
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator


start_time = time.time()

LETTER_IMAGES_FOLDER = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Data\extracted_letter_images"
MODEL_FILENAME = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\API_KERAS\captcha_model.hdf5"
MODEL_LABELS_FILENAME = r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\API_KERAS\model_labels.dat"

# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in tqdm(paths.list_images(LETTER_IMAGES_FOLDER)):

    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Resize the letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2]

    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# print(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.30, random_state=0)
#
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
#Augmenting image data_set
sm = SMOTE(random_state=42)
X_train = np.reshape(X_train, (X_train.shape[0], 20*20*1))
# X_test = np.reshape(X_test, (18288,20*20*1))

X_smote, Y_smote = sm.fit_resample(X_train, Y_train)
# X_test_smote,Y_test_smote = sm.fit_resample(X_test,Y_test)
# print(X_test_smote.shape)
# print(Y_smote.shape)

X_smote = np.reshape(X_smote,(X_smote.shape[0],20,20,1))
traingen = ImageDataGenerator(rotation_range = 5, width_shift_range = [-2,2])
traingen.fit(X_smote)
train_set = traingen.flow(X_smote, Y_smote)
trainX, trainy = train_set.next()
# X_test_smote = np.reshape(X_test_smote, (44103,20,20,1))

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_smote)
Y_train = lb.transform(Y_smote)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling input_shape=(20, 20, 1)
model.add(Conv2D(32, kernel_size =(5,5), padding="same", input_shape=(20,20,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(64, kernel_size =(5,5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size =(5,5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layer
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))

# Output layer with 61 nodes (one for each possible letter/number we predict)
model.add(Dense(61, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
checkp = ModelCheckpoint('API_KERAS/result_model_letter.h5', monitor ='val_loss', verbose = 1, save_best_only = True)
reduce = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, verbose = 1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(traingen.flow(X_smote, Y_train, batch_size = 128), validation_data=(X_test, Y_test), epochs=50, verbose=1,
                    callbacks = [checkp,reduce,early_stopping])
model.save(MODEL_FILENAME)

plt.figure(figsize = (20,10))
plt.subplot(2,1,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend(['train loss','val loss'])
plt.title('Loss function x Epoch')

plt.subplot(2,1,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train acc' , 'val acc'])
plt.title('Model accuracy x Epoch')
plt.savefig(r"C:\Users\DANIEL BEMERGUY\OneDrive\CaptchaML\Plots\model_training.png")

model = load_model("result_model_letter.h5")
pred = model.predict(X_test)

pred = np.argmax(pred, axis = 1)
yres = np.argmax(Y_test,axis= 1)

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# target_name = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","P","Q","R","S","T","U","V","X","W","Y","Z"]
# target_name = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","P","Q","R","S","T","U","V","X","W","Y","Z","a_",
#               "b_","c_","d_","e_","f_","g_","h_","i_","j_","k_","l_","m_","n_","o_","p_","q_","r_","s_","t_","u_","v_","x_","y_","z_"]

print('Accuracy : ' + str(accuracy_score(yres, pred)))
# print(classification_report(yres, pred,target_names=target_name))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("%s minutes" %((time.time() - start_time)/60))