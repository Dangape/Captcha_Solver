import cv2
import pandas
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import time
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report




start_time = time.time()
# path1 = 'Data/training_data/23n88.png'
# path2 = 'Data/training_data/23mdg.png'

path = 'Data/generated_captcha_images/'

#Image preprocessing
#adaptative threshholding function
def t_img (img) :
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

#closing
def c_img (img) :
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,2), np.uint8))

#dilation
def d_img (img) :
    return cv2.dilate(img, np.ones((2,2), np.uint8), iterations = 1)

#smoothing images
def b_img (img) :
    return cv2.GaussianBlur(img, (1,1), 0)

X = []
y = []

for image in os.listdir(path):

    if image[6:] != 'png':
        continue

    img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE) #gray scaling images

    #preprocessing using functions defined above
    img = t_img(img)
    img = c_img(img)
    img = d_img(img)
    img = b_img(img)

    image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]

    for i in range(5):
        X.append(img_to_array(Image.fromarray(image_list[i])))
        y.append(image[i])
X = np.array(X)
y = np.array(y)

print(len(X))
print(set(y))

X /= 255.0 #normalizing pixels to be between 0 and 1

plt.figure(figsize = (15,5),dpi=80)
for i in range(5) :
    plt.subplot(1,5,i+1)
    plt.imshow(X[i], 'gray')
    plt.title('Label is ' + str(y[i]))
plt.plot()
plt.savefig("Plots/preprocess.png")

y_combine = LabelEncoder().fit_transform(y)
y_one_hot = OneHotEncoder(sparse = False).fit_transform(y_combine.reshape(len(y_combine),1)) #convert label to numeric array


info = {y_combine[i] : y[i] for i in range(len(y))}

#Building model
#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.2, random_state = 1)


def conv_layer(filterx):
    model = Sequential()

    model.add(Conv2D(filterx, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    return model


def dens_layer(hiddenx):
    model = Sequential()

    model.add(Dense(hiddenx, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    return model


def cnn(filter1, filter2, filter3, hidden1, hidden2):
    model = Sequential()
    model.add(Input((40, 20, 1,)))

    model.add(conv_layer(filter1))
    model.add(conv_layer(filter2))
    model.add(conv_layer(filter3))

    model.add(Flatten())
    model.add(dens_layer(hidden1))
    model.add(dens_layer(hidden2))

    model.add(Dense(19, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#SMOTE
X_train = np.reshape(X_train, (4160, 40*20*1))
X_train, y_train = SMOTE(sampling_strategy = 'auto', random_state = 1).fit_resample(X_train, y_train)
print(X_train.shape)
print(y_train.shape)
X_train = np.reshape(X_train, (8037, 40, 20, 1))

plt.figure(figsize = (20,20))

hi = 7800
lo = 5000

for i in range(25) :
    plt.subplot(5,5,i+1)
    x = np.random.randint(lo, hi)
    plt.imshow(X_train[x], 'gray')
    plt.title('Label is ' + str(info[np.argmax(y_train[x])]))
plt.savefig("Plots/smote.png")

traingen = ImageDataGenerator(rotation_range = 5, width_shift_range = [-2,2])
traingen.fit(X_train)

train_set = traingen.flow(X_train, y_train)
trainX, trainy = train_set.next()

plt.figure(figsize = (20,20))

hi = 32
lo = 0

for i in range(25) :
    plt.subplot(5,5,i+1)
    x = np.random.randint(lo, hi)
    plt.imshow(trainX[x], 'gray')
    plt.title('Label is ' + str(info[np.argmax(trainy[x])]))
plt.savefig("Plots/rotated_images.png")

# model = cnn(128, 32, 16, 32, 32)
# model.summary()
#
# checkp = ModelCheckpoint('result_model.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)
# reduce = ReduceLROnPlateau(monitor = 'val_loss', patience = 20, verbose = 1)
# history = model.fit(traingen.flow(X_train, y_train, batch_size = 32),
#                     validation_data = (X_test, y_test), epochs = 150, steps_per_epoch = len(X_train)/32,
#                     callbacks = [checkp])
# plt.figure(figsize = (20,10))
# plt.subplot(2,1,1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Losses')
# plt.legend(['train loss','val loss'])
# plt.title('Loss function wrt epochs')
#
# plt.subplot(2,1,2)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(['train acc' , 'val acc'])
# plt.title('Model accuracy wrt Epoch')
#
# plt.savefig("Plots/model_training.png")
# print("%s minutes" %((time.time() - start_time)/60))

model = load_model('result_model.h5')
pred = model.predict(X_test)

pred = np.argmax(pred, axis = 1)
yres = np.argmax(y_test,axis= 1)

target_name = []
for i in sorted(info) :
    target_name.append(info[i])

print('Accuracy : ' + str(accuracy_score(yres, pred)))
print(classification_report(yres, pred, target_names = target_name))
