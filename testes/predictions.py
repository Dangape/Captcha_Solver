from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os


#Load model
path = 'Data/training_data/'
model = load_model('result_model.h5')

#Preprocessing pipeline
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

############################################
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

y_combine = LabelEncoder().fit_transform(y)
y_one_hot = OneHotEncoder(sparse = False).fit_transform(y_combine.reshape(len(y_combine),1)) #convert label to numeric array


info = {y_combine[i] : y[i] for i in range(len(y))}
#############################################

def get_demo(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, 'gray')
    plt.axis('off')
    # plt.show()

    img = t_img(img)
    img = c_img(img)
    img = d_img(img)
    img = b_img(img)

    image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]

    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.show()
    Xdemo = []
    for i in range(5):
        Xdemo.append(img_to_array(Image.fromarray(image_list[i])))

    Xdemo = np.array(Xdemo)
    Xdemo /= 255.0

    ydemo = model.predict(Xdemo)
    ydemo = np.argmax(ydemo, axis=1)

    pred_str = ""
    for res in ydemo:
        pred_str += str(info[res])

    # print("Real captcha:",img_path[-9:])
    # print("Predicted captcha:",pred_str)
    return img_path[-9:],pred_str

# test = get_demo('Testes reais/8ldip.jpg')
# print(test[0])