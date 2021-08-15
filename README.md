# Captcha Machine Learning Project (Work in progress)
## Running instructions

All the needed files are inside the `API_KERAS` folder. 
In order to run the API locally just run the `application_keras.py` and use Postman to make a POST request.

If you want to download the container directly from my [Docker Hub profile](https://hub.docker.com/u/dangape) just run the following code:
```
docker pull dangape/api_keras
```

In order to test the container just run the image using the following command:
```
docker run dangape/api_keras
```

In order to test the API you can use postman or another request app to make a POST request.
Notice that you can make a POST request with a base64 string image and with the key `string`, just like the image bellow.  
Or you can make a POST request with a image file, just make sure to change the key name to 'file' in this case. 
Also remember to change the request code line inside the `application_keras.py` file. You can use [this](https://base64.guru/converter/encode/image) website to create a base64 string
![Request tutorial](/Plots/postman_prt.png)


## Python requirements
You can find all the requirements to run the code in the `requirements.txt` file. But to make it easier I´ll list them below:
- numpy
- Flask
- opencv-python
- imageio
- Pillow
- imutils
- uwsgi
- tensorflow
- keras
- sklearn

## CAPTCHA images
This is a work in progress and currently the algorithm works just for simple CAPTCHAS, preferably with five letters.   
Below you can see 4 examples of images that the algorithm can handle well.

![Request tutorial](/Plots/captcha1.png)
![Request tutorial](/Plots/captcha2.png)
![Request tutorial](/Plots/captcha3.png)
![Request tutorial](/Plots/captcha4.png)

## Folder and Files
- `LETTER_LAB.py` perform opencv tests on captcha files, with this file you can test process to see how they; will handle the given captcha.
- `get_captcha.py` perform web scrapping in some sites to download captcha files to train the model;
- `label_data.py` uses an online API to label downloaded data;
- `build_training_data_letters.py` split captcha into single letters and assign them to a folder with the matching letter name;
- `train_model.py` trains CNN model;
- `test_accuracy.py` tests the model accuracy;
- `application_keras.py` run the Flask API;
- `processing_lab.py` contains different models for processing different captchas;

## Models

- Use `model1` for the first and second captcha type;
- Use `model2` for the third captcha type;
- Use `model3` for the fourth captcha type.

## Accuracy socres
- `model1`: 87,5%
	- first captcha: 87,5%
	- second captcha: 69,5%
- `model2`: 45,00%
- `model3`: 99,00%