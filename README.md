# Captcha Machine Learning Project (Work in progress)
## Running instructions

All the needed files are inside the `API_KERAS` folder. 
In order to run the API locally just run the `application_keras.py` and use Postman to make a POST request.

In order to run the API and create the container you will need to run the following code on your terminal:
```
docker-compose build && docker-compose up
```

If you want to download the conatiner directly from my [Docker Hub profile](https://hub.docker.com/u/dangape) just run the following code:
```
docker pull dangape/api_keras
```

In order to test the container just run the image using the following command:
```
docker run dangape/api_keras
```

In order to test the API you can use postman or another request app to make a POST request.
Notice that you can make a POST request with a base64 string image and with the key `string`, just like the image bellow.  
Or you can make a POST request with a image file, just make sure to change the key name to 'file' in this case. Also remember to change the request code line inside the `application_keras.py` file
![Request tutorial](/Plots/postman_prt.png)


Furthermore, all the required Python packages can be found in the `requirements.txt` file.

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
Below you can see 2 examples of images that the algorithm can handle well.

![Request tutorial](/Plots/captcha1.png)
![Request tutorial](/Plots/captcha2.png)
