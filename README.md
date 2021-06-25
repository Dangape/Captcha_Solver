# Captcha Machine Learning Project
## Running instructions

All the needed files are inside the `API_KERAS` folder. In order to run the API and create the container you will need to run the following code on your terminal:
```
docker-compose build && docker-compose up
```

If you want to download the conatiner directly from my dokcer hub profile just run the following code:
```
docker pull dangape/api_keras
```

In order to test the container just run the image using the following command:
```
docker run dangape/api_keras
```

Use postman or another request app to make a POST request with a base64 string image and with the name `string`, just like the image bellow:
![Request tutorial](/Plots/postman_prt.png)


Furthermore, all the required Python packages can be found in the `requirements.txt` file.