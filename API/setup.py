from setuptools import setup

setup(
    name='CAPTCHA_ML',
    description="Translate CAPTCHA images into text",
    author="Daniel Bemerguy",
    author_email="daniel06197@gmail.com",
    install_requires=['flask','numpy','pytesseract','opencv-python','imageio','pillow','imutils','uwsgi'],
    python_requires='>=3.6',
)