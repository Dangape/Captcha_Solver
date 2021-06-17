from setuptools import setup

setup(
    name='CAPTCHA ML',
    description="Translate CAPTCHA images into text",
    author="Daniel Bemerguy",
    author_email="daniel06197@gmail.com",
    install_requires=['flask', 'numpy','string','pytesseract','opencv-python','imageio','io','base64','pillow','imutils'],
    python_requires='>=3.6',
)