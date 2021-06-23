#FROM python:3.7
FROM ubuntu:18.04

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update ##[edited]
RUN apt-get --fix-missing update && apt-get --fix-broken install && apt-get install -y poppler-utils && apt-get install && \
    apt-get install -y libleptonica-dev && ldconfig && apt-get install -y python3.6 && \
    apt-get install -y python3-pip && apt install -y libsm6 libxext6


RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 5000

ENTRYPOINT [ "uwsgi", "--ini", "uwsgi.ini"]
#CMD ['python',"app.py"]