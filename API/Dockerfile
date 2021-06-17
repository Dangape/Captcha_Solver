FROM python:3.6

WORKDIR /user/source/app

COPY setup.py ./
COPY API.py ./API.py

RUN pip install -e .

COPY . .

ENTRYPOINT ["uwsgi", "--ini", "uwsgi.ini"]