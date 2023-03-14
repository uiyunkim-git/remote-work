ARG UBUNTU_VER

FROM ubuntu:20.04

WORKDIR /code

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get install -y curl
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN apt-get install -y --no-install-recommends \
    python3-setuptools python3.10-distutils

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt

COPY ./src ./src

COPY entrypoint.sh /code/entrypoint.sh
RUN chmod +x /code/entrypoint.sh
EXPOSE 80

ENTRYPOINT export PATH="/bin:$PATH" &&\
    sh /code/entrypoint.sh

