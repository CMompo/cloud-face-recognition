FROM python:3.7-slim-buster

MAINTAINER Adrian Sanchez "@CMompo"

RUN mkdir -p /app && \
    apt-get update -y && \
    apt-get install -y build-essential python3-dev libsm6 libxext6 libxrender-dev libglib2.0-0 nfs-common

WORKDIR /app

# We copy just the requirements.txt first to leverage Docker cache in case other data is changed
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

COPY training /app/training
COPY face_recognition /app/face_recognition

COPY api /app/api
COPY tasks/run_prod_server.sh /app/run_prod_server.sh
RUN chmod +x run_prod_server.sh

#RUN mount -t nfs 192.168.1.13:/volume1/datasets/lfw-30-min /app/dataset
#RUN mount -t nfs 192.168.1.13:/volume1/models /app/models

CMD [ "./run_prod_server.sh" ]