#FROM tensorflow/tensorflow:latest-gpu-py3
FROM tensorflow/tensorflow:1.4.0-rc1-gpu-py3

ADD . /app
WORKDIR /app

RUN pip3 install -r requirements.txt
RUN mkdir -p ./tensorboard
RUN tensorboard --logdir=./tensorboard &

EXPOSE 6006

CMD python3 gan-script.py
