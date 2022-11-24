From nvcr.io/nvidia/pytorch:21.08-py3

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get -y install \
    wget \ 
    zip \ 
    htop \
    screen \
    libgl1-mesa-glx

# mkdir and download weight
RUN mkdir -p /weight
WORKDIR /weight

RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt
Run wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt
Run wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt

# pip install required packages
RUN pip install \
    seaborn \
    thop \
    AIMaker==1.4.3 \
    AIMakerMonitor==1.0.5 \
    mlflow==1.24.0 \
    boto3==1.21.0 \
    AIMakerMonitor==1.0.5 \
    supervisor==4.1.0 \
    jupyterlab==3.4.7

#Jupyter
ARG ASUS_CLOUDINFRA_DIR=/opt/ASUSCloudInfra
ENV JUPYTER_CONFIG_DIR=${ASUS_CLOUDINFRA_DIR}
COPY ./supervisor/jupyter_lab_config.py ${JUPYTER_CONFIG_DIR}/

#supervisor
COPY ./supervisor/supervisord.conf /etc/supervisor/supervisord.conf
COPY ./supervisor/entrypoint.sh /usr/local/bin/
COPY ./supervisor/start_jupyter.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start_jupyter.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /yolov7
COPY ./yolov7 /yolov7
RUN pip install -r /yolov7/requirements.txt
COPY ./asus/* /yolov7/

#jupyter-example
COPY ./jupyter-example /jupyter-example/

RUN mkdir -p /datasetTemp
RUN chmod -R 777 /datasetTemp
RUN mkdir -p /temp
RUN chmod -R 777 /temp
WORKDIR /yolov7
CMD ["entrypoint.sh"]