FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL name="Ildar Kashaev"
LABEL description="research"
RUN apt update  && \
    apt -y install build-essential python3-pip && \
    pip3 install -r requirements.txt
