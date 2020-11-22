FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL name="Ildar Kashaev"
LABEL description="pycuda"
RUN apt -qq update  && \
    apt -qq install build-essential python3-pip && \
    pip3 install pycuda
