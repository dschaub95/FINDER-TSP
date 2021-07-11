FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt update \
    && apt upgrade -y

RUN apt install -y \
        build-essential \
        git

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version


RUN pip install -U --no-cache-dir \
    numpy \
    pandas \
    networkx \
    matplotlib \
    seaborn \
    statsmodels\
    scipy \
    tsplib95 \
    jupyter \
    notebook 

RUN conda update -y -n base conda

RUN conda create -y --name findervenv python=3.7 
# RUN conda init bash

SHELL ["conda", "run", "-n", "findervenv", "/bin/bash", "-c"]
# RUN conda activate findervenv

RUN pip install -U --no-cache-dir \
    cython==0.29.13 \
    networkx==2.3 \
    numpy==1.17.3 \
    pandas==0.25.2 \
    scipy==1.3.1 \
    tensorflow-gpu==1.14.0 \
    tqdm==4.36.1


# ENV CUDA_HOME=/usr/local/cuda-10.0
# ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
# ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH