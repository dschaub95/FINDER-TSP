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

RUN conda install pip -y

RUN conda update -y -n base conda

RUN conda create -y --name FINDER_TSP python=3.7 
RUN conda init bash
# SHELL ["conda", "run", "-n", "FINDER_TSP", "/bin/bash", "-c"]
# RUN conda activate FINDER_TSP


# ENV CUDA_HOME=/usr/local/cuda-10.0
# ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
# ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH