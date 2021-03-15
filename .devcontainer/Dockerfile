# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.163.1/containers/python-3/.devcontainer/base.Dockerfile

FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc=4:7.4.0-1ubuntu2.3 \
        python-pip\
        python-setuptools=39.0.1-2 \
        python-wheel=0.30.0-0.2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    cython==0.29.13 \
    networkx==2.3 \
    numpy==1.17.3 \
    pandas==0.25.2 \
    scipy==1.3.1 \
    tensorflow-gpu==1.14.0 \
    tqdm==4.36.1


# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1