FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install --no-install-recommends software-properties-common libgl1-mesa-dev wget libssl-dev

RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get -y install --no-install-recommends python3.9-dev python3.9-distutils python3-pip python3.9-venv
# change default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# clear cache
RUN rm -rf /var/lib/apt/lists/*

# create virtual environment
RUN python -m venv /venv
RUN /venv/bin/pip install -U pip distlib setuptools

# and add aliases
RUN echo 'alias python="/venv/bin/python"' >> ~/.bashrc
RUN echo 'alias pip="/venv/bin/pip"' >> ~/.bashrc
RUN echo 'alias jupyter-lab="/venv/bin/jupyter-lab"' >> ~/.bashrc
RUN echo 'alias pytest="/venv/bin/pytest"' >> ~/.bashrc
RUN echo 'alias tensorboard="/venv/bin/tensorboard"' >> ~/.bashrc


WORKDIR /tmp
COPY pyproject.toml .
COPY src/ src/
RUN /venv/bin/pip install -e .[dev]

WORKDIR /workspace

SHELL ["/bin/bash", "-c"]