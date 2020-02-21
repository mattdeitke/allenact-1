# From https://github.com/allenai/robothor-challenge/blob/master/Dockerfile

ARG CUDA_VERSION

FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install python3-pip libxrender1 libsm6 xserver-xorg-core xorg python3-venv vim pciutils wget git module-init-tools tzdata

#RUN git clone https://github.com/allenai/robothor-challenge && cd /robothor-challenge && git checkout ee41b40ccb59218f42ace213cdc1d404712e7263 && NVIDIA_VERSION=$NVIDIA_VERSION /bin/bash /robothor-challenge/scripts/install_nvidia.sh
COPY docker/install_nvidia.sh /
RUN NVIDIA_VERSION=$NVIDIA_VERSION /bin/bash /install_nvidia.sh
RUN rm /install_nvidia.sh

RUN pip3 install pipenv==2018.11.26
COPY Pipfile /embodied-rl/
WORKDIR /embodied-rl
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN pipenv install --skip-lock --dev

RUN pipenv run python3 -c "import ai2thor.controller; ai2thor.controller.Controller(download_only=True)"

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . /embodied-rl/
# run in interactive session: pipenv shell
ENTRYPOINT ["docker/docker_entrypoint.sh"]
