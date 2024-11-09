FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# install ubuntu dependencies and clear cache
RUN apt-get -y update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential software-properties-common && \
    apt-get -y update && \
    add-apt-repository universe && \
    apt-get -y install python3 && \
    apt-get -y install python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN dpkg --add-architecture i386
RUN apt-get update -y
RUN apt-get install libc6-dbg -y
RUN apt-get install libc6-dbg:i386 -y

# install python dependencies0
RUN pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
COPY requirements/requirements-envpool.txt requirements.txt
COPY requirements/requirements-atari.txt requirements-atari.txt
RUN pip install -r requirements.txt
RUN pip install -r requirements-atari.txt
RUN pip install gymnasium[other]
RUN pip install wandb --user

# Create useful directories
RUN mkdir /dataset /tmp_log /final_log /src /wandb