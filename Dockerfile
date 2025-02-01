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

# install python dependencies0
RUN pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install wandb --user

# Create useful directories
RUN mkdir /dataset /tmp_log /final_log /src /wandb