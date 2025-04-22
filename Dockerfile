FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Ubuntu dependencies and clear cache
RUN apt-get -y update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential software-properties-common ca-certificates && \
    apt-get -y update && \
    add-apt-repository universe && \
    apt-get -y install python3 && \
    apt-get -y install python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    update-ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install lightly
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install wandb --user

# Set CA certificate environment variables
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Create useful directories
RUN mkdir /dataset /tmp_log /final_log /src /wandb