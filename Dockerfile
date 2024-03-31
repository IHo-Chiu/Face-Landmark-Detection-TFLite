# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:12.3.2-base-ubuntu20.04

# Skip any interactivate message
ENV DEBIAN_FRONTEND noninteractive

# Install linux packages
RUN apt update && apt upgrade -y
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y libgl1
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext6
RUN apt-get install -y libglib2.0-0

# Install pip packages
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install tflite-runtime==2.11.0
RUN pip install tensorflow

# Solve cuda package issue
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8.4.0 /usr/lib/libcudnn.so
RUN ln -s /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcublas.so.11.9.2.110 /usr/lib/libcublas.so

# Add code
ADD Inference /app
WORKDIR /app
