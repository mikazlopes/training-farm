# Use official Ubuntu image as a parent image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set environment variables to suppress warnings and to avoid prompts
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn \
    DEBIAN_FRONTEND=noninteractive

# Install packages and clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    wget git build-essential cmake libopenmpi-dev python3-dev zlib1g-dev \
    libgl1-mesa-glx swig libopenblas-dev libsuitesparse-dev libgsl0-dev \
    libfftw3-dev libglpk-dev libdsdp-dev python3.10 python3-pip nano \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install requirements
RUN pip install \
    wrds \
    swig \
    optuna \
    git+https://github.com/mikazlopes/FinRLOptimized

# Clone your training-farm repository and install additional requirements
RUN git clone https://github.com/mikazlopes/training-farm.git && \
    cd training-farm && \
    pip install -r requirements.txt

EXPOSE 5678

WORKDIR /training-farm

# Command to run the application
CMD ["python", "main.py"]
