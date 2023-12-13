# Use official Ubuntu image as a parent image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables to suppress warnings and to avoid prompts
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn \
    DEBIAN_FRONTEND=noninteractive

# Install packages and clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    wget git build-essential cmake libopenmpi-dev python3-dev zlib1g-dev \
    libgl1-mesa-glx swig libopenblas-dev libsuitesparse-dev libgsl0-dev \
    libfftw3-dev libglpk-dev libdsdp-dev python3.10 python3-pip nano \
    default-mysql-server default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install Optuna, Optuna-dashboard and additional requirements
RUN pip install \
    optuna optuna-dashboard \
    wrds swig \
    git+https://github.com/mikazlopes/FinRLOptimized

# Clone your training-farm repository and install additional requirements
RUN git clone -b adding-optuna https://github.com/mikazlopes/training-farm.git && \
    cd training-farm && \
    pip install -r requirements.txt

# Expose ports for the services
EXPOSE 5678 8080 3306

WORKDIR /training-farm

# Create a cache directory and copy local contents into it
RUN mkdir cache
COPY ./cache ./cache

# Copy entrypoint script into the image
COPY entrypoint.sh /entrypoint.sh

# Set the entrypoint script to be executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Command to run the application, default arguments can be overridden at runtime
CMD ["python", "hp-tuner.py", "--period_years", "10", "--gpu_id", "0"]



