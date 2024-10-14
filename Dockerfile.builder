# This image purpose is to build "nvidia_resiliency_ext" wheels using different Python versions.
# There are python3.10, python3.11 and python3.12 installed.
# Base image is CUDA, as Straggler Detection package uses CUPTI.
# Wheel for Python3.10 can be created with "python3.10 -m build --wheel" etc.

# Choose a base CUDA image from NVIDIA
# nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04, nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04 etc.
ARG BASE_CUDA_IMG=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM ${BASE_CUDA_IMG}

# Set environment variables to non-interactive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Repo with Pythons
RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

# Install common dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3.11 python3.11-dev python3.11-distutils \
    python3.12 python3.12-dev python3.12-distutils \
    wget curl build-essential gcc-10 g++-10\
    && rm -rf /var/lib/apt/lists/*

# Install pip for each Python version
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install deps,
# FIXME: for some reason six needs to be manually updated
# otherwise wheel building fails with: ModuleNotFoundError: No module named 'six'
RUN python3.10 -m pip install build poetry && \
    python3.11 -m pip install build poetry && \
    python3.12 -m pip install build poetry && \
    python3.10 -m pip install -U six && \
    python3.11 -m pip install -U six && \
    python3.12 -m pip install -U six

# Set the working directory
WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-c"]
