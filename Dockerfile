FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to prevent interactive prompts and reduce cache size
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.12.7

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    liblzma-dev

# Download and build Python
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xvf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations --with-lto --prefix=/usr/local \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py

# Clean up
RUN apt-get remove -y wget build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.12 as the default
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm -rf requirements.txt
