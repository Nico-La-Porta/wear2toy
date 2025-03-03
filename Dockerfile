# Use the official Python image
FROM python:3.12.7

# Update package lists and install required dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*  # Clean up APT cache

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

RUN pip install python-dotenv==1.0.1
RUN pip install numpy==2.2.3
RUN pip install matplotlib==3.10.0
RUN pip install scikit_learn==1.6.1
RUN pip install seaborn==0.13.2
RUN pip install tqdm==4.67.1
RUN pip install torch==2.5.1
RUN pip install loguru==0.7.2
RUN pip install optuna==4.1.0

# Set the working directory
WORKDIR /app