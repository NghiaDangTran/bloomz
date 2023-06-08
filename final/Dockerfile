FROM nvidia/cuda:11.5.0-base-ubuntu20.04

WORKDIR /home

COPY requirements.txt /home/requirements.txt

# Add the deadsnakes PPA for Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common libgl1-mesa-glx cmake protobuf-compiler && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10 and dev packages
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip  && \
    rm -rf /var/lib/apt/lists/*

# Install virtualenv
RUN pip3 install virtualenv

# Create a virtual environment with Python 3.10
RUN virtualenv -p python3.10 venv

# Activate the virtual environment
ENV PATH="/home/venv/bin:$PATH"

# Install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install --default-timeout=10000000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip3 install --default-timeout=10000000 -r requirements.txt

# Copy files
COPY api.py /home/api.py

# Set the PYTHONPATH and LD_LIBRARY_PATH environment variable to include the CUDA libraries
ENV PYTHONPATH=/usr/local/cuda-11.5/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64

# Set the CUDA_PATH and CUDA_HOME environment variable to point to the CUDA installation directory
ENV CUDA_PATH=/usr/local/cuda-11.5
ENV CUDA_HOME=/usr/local/cuda-11.5

# Set the default command
CMD ["sh", "-c", ". /home/venv/bin/activate && python api.py $@"]
