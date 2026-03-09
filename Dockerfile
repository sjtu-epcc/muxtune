# Driver Version: 515.48.07
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# # Driver Version: > 525
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install common tool & conda
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install wget -y && \
    apt install git -y && \
    apt install curl -y && \
    apt install vim -y && \
    apt install bc && \
    apt-get install net-tools -y && \
    apt install ssh -y && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    mkdir -p /opt/conda/envs/muxtune && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Workspace
WORKDIR /app

# Construct conda environment
COPY requirements.txt requirements.txt
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name muxtune python=3.10 -y && \
    conda activate muxtune && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \ 
    && python3.10 -m pip install -r requirements.txt

# CUDA path
ENV CUDA_PATH=/usr/local/cuda
# ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/compat:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
# If the host driver is new enough, don't add `$CUDA_PATH/compat` otherwise will occur: 
#   `CUDA Error: system has unsupported display driver / cuda driver combination`.
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV CUDNN_PATH=/usr/include
# Transformer engine path
ENV NVTE_FRAMEWORK=pytorch

# Install flash-attn (this is resource-intensive and time-consuming)
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate muxtune && \
    MAX_JOBS=16 pip install flash-attn==2.6.3 --no-build-isolation

# Install megatron-lm 
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate muxtune && \
    pip install --upgrade setuptools && \
    MAX_JOBS=16 pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable && \
    git clone --recursive https://github.com/DicardoX/Megatron-LM.git && \
    cd Megatron-LM && \ 
    MAX_JOBS=16 pip install -e .

# Install nvidia apex
# NOTE: This requires the version of torch cuda and the CUDA version are the same.
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate muxtune && \
    pip install --upgrade setuptools && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    MAX_JOBS=16 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Copy workspace
COPY . .

# Enterpoint for bash shell
ENTRYPOINT ["/bin/bash"]
