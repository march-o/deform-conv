ARG NVIDIA_CONTAINER_MAJOR=23
ARG NVIDIA_CONTAINER_MINOR=11

FROM nvcr.io/nvidia/pytorch:${NVIDIA_CONTAINER_MAJOR}.${NVIDIA_CONTAINER_MINOR}-py3

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"

# Some deps and stuff
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 git sudo tmux

RUN pip install pip --upgrade
# Copy the requirements file into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mmdet and friends
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install mmpretrain

# Compile ops for only specific common cuda architectures, 89 - 40X0, 86 - [30X0, A6000], 80 - A100, 75 - 20X0
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9+PTX"
ENV CUDA_HOME="/usr/local/cuda"

# RUN FORCE_CUDA="1" MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES} mim install "mmcv>=2.0.0" -v
RUN FORCE_CUDA="1" MMCV_WITH_OPS=1 mim install "mmcv>=2.0.0" -v

RUN git clone https://github.com/open-mmlab/mmdetection.git
RUN cd mmdetection && FORCE_CUDA="1" MMCV_WITH_OPS=1 pip install -e . -v

COPY ops_dcnv3 ./ops_dcnv3
RUN cd ops_dcnv3 && sh make.sh

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
ARG USERNAME=user
ARG USER_UID=1001
ARG USER_GID=1002
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support for the non-root user
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME