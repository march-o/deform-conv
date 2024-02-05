FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Opencv dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# create non-root user and set the default user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME 
USER $USERNAME

# make all python tools installed by pip accesible
ENV PATH=$PATH:/home/$USERNAME/.local/bin

RUN pip install pip --upgrade

# Copy the requirements file into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set some cuda env vars for mmcv
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

# Cuda arch list
ENV TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"

# Install mm things
RUN pip install -U openmim
RUN mim install mmengine

RUN mim install "mmcv>=2.0.0" -v

RUN sudo chown -R $USERNAME .

RUN git clone https://github.com/open-mmlab/mmdetection.git
RUN cd mmdetection && pip install -e . -v

# File permissions
RUN sudo chown -R $USERNAME /opt/conda
RUN sudo chown -R $USERNAME /usr/local
RUN sudo chown -R $USERNAME /home/$USERNAME

RUN ../ops_dcnv3/install.sh

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=

CMD ["/bin/bash"]