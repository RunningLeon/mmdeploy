FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0

# important dependencies
ARG OPENCV_VERSION==4.5.4.60
ARG PPLCV_VERSION=0.7.0

# backends
ARG ONNXRUNTIME_VERSION=1.8.1
ARG PPLNN_VERSION=0.8.1
ARG NCNN_VERSION=20221128
ARG TENSORRT_VERSION=8.2.3.0
# tensorrt tar file url
ARG TENSORRT_URL

# Whether to change pip source to aliyun
ARG USE_SRC_INSIDE=true

USER root
WORKDIR /root/workspace

ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"
ENV CUDA_VERSION=11.3
ENV CUDA_INT=113

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libprotobuf-dev protobuf-compiler \
        gcc-7 \
        g++-7 \
        git \
        vim \
        wget \
        libopencv-dev \
        cmake \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# install jdk, onnxruntime, openvino, and other python packages
RUN wget https://download.java.net/java/GA/jdk18/43f95e8614114aeaa8e8a5fcf20a682d/36/GPL/openjdk-18_linux-x64_bin.tar.gz &&\
    tar xvf openjdk-18_linux-x64_bin.tar.gz && rm -rf openjdk-18_linux-x64_bin.tar.gz && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    pip3 install onnxruntime-gpu==${ONNXRUNTIME_VERSION} &&\
    pip3 install openvino-dev[onnx] pycuda openmim cmake &&\
    pip3 install opencv-python==${OPENCV_VERSION} opencv-python-headless==${OPENCV_VERSION} opencv-contrib-python==${OPENCV_VERSION} &&\
    pip3 install torch==${TORCH_VERSION}+cu${CUDA_INT} torchvision==${TORCHVISION_VERSION}+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html

# create env
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA}/lib64/:$LD_LIBRARY_PATH
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA}
ENV JAVA_HOME=/root/workspace/jdk-18
ENV PATH=$JAVA_HOME/bin:$PATH
ENV ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:$LD_LIBRARY_PATH

### install ppl.nn
RUN git clone --depth 1 --branch v${PPLNN_VERSION} --recursive https://github.com/openppl-public/ppl.nn.git &&\
    cd ppl.nn &&\
    ./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_USE_CUDA=ON -DPPLNN_ENABLE_PYTHON_API=ON && \
    cd ./python/package && \
    ./build.sh && \
    cd /tmp/pyppl-package/dist && \
    pip3 install pyppl*.whl

ENV PPLNN_DIR=/root/workspace/ppl.nn
ENV pplnn_DIR=${PPLNN_DIR}/pplnn-build/install/lib/cmake/ppl
ENV PYTHONPATH=/root/workspace/ppl.nn/install/lib:$PYTHONPATH

### build ncnn
RUN git clone --depth 1 --branch ${NCNN_VERSION} --recursive https://github.com/Tencent/ncnn.git &&\
    cd ncnn &&\
    export NCNN_DIR=$(pwd) &&\
    mkdir -p build && cd build &&\
    cmake -DNCNN_VULKAN=OFF -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DCMAKE_INSTALL_PREFIX=$NCNN_DIR/install .. &&\
    make -j $(nproc) && make install &&\
    cd $NCNN_DIR/python &&\
    pip3 install -e .

ENV ncnn_DIR=/root/workspace/ncnn/install/lib/cmake/ncnn
ENV PYTHONPATH=/root/workspace/ncnn/python:$PYTHONPATH

### install ppl.cv
RUN git clone --depth 1 --branch v${PPLCV_VERSION} https://github.com/openppl-public/ppl.cv.git  &&\
    cd ppl.cv &&\
    ./build.sh cuda

ENV pplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl

### install tensorrt
RUN wget $TENSORRT_URL -C /root/workspace && \
    tar -zxvf /root/workspace/TensorRT-${TENSORRT_VERSION}*.tar.gz &&\
    ln -sf /root/workspace/TensorRT-${TENSORRT_VERSION} /root/workspace/TensorRT  &&\
    export PY_VERSION=$(python3 -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}') && \
    pip3 install /root/workspace/TensorRT/python/tensorrt-*-cp${PY_VERSION}-none-linux_x86_64.whl

ENV TENSORRT_DIR=/root/workspace/TensorRT
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
