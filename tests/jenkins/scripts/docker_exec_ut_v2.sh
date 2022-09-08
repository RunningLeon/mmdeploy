#!/bin/bash

## keep container alive
nohup sleep infinity > sleep.log 2>&1 &

## init conda
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup

echo "time-$(date +%Y%m%d%H%M)"
# install sys libs
apt update && apt-get install -y lcov

## parameters
export codebase=$1

export MMDEPLOY_DIR=/root/workspace/mmdeploy
export MMENGIINE_DIR=/root/workspace/mmengine
export MMCV_DIR=/root/workspace/mmcv
export CODEBASE_DIR=/root/workspace/${codebase_fullname}

git clone --depth 1 https://github.com/open-mmlab/mmengine.git $MMENGIINE_DIR
git clone --depth 1 --branch 2.x https://github.com/open-mmlab/mmcv.git $MMCV_DIR
git clone --depth 1 --branch $branch_name $ https://github.com/open-mmlab/${codebase_fullname}.git ${CODEBASE_DIR}

#### TODO: to be removed
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH/\/root\/workspace\/libtorch\/lib:/}

for TORCH_VERSION in 1.8.1 1.9.0 1.10.0 1.11.0 1.12.0
do
    conda activate torch${TORCH_VERSION}
    # export libtorch cmake dir, ran example: /opt/conda/envs/torch1.11.0/lib/python3.8/site-packages/torch/share/cmake/Torch
    export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")
    # need to build for each env
    # TODO add openvino
    mkdir -p $MMDEPLOY_DIR/build && cd $MMDEPLOY_DIR/build
    cmake .. -DMMDEPLOY_BUILD_SDK=ON \
            -DMMDEPLOY_BUILD_EXAMPLES=ON \
            -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
            -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
            -DMMDEPLOY_COVERAGE=ON \
            -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
            -DMMDEPLOY_TARGET_BACKENDS="trt;ort;ncnn" \
            -DMMDEPLOY_SHARED_LIBS=OFF \
            -DTENSORRT_DIR=${TENSORRT_DIR} \
            -DCUDNN_DIR=${CUDNN_DIR} \
            -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
            -Dncnn_DIR=${ncnn_DIR} \
            -DTorch_DIR=${Torch_DIR} \
            -Dpplcv_DIR=${pplcv_DIR} \
            -DMMDEPLOY_TARGET_DEVICES="cuda;cpu"

    make -j $(nproc) && make install

    # sdk tests
    mkdir -p mmdeploy_test_resources/transform
    cp ../tests/data/tiger.jpeg mmdeploy_test_resources/transform/
    ./bin/mmdeploy_tests
    lcov --capture --directory . --output-file coverage.info
    ls -lah coverage.info
    cp coverage.info $MMDEPLOY_DIR/../ut_log/${TORCH_VERSION}_sdk_ut_converage.info

    cd $MMDEPLOY_DIR
    # install mmengine mmcv codebase
    pip install -v $MMENGIINE_DIR
    MMCV_WITH_OPS=1 pip install -v $MMCV_DIR
    pip install -v $CODEBASE_DIR

    # install mmdeploy
    pip install -r $MMDEPLOY_DIR/requirements/tests.txt
    pip install -r $MMDEPLOY_DIR/requirements/runtime.txt
    pip install -r $MMDEPLOY_DIR/requirements/build.txt
    pip install -v $MMDEPLOY_DIR
    ## start python tests
    coverage run --branch --source mmdeploy -m pytest -rsE tests
    coverage xml
    coverage report -m
    cp coverage.xml $MMDEPLOY_DIR/../ut_log/${TORCH_VERSION}_converter_converage.xml
done
echo "time-$(date +%Y%m%d%H%M)"
