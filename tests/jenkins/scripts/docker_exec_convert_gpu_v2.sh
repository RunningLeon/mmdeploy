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

## func
function getFullName() {
    local codebase_=$1
    codebase_fullname=""
    if [ "$codebase_" = "mmdet" ]; then codebase_fullname="mmdetection"; fi
    if [ "$codebase_" = "mmcls" ]; then codebase_fullname="mmclassification"; fi
    if [ "$codebase_" = "mmdet3d" ]; then codebase_fullname="mmdetection3d"; fi
    if [ "$codebase_" = "mmedit" ]; then codebase_fullname="mmediting"; fi
    if [ "$codebase_" = "mmocr" ]; then codebase_fullname="mmocr"; fi
    if [ "$codebase_" = "mmpose" ]; then codebase_fullname="mmpose"; fi
    if [ "$codebase_" = "mmrotate" ]; then codebase_fullname="mmrotate"; fi
    if [ "$codebase_" = "mmseg" ]; then codebase_fullname="mmsegmentation"; fi
}

function getBranchName() {
    local codebase_=$1
    branch_name=""
    if [ "$codebase_" = "mmdet" ]; then branch_name="3.x"; fi
    if [ "$codebase_" = "mmcls" ]; then branch_name="1.x"; fi
    if [ "$codebase_" = "mmdet3d" ]; then branch_name="dev-1.x"; fi
    if [ "$codebase_" = "mmedit" ]; then branch_name="1.x"; fi
    if [ "$codebase_" = "mmocr" ]; then branch_name="1.x"; fi
    if [ "$codebase_" = "mmpose" ]; then branch_name="1.x"; fi
    if [ "$codebase_" = "mmrotate" ]; then branch_name="dev-1.x"; fi
    if [ "$codebase_" = "mmseg" ]; then branch_name="1.x"; fi
}

## parameters
export codebase=$1
getFullName $codebase
getBranchName $codebase

#### TODO: to be removed
export ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH/\/root\/workspace\/libtorch\/lib:/}
export ONNXRUNTIME_VERSION=1.8.1

## avoid dataloader OOM error of too many workers
sed -i 's/workers_per_gpu=model_cfg.data.workers_per_gpu/workers_per_gpu=1/g' $MMDEPLOY_DIR/tools/test.py

echo "time-$(date +%Y%m%d%H%M)"
export MMDEPLOY_DIR=/root/workspace/mmdeploy
export MMENGIINE_DIR=/root/workspace/mmengine
export MMCV_DIR=/root/workspace/mmcv
export CODEBASE_DIR=/root/workspace/${codebase_fullname}
git clone --depth 1 https://github.com/open-mmlab/mmengine.git $MMENGIINE_DIR
git clone --depth 1 --branch 2.x https://github.com/open-mmlab/mmcv.git $MMCV_DIR
git clone --depth 1 --branch $branch_name $ https://github.com/open-mmlab/${codebase_fullname}.git ${CODEBASE_DIR}

## build mmdeploy
ln -s /root/workspace/mmdeploy_benchmark $MMDEPLOY_DIR/data

for TORCH_VERSION in 1.8.1 1.9.0 1.10.0 1.11.0 1.12.0
do
    start_time_per_torch=$SECONDS
    conda activate torch${TORCH_VERSION}
    # install mmengine mmcv codebase
    pip install -v $MMENGIINE_DIR
    MMCV_WITH_OPS=1 pip install -v $MMCV_DIR
    pip install -v $CODEBASE_DIR

    # export libtorch cmake dir, ran example: /opt/conda/envs/torch1.11.0/lib/python3.8/site-packages/torch/share/cmake/Torch
    export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")
    # need to build for each env
    # TODO add openvino
    mkdir -p $MMDEPLOY_DIR/build && cd $MMDEPLOY_DIR/build
    cmake .. -DMMDEPLOY_BUILD_SDK=ON \
            -DMMDEPLOY_BUILD_EXAMPLES=ON \
            -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
            -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
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
    make -j $(nproc)
    make install && cd $MMDEPLOY_DIR

    pip install -r requirements/tests.txt
    pip install -r requirements/runtime.txt
    pip install -r requirements/build.txt
    pip install -v .

    ## start regression
    log_dir=/root/workspace/mmdeploy_regression_working_dir/${codebase}/torch${TORCH_VERSION}
    log_path=${log_dir}/convert.log
    mkdir -p ${log_dir}
    # ignore pplnn as it's too slow
    python ./tools/regression_test.py \
        --codebase ${codebase} \
        --work-dir ${log_dir} \
        --device cuda:0 \
        --backends onnxruntime tensorrt ncnn torchscript openvino \
        --performance \
        2>&1 | tee ${log_path}
    elapsed_per_torch=$((SECONDS - start_time_per_torch)))
    eval "echo Elapsed($TORCH_VERSION): $(date -ud "@$elapsed_per_torch" +'$((%s/3600/24)) days %H hr %M min %S sec')"
done

echo "time-$(date +%Y%m%d%H%M)"
