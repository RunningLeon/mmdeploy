export MMDEPLOY_DIR=/root/workspace/mmdeploy

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
    -Dpplcv_DIR=${pplcv_DIR} \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu"
make -j $(nproc)
make install && cd $MMDEPLOY_DIR
