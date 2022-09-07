#!/bin/bash

## parameters
# export docker_image=mmdeploy-ci-ubuntu-18.04
export docker_image=$1

date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
# docker run cmd for build
log_dir=/data2/regression_log/build_log/${date_snap}/${time_snap}
mkdir -p ${log_dir}
container_name=build-${time_snap}
container_id=$(
    docker run -itd \
    --gpus all \
    -v ~/mmdeploy/tests/jenkins/scripts:/root/workspace/mmdeploy_script \
    --name ${container_name} \
    ${docker_image} /bin/bash
)
echo "container_id=${container_id} --name ${container_name}"

nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch master --recursive https://github.com/open-mmlab/mmdeploy.git &&\
/root/workspace/mmdeploy_script/docker_exec_build.sh" > ${log_dir}/build.log 2>&1 &

wait