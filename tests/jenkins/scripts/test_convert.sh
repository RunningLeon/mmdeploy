#!/bin/bash

## parameters

docker_image=$1
codebase_list=($2)
max_job_nums=${3:-'4'}
mmdeploy_branch=${4:-'master'}
repo_url=${5:-'https://github.com/open-mmlab/mmdeploy.git'}
script_version=${6:-''}

date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/convert_log/${date_snap}/${time_snap}
mkdir -p ${log_dir}

## make & init mutex
trap "exec 1000>&-; exec 1000<&-;exit 0" 2
mkfifo mutexfifo
exec 1000<>mutexfifo
rm -rf mutexfifo
for ((n=1;n<=${max_job_nums};n++))
do
    echo >&1000
done

## docker run cmd for convert
for codebase in ${codebase_list[@]}
do
    read -u1000
    {
        container_name=convert-${codebase}-${time_snap}
        container_id=$(
            docker run -itd \
                --gpus all \
                -v /data2/checkpoints/${codebase}:/root/workspace/mmdeploy_checkpoints \
                -v ${log_dir}:/root/workspace/mmdeploy_regression_working_dir \
                -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
                -v ~/mmdeploy/tests/jenkins/scripts:/root/workspace/mmdeploy_script \
                --name ${container_name} \
                ${docker_image} /bin/bash
        )
        echo "container_id=${container_id}"
        nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} &&\
        /root/workspace/mmdeploy_script/docker_exec_convert_gpu${script_version}.sh ${codebase}" > ${log_dir}/${codebase}.log 2>&1 &
        wait
        docker stop $container_id
        echo "${codebase} convert finish!"
        cat ${log_dir}/${codebase}.log
        echo >&1000
    }&
done

wait
exec 1000>&-
exec 1000<&-
