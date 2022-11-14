# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import yaml

from mmdeploy.utils import get_backend, get_task_type, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate markdown table from yaml file')
    parser.add_argument('yml_file', help='Input yaml config path.')
    parser.add_argument('output', help='Output markdown file path.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert osp.exists(args.yml_file), f'File not exists: {args.yml_file}'
    output_dir, _ = osp.split(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # selected backends
    backends = [
        'onnxruntime', 'tensorrt', 'torchscript', 'pplnn', 'openvino', 'ncnn'
    ]
    header = ['model', 'task'] + backends
    aligner = [':--'] * 2 + [':--:'] * len(backends)

    # simple function to write a row data
    def write_row_f(writer, row):
        writer.write('|' + '|'.join(row) + '|\n')

    print(f'Processing {args.yml_file}')
    with open(args.yml_file, 'r') as reader, open(args.output, 'w') as writer:
        config = yaml.load(reader, Loader=yaml.FullLoader)
        write_row_f(writer, header)
        write_row_f(writer, aligner)
        repo_url = config['globals']['repo_url']
        for model in config['models']:
            config_url = osp.join(repo_url, model['model_configs'][0])
            config_url, _ = osp.split(config_url)
            supported_backends = {b: 'N' for b in backends}
            task = ''
            for pipeline in model['pipelines']:
                deploy_cfg = load_config(pipeline['deploy_config'])[0]
                if not task:
                    task = get_task_type(deploy_cfg).value
                backend_type = get_backend(deploy_cfg).value
                supported_backends[backend_type] = 'Y'
            # convert to list with same order
            supported_backends = [supported_backends[b] for b in backends]
            model_name = f'[{model["name"]}]({config_url})'
            row = [model_name, task] + supported_backends
            write_row_f(writer, row)

        print(f'Save to {args.output}')


if __name__ == '__main__':
    main()
