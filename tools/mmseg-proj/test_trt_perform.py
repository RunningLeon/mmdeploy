# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import subprocess

import yaml
from easydict import EasyDict as edict
from tqdm import tqdm


def run_cmd(cmd_lines, log_path):
    """
    Args:
        cmd_lines: (list[str]): A command in multiple line style.
        log_path (str): Path to log file.

    Returns:
        int: error code.
    """
    sep = '\\'
    cmd_for_run = f' {sep}\n'.join(cmd_lines) + '\n'
    parent_path = osp.split(log_path)[0]
    os.makedirs(parent_path, exist_ok=True)

    print(100 * '-')
    print(f'Start running cmd\n{cmd_for_run}')
    print(f'Logging log to \n{log_path}')

    with open(log_path, 'w', encoding='utf-8') as file_handler:
        # write cmd
        file_handler.write(f'Command:\n{cmd_for_run}\n')
        file_handler.flush()
        process_res = subprocess.Popen(
            cmd_for_run,
            cwd=os.getcwd(),
            shell=True,
            stdout=file_handler,
            stderr=file_handler)
        process_res.wait()
        return_code = process_res.returncode

    if return_code != 0:
        print(f'Got shell return code={return_code}')
        with open(log_path, 'r') as f:
            content = f.read()
            print(f'Log message\n{content}')
    return return_code


def parse_args():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('input_txt')
    parser.add_argument('config')
    parser.add_argument('checkpoint_dir')
    parser.add_argument('--plantform', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert osp.exists(args.input_txt)
    with open(args.input_txt, 'r') as f:
        onnx_paths = f.readlines()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    nrof_file = len(onnx_paths)
    for i in tqdm(range(nrof_file), desc='Processing'):
        onnx_path = osp.join(args.checkpoint_dir, onnx_paths[i].strip())
        if not osp.exists(onnx_path):
            logging.warning(f'File not exists: {onnx_path}')
            continue
        work_dir, _ = osp.split(onnx_path)
        engine_dir = osp.join(work_dir, args.plantform, config.precision)
        os.makedirs(engine_dir, exist_ok=True)
        engine_path = osp.join(engine_dir, 'end2end.engine')
        log_path = osp.join(engine_dir, 'trtexec.log')
        time_json = osp.join(engine_dir, 'time.json')
        output_json = osp.join(engine_dir, 'output.json')
        profile_json = osp.join(engine_dir, 'profile.json')
        layerinfo_json = osp.join(engine_dir, 'layerinfo.json')
        cmd_lines = [
            'trtexec', '--useCudaGraph', f'--onnx={onnx_path}', '--verbose'
        ]
        cmd_lines.append(f'--saveEngine={engine_path}')
        cmd_lines.append(f'--exportTimes={time_json}')
        cmd_lines.append(f'--exportOutput={output_json}')
        cmd_lines.append(f'--exportProfile={profile_json}')
        cmd_lines.append(f'--exportLayerInfo={layerinfo_json}')
        cmd_lines.extend(
            [f'--avgRuns={config.avgRuns}', f'--workspace={config.workspace}'])
        input_shape = f'input:{config.maxBatch}x3x{config.input_shape}'
        cmd_lines.extend([
            f'--shapes=input:{input_shape}',
            f'--iterations={config.iterations}'
        ])
        cmd_lines.extend([
            f'--warmUp={config.warmUp}', f'--streams={config.streams}',
            f'--sparsity={config.sparsity}'
        ])
        if config.precision == 'fp16':
            cmd_lines.append('--fp16')
        elif config.precision == 'int8':
            cmd_lines.extend(['--fp16 --int8'])
            if osp.exists(config.calib):
                cmd_lines.append(f'--calib={config.calib}')
        elif config.precision == 'best':
            cmd_lines.append('--best')

        run_cmd(cmd_lines, log_path)


if __name__ == '__main__':
    main()
