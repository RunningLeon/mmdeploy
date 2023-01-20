# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import logging
import os
import os.path as osp
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description='Copy ONNX models for onnx2tensorrt.')
    parser.add_argument(
        'src_dir', help='Source mmdeploy_regression_working_dir')
    parser.add_argument('dst_dir', help='Dest mmdeploy_regression_working_dir')
    parser.add_argument('--codebase', default='mmpose')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    onnx_files = glob.glob(
        osp.join(args.src_dir, args.codebase, '*', 'tensorrt',
                 '**/end2end.onnx'),
        recursive=True)
    num = len(onnx_files)
    logging.info(f'Found totally {num} onnx files for tensorrt backend')
    for src_file in onnx_files:
        dst_file = src_file.replace(args.src_dir, args.dst_dir)
        _dst_dir, _ = osp.split(dst_file)
        os.makedirs(_dst_dir, exist_ok=True)
        shutil.copy(src_file, dst_file)


if __name__ == '__main__':
    main()
