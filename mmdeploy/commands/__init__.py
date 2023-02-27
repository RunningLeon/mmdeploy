# Copyright (c) OpenMMLab. All rights reserved.
from .command import COMMAND_REGISTRY
from .deploy import DeployCommand
from .profiler import ProfilerCommand
from .test import TestCommand
from .torch2onnx import Torch2OnnxCommand

__all__ = [
    'COMMAND_REGISTRY', 'DeployCommand', 'ProfilerCommand',
    'Torch2OnnxCommand', 'TestCommand'
]
