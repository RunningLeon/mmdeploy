# Copyright (c) OpenMMLab. All rights reserved.
import abc

from mmengine import Registry

COMMAND_REGISTRY = Registry('command')


class Command(abc.ABC):

    @abc.abstractmethod
    def add_subparser(self, name, parser):
        raise NotImplementedError
