from typing import Union

import mmcv


def is_dynamic_batch(deploy_cfg: Union[str, mmcv.Config],
                     input_name: str = 'input'):
    """Check if input batch is dynamic."""

    # load deploy_cfg
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    if not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    # check if dynamic axes exist
    dynamic_axes = deploy_cfg['pytorch2onnx'].get('dynamic_axes', None)
    if dynamic_axes is None:
        return False

    # check if given input name exist
    input_axes = dynamic_axes.get(input_name, None)
    if input_axes is None:
        return False

    # check if 2 and 3 in input axes
    if 0 in input_axes:
        return True

    return False


def is_dynamic_shape(deploy_cfg: Union[str, mmcv.Config],
                     input_name: str = 'input'):
    """Check if input shape is dynamic."""

    # load deploy_cfg
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    if not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    # check if dynamic axes exist
    dynamic_axes = deploy_cfg['pytorch2onnx'].get('dynamic_axes', None)
    if dynamic_axes is None:
        return False

    # check if given input name exist
    input_axes = dynamic_axes.get(input_name, None)
    if input_axes is None:
        return False

    # check if 2 and 3 in input axes
    if 2 in input_axes and 3 in input_axes:
        return True

    return False
