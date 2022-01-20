# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task, get_input_shape, get_root_logger
from .mmsegmentation import MMSEG_TASK


def process_model_config(model_cfg: mmcv.Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (mmcv.Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        mmcv.Config: the model config after processing.
    """
    from mmseg.apis.inference import LoadImage
    cfg = model_cfg.copy()

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    # for static exporting
    if input_shape is not None:
        cfg.data.test.pipeline[1]['img_scale'] = tuple(input_shape)
    cfg.data.test.pipeline[1]['transforms'][0]['keep_ratio'] = False
    cfg.data.test.pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]

    return cfg


@MMSEG_TASK.register_module(Task.SEGMENTATION.value)
class Segmentation(BaseTask):
    """Segmentation task class.

    Args:
        model_cfg (mmcv.Config): Original PyTorch model config file.
        deploy_cfg (mmcv.Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str):
        super(Segmentation, self).__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Optional[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .segmentation_model import build_segmentation_model
        model = build_segmentation_model(
            model_files, self.model_cfg, self.deploy_cfg, device=self.device)
        return model.eval()

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           cfg_options: Optional[Dict] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                defaults to `None`.
            cfg_options (dict): Optional config key-pair parameters.

        Returns:
            nn.Module: An initialized torch model generated by OpenMMLab
                codebases.
        """
        from mmseg.apis import init_segmentor
        from mmdeploy.codebase.mmseg.deploy import convert_syncbatchnorm
        model = init_segmentor(self.model_cfg, model_checkpoint, self.device)
        model = convert_syncbatchnorm(model)

        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Sequence[int] = None) \
            -> Tuple[Dict, torch.Tensor]:
        """Create input for segmentor.

        Args:
            imgs (Any): Input image(s), accepted data type are `str`,
                `np.ndarray`, `torch.Tensor`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmseg.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
        test_pipeline = Compose(cfg.data.test.pipeline)
        data_list = []
        for img in imgs:
            # prepare data
            data = dict(img=img)
            # build the data pipeline
            data = test_pipeline(data)
            data_list.append(data)

        data = collate(data_list, samples_per_gpu=len(imgs))

        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0][None, :] for img in data['img']]
        if self.device != 'cpu':
            data = scatter(data, [self.device])[0]

        return data, data['img']

    def visualize(self,
                  model,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  opacity: float = 0.5):
        """Visualize predictions of a model.

        Args:
            model (nn.Module): Input model.
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
            opacity: (float): Opacity of painted segmentation map.
                    Defaults to `0.5`.
        """
        show_img = mmcv.imread(image) if isinstance(image, str) else image
        output_file = None if show_result else output_file
        # Need to wrapper the result with list for mmseg
        result = [result]
        model.show_result(
            show_img,
            result,
            out_file=output_file,
            win_name=window_name,
            show=show_result,
            opacity=opacity)

    @staticmethod
    def run_inference(model, model_inputs: Dict[str, torch.Tensor]):
        """Run inference once for a segmentation model of mmseg.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        return model(**model_inputs, return_loss=False, rescale=True)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        raise NotImplementedError('Not supported yet.')

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any]) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info and image
                tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        return input_data['img'][0]

    @staticmethod
    def evaluate_outputs(model_cfg,
                         outputs: Sequence,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False):
        """Perform post-processing to predictions of model.

        Args:
            outputs (list): A list of predictions of model inference.
            dataset (Dataset): Input dataset to run test.
            model_cfg (mmcv.Config): The model config.
            metrics (str): Evaluation metrics, which depends on
                the codebase and the dataset, e.g., e.g., "mIoU" for generic
                datasets, and "cityscapes" for Cityscapes in mmseg.
            out (str): Output result file in pickle format, defaults to `None`.
            metric_options (dict): Custom options for evaluation, will be
                kwargs for dataset.evaluate() function. Defaults to `None`.
            format_only (bool): Format the output results without perform
                evaluation. It is useful when you want to format the result
                to a specific format and submit it to the test server. Defaults
                to `False`.
        """
        if out:
            logger = get_root_logger()
            logger.info(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            dataset.evaluate(outputs, metrics, **kwargs)

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        load_from_file = self.model_cfg.data.test.pipeline[0]
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.data.test.pipeline
        preprocess[0] = load_from_file
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Nonthing for super resolution.
        """
        postprocess = self.model_cfg.model.decode_head
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'decode_head' in self.model_cfg.model, 'model config contains'
        ' no decode_head'
        name = self.model_cfg.model.decode_head.type[:-4].lower()
        return name
