# Copyright (c) OpenMMLab. All rights reserved.

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmpose.deploy.mmpose import MMPOSE_TASK
from mmdeploy.utils import Task


@MMPOSE_TASK.register_module(Task.POSE_DETECTION.value)
class PoseDetection(BaseTask):
    """Pose detection task class.

    Args:
        model_cfg (mmcv.Config): Original PyTorch model config file.
        deploy_cfg (mmcv.Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Sequence[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files. Default is None.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .pose_detection_model import build_pose_detection_model
        model = build_pose_detection_model(
            model_files, self.model_cfg, self.deploy_cfg, device=self.device)
        return model.eval()

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                defaults to `None`.

        Returns:
            nn.Module: An initialized torch model generated by other OpenMMLab
                codebases.
        """
        from mmpose.apis import init_pose_model
        from mmcv.cnn.utils import revert_sync_batchnorm
        model = init_pose_model(self.model_cfg, model_checkpoint, self.device)
        model = revert_sync_batchnorm(model)
        model.eval()
        return model

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Sequence[int] = None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Create input for pose detection.

        Args:
            imgs (Any): Input image(s), accepted data type are ``str``,
                ``np.ndarray``.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to ``None``.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmpose.datasets.pipelines import Compose
        from mmpose.datasets.dataset_info import DatasetInfo
        from mmpose.apis.inference import LoadImage, _box2cs

        cfg = self.model_cfg

        dataset_info = cfg.data.test.dataset_info
        dataset_info = DatasetInfo(dataset_info)

        # create dummy person results
        if isinstance(imgs, str):
            height, width = mmcv.imread(imgs).shape[:2]
        else:
            height, width = imgs.shape[:2]
        person_results = [{'bbox': np.array([0, 0, width, height])}]
        bboxes = np.array([box['bbox'] for box in person_results])

        # build the data pipeline
        channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
        test_pipeline = [LoadImage(channel_order=channel_order)
                         ] + cfg.test_pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs
        batch_data = []
        if self.deploy_cfg.onnx_config['input_shape'] is not None:
            image_size = self.deploy_cfg.onnx_config['input_shape']
        else:
            image_size = np.array(cfg.data_cfg['image_size'])
        for bbox in bboxes:
            center, scale = _box2cs(cfg, bbox)

            # prepare data
            data = {
                'img_or_path':
                imgs,
                'center':
                center,
                'scale':
                scale,
                'bbox_score':
                bbox[4] if len(bbox) == 5 else 1,
                'bbox_id':
                0,  # need to be assigned if batch_size > 1
                'dataset':
                dataset_name,
                'joints_3d':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'joints_3d_visible':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'rotation':
                0,
                'ann_info': {
                    'image_size': image_size,
                    'num_joints': cfg.data_cfg['num_joints'],
                    'flip_pairs': flip_pairs
                }
            }
            data = test_pipeline(data)
            batch_data.append(data)

        batch_data = collate(batch_data, samples_per_gpu=1)
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(torch.device(self.device))
        # get all img_metas of each bounding box
        batch_data['img_metas'] = [
            img_metas[0] for img_metas in batch_data['img_metas'].data
        ]
        return batch_data, batch_data['img']

    def visualize(self,
                  model: torch.nn.Module,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str,
                  show_result: bool = False,
                  **kwargs):
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
        """
        from mmpose.datasets.dataset_info import DatasetInfo
        dataset_info = self.model_cfg.data.test.dataset_info
        dataset_info = DatasetInfo(dataset_info)
        skeleton = dataset_info.skeleton
        pose_kpt_color = dataset_info.pose_kpt_color
        pose_link_color = dataset_info.pose_link_color
        if hasattr(model, 'module'):
            model = model.module
        if isinstance(image, str):
            image = mmcv.imread(image)
        # convert result
        result = [dict(keypoints=pose) for pose in result['preds']]
        model.show_result(
            image,
            result,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            out_file=output_file,
            show=show_result,
            win_name=window_name)

    @staticmethod
    def evaluate_outputs(model_cfg: mmcv.Config,
                         outputs: Sequence,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         **kwargs):
        """Perform post-processing to predictions of model.

        Args:
            model_cfg (mmcv.Config): The model config.
            outputs (list): A list of predictions of model inference.
            dataset (Dataset): Input dataset to run test.
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
        res_folder = '.'
        if out:
            logging.info(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
            res_folder, _ = os.path.split(out)
        os.makedirs(res_folder, exist_ok=True)

        eval_config = model_cfg.get('evaluation', {}).copy()
        if metrics is not None:
            eval_config.update(dict(metric=metrics))

        results = dataset.evaluate(outputs, res_folder, **eval_config)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name

    @staticmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config for mmpose.

        Args:
            partition_type (str): A string specifying partition type.
        """
        raise NotImplementedError('Not supported yet.')

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK."""
        raise NotImplementedError('Not supported yet.')

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK."""
        raise NotImplementedError('Not supported yet.')

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any],
                              **kwargs) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info and image
                tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        img = input_data['img']
        if isinstance(img, (list, tuple)):
            img = img[0]
        return img

    @staticmethod
    def run_inference(model, model_inputs: Dict[str, torch.Tensor]):
        """Run inference once for a pose model of mmpose.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        output = model(
            **model_inputs,
            return_loss=False,
            return_heatmap=False,
            target=None,
            target_weight=None)
        return [output]
