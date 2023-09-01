# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn.functional as F
from mmdet.models.detectors.base import ForwardResults
from mmdet.structures import DetDataSample
from mmdet.structures.det_data_sample import OptSampleList

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


@mark('detr_predict', inputs=['input'], outputs=['dets', 'labels', 'masks'])
def __predict_impl(self, batch_inputs, data_samples, rescale):
    """Rewrite and adding mark for `predict`.

    Encapsulate this function for rewriting `predict` of DetectionTransformer.
    1. Add mark for DetectionTransformer.
    2. Support both dynamic and static export to onnx.
    """
    img_feats = self.extract_feat(batch_inputs)
    head_inputs_dict = self.forward_transformer(img_feats, data_samples)
    results_list = self.bbox_head.predict(
        **head_inputs_dict, rescale=rescale, batch_data_samples=data_samples)
    return results_list


@torch.fx.wrap
def _set_metainfo(data_samples, img_shape):
    """Set the metainfo.

    Code in this function cannot be traced by fx.
    """

    # fx can not trace deepcopy correctly
    data_samples = copy.deepcopy(data_samples)
    if data_samples is None:
        data_samples = [DetDataSample()]

    # note that we can not use `set_metainfo`, deepcopy would crash the
    # onnx trace.
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')

    return data_samples


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.base_detr.DetectionTransformer.forward')
def detection_transformer__forward(self,
                                   batch_inputs: torch.Tensor,
                                   shape_info: torch.Tensor = None,
                                   data_samples: OptSampleList = None,
                                   rescale: bool = True,
                                   **kwargs) -> ForwardResults:
    """Rewrite `predict` for default backend.

    Support configured dynamic/static shape for model input and return
    detection result as Tensor instead of numpy array.

    Args:
        batch_inputs (Tensor): Inputs with shape (N, C, H, W).
        data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        rescale (Boolean): rescale result or not.

    Returns:
        tuple[Tensor]: Detection results of the
        input images.
            - dets (Tensor): Classification bboxes and scores.
                Has a shape (num_instances, 5)
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
    """
    ctx = FUNCTION_REWRITER.get_context()

    deploy_cfg = ctx.cfg

    # get origin input shape as tensor to support onnx dynamic shape
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    img_shape = torch._shape_as_tensor(batch_inputs)[2:].to(
        batch_inputs.device)
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]

    # set the metainfo
    data_samples = _set_metainfo(data_samples, img_shape)
    if shape_info is not None:
        data_samples[0].set_field(
            name='shape_info', value=shape_info, field_type='metainfo')
    return __predict_impl(self, batch_inputs, data_samples, rescale)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.detr.DETR.pre_transformer')
def detection_transformer__pre_transformer(
        self, img_feats, batch_data_samples: OptSampleList = None):

    feat = img_feats[-1]  # NOTE img_feats contains only one feature.
    batch_size, feat_dim, _, _ = feat.shape
    # construct binary masks which for the transformer.
    assert batch_data_samples is not None
    # masks = batch_data_samples[0].masks.to(torch.float32)
    batch_input_shape = batch_data_samples[0].img_shape
    if 'shape_info' in batch_data_samples[0]:
        batch_shape_info = batch_data_samples[0].shape_info
        masks_h = torch.arange(
            batch_input_shape[0],
            device=feat.device).reshape(1, -1,
                                        1).expand(batch_size, -1,
                                                  batch_input_shape[1])
        masks_w = torch.arange(
            batch_input_shape[1],
            device=feat.device).reshape(1, 1,
                                        -1).expand(batch_size,
                                                   batch_input_shape[0], -1)
        masks_h = masks_h >= batch_shape_info[:, 0].view(-1, 1, 1)
        masks_w = masks_w >= batch_shape_info[:, 1].view(-1, 1, 1)
        masks = torch.logical_or(masks_h, masks_w).to(torch.float32)
    else:
        masks = torch.zeros(
            batch_size,
            batch_input_shape[0],
            batch_input_shape[1],
            device=feat.device)

    # NOTE following the official DETR repo, non-zero values represent
    # ignored positions, while zero values mean valid positions.

    masks = F.interpolate(
        masks.unsqueeze(1), size=feat.shape[-2:]).to(torch.bool).squeeze(1)
    # [batch_size, embed_dim, h, w]
    pos_embed = self.positional_encoding(masks)

    # use `view` instead of `flatten` for dynamically exporting to ONNX
    # [bs, c, h, w] -> [bs, h*w, c]
    feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
    pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
    # [bs, h, w] -> [bs, h*w]
    masks = masks.view(batch_size, -1)

    # prepare transformer_inputs_dict
    encoder_inputs_dict = dict(feat=feat, feat_mask=masks, feat_pos=pos_embed)
    decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
    return encoder_inputs_dict, decoder_inputs_dict
