# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.TopdownHeatmapMSMUHead.inference_model')
def top_down_heatmap_msmu_head__inference_model(ctx, self, x, flip_pairs=None):
    """Rewrite ``inference_model`` for default backend.

    Rewrite this function to run forward directly.

    Args:
    x (list[torch.Tensor[N,K,H,W]]): Input features.
    flip_pairs (None | list[tuple]):
        Pairs of keypoints which are mirrored.

    Returns:
        output_heatmap (np.ndarray): Output heatmaps.
    """
    assert flip_pairs is None
    output = self.forward(x)
    assert isinstance(output, list)
    output = output[-1]
    return output
