_base_ = ['./segmentation_static.py', '../_base_/backends/ncnn.py']

onnx_config = dict(input_shape=[1024, 512])
