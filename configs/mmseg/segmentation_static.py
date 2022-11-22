_base_ = ['../_base_/onnx_config.py']
codebase_config = dict(type='mmseg', task='Segmentation')
onnx_config = dict(dynamic_axes={
    'input': {
        0: 'batch',
    },
    'output': {
        0: 'batch',
    },
})
