# pytorch 模型到onnx模型和tensorrt模型的转换

## pytorch 模型到 onnx 模型的转换
执行以下命令将 pytorch 模型转换为 onnx 模型 (以Unet为例)
```
python tools/onnx_tensorrt/pytorch2onnx.py configs/Unet/unet.py --work_dirs=None
```
* --work_dirs 转换后的onnx模型保存目录
* --simple_flag 是否开启onnx-simplifer (onnx模型的简化版本)，默认为True。

## onnx 模型到 tensorrt 模型的转换
执行以下命令将 onnx 模型转换为 tensorrt 模型 (以Unet为例)
```
python tools/onnx_tensorrt/onnx2tensorrt.py --onnx_model_name="work_dirs/Unet/unet.onnx"
```
* --onnx_model_name 需要转换onnx模型名

转换后的tensorrt模型默认保存在onnx模型同级目录下
