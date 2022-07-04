# pytorch to onnx and onnx to tensorrt

## pytorch to onnx
Transfer pytorch to onnx by executing following command

```
python tools/onnx_tensorrt/pytorch2onnx.py configs/Unet/unet.py --work_dirs=None
```
* --work_dirs transferrd onnx saved path
* --simple_flag onnx-simplifer switch, True as default

## onnx to tensorrt
Transfer onnx to tensorrt by executing following command

```
python tools/onnx_tensorrt/onnx2tensorrt.py --onnx_model_name="work_dirs/Unet/unet.onnx"
```
* --onnx_model_name need change onnx model name

Transferred tensorrt will be saved in *onnx_tensorrt* folder
