# 模型Params和FLOPs分析
首先执行以下命令安装 thop 
```
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
```
执行以下命令计算模型Params和FLOPs (以Unet为例)
```
python tools/params_flops.py configs/Unet/unet.py --work_dirs=None
```
*--work_dirs 指定分析日志保存目录,日志主要内容为模型配置和模型Params和FLOPs具体值

```
Model Info: #模型信息
Unet(
    ...
)

Params: 0.82 M  #模型Params
FLOPs: 53.63 G #模型FLOPs
```