# 模型Params和FLOATs分析
首先执行以下命令安装 thop 
```
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
```
执行以下命令计算模型Params和FLOATs (以Unet为例)
```
python tools/params_floats.py configs/Unet/unet.py --work_dirs=None
```
*--work_dirs 指定分析日志保存目录,日志主要内容为模型配置和模型Params和FLOATs具体值

```
Model Info: #模型信息
Unet(
    ...
)

Params: 0.82 M  #模型Params
FLOATs: 53.63 G #模型FLOATs
```