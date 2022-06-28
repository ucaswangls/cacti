# 新增自定义模型

## 1. 添加自定义模型的代码（以Unet网络为例）： 

新建cacti/models/unet.py文件
```
from torch import nn 
from .builder import MODELS

@MODELS.register_module
class Unet(nn.Module):
    def __init__(self,args):
        pass
    def forward(self,y,Phi,Phi_s):
        pass
```
注意：模型forward 参数一般设置为 measurement(y), mask(Phi), mask_s(Phi_s)
## 2. 导入该模块
在cacti/models/\__init__.py文件中添加以下代码：
```
from .unet import Unet
```
## 3. 添加配置文件
新建configs/Unet/unet.py文件
```
_base_=[
        "../_base_/six_gray_sim_data.py", #测试数据配置文件
        "../_base_/davis.py", #训练数据配置文件
        "../_base_/default_runtime.py" #训练配置文件
        ]

# dataloader 配置
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

#模型参数配置
model = dict(
    type='Unet',
    in_ch=8,
    out_ch=64
)

#训练过程中验证相关配置
eval=dict(
    flag=True,
    interval=1
)
```