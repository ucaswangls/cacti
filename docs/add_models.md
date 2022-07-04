# Add new Model

## 1. Add model（Take the Unet network as an example）： 

Create *cacti/models/unet.py* file

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
Notice: forward model inputs usually means y(measurement), Phi(mask), Phi_s(mask_s)

## 2. Import Module
Add following command in *cacti/models/\__init__.py* file

```
from .unet import Unet
```
## 3. Add configuration file 
Create *configs/Unet/unet.py* file

```
_base_=[
        "../_base_/six_gray_sim_data.py", #Test data configuration file 
        "../_base_/davis.py", #Training data configuration file
        "../_base_/default_runtime.py" #Traing configuration file 
        ]

#dataloader setting dataloader
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

#Model parameter setting
model = dict(
    type='Unet',
    in_ch=8,
    out_ch=64
)

#Validate setting while training 
eval=dict(
    flag=True,
    interval=1
)
```