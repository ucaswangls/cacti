# Model Params and FLOPs Analysis 
Install thop by executing following command

```
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
```
Calculate model Params and FLOPs by executing following command

```
python tools/params_flops.py configs/Unet/unet.py --work_dirs=None
```
*--work_dirs specify log saving path, model params and FLOPs will be stored in log file.

```
Model Info:
Unet(
    ...
)

Params: 0.82 M 
FLOPs: 53.63 G
```