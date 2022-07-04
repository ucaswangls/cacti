# Model Params and FLOPs Analysis 
Install thop by executing following command

```
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
```
Calculate model Params and FLOATs by executing following command

```
python tools/params_floats.py configs/Unet/unet.py --work_dirs=None
```
*--work_dirs specify log saving path, model params and FLOATs will be stored in log file.

```
Model Info:
Unet(
    ...
)

Params: 0.82 M 
FLOATs: 53.63 G
```