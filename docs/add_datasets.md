# Model Training Dataset 
## DAVIS Dataset 
Library currently has two preprocessing method

### 1. Preprocessing Method in [BIRNAT](../configs/BIRNAT/README_cn.md)
Crop, rotate, flip and etc., on image dataset before training your model and store the augmented data.

Click [DAVIS_mat]() download augmented dataset, then modify *data_root* value in *configs/_base_/birnat_davis.py* file, make sure *data_root* link to your training dataset path.

Do not do any augmentation while training. Codes can be found in *cacti/datasets/birnat_davis.py* file.

### 2. Data Augmentation when Training 
Download DAVIS 2017 Dataset from [DAVIS website](https://davischallenge.org/), then modify *data_root* value in *configs/_base_/davis.py* file, make sure *data_root* link to your training dataset path.

Do not do any augmentation before training. Crop, scale, flip and etc., while training. Codes can be found in *cacti/datasets/pipelines/augmentation.py* file.

Moreover, users can modify their own preprocessing function in the *augmentation.py* file. (e.g. random scaling)

```
@PIPELINES.register_module
class RandomResize:
    def __init__(self,scale=(0.8,1.2)):
        pass
    def __call__(self, imgs):
        pass

```
Add new preprocessing function in *configs/\__base__/daivs.py*

```
#Add or delete dataaugmentation method
train_pipeline = [
    ... 
    dict(type='RandomResize',scale=(0.6,1.4))
] 
```

Create new cacti/datasets/davis.py file

```
from .builder import DATASETS

@DATASETS.register_module 
class DavisData(Dataset):
    def __init__(self,data_root,mask_path):
        pass
    def __getitem__(self,index):
        pass
```
Notice: function *\__getitem__* returns to *gt* as default.

Add following command in *cacti/datasets/\__init__.py* file

```
from .davis import  DavisData
```

Create configs/\__base__/davis.py file

```
#Add or delete dataaugmentation method
train_pipeline = [
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Resize',resize_h=256,resize_w=256)
]

gene_meas = dict(type='GenerationGrayMeas') 

train_data = dict(
    type="DavisData",
    data_root="", #dataset path 
    mask_path="test_datasets/mask/mask.mat", #mask path 
    pipeline=train_pipeline, #data augmentation 
    gene_meas = gene_meas, #generate measurement and gt 
    mask_shape = None
    #If mask_path=None, set mask shape here
)
```