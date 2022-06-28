# 模型训练数据集
## 使用DAVIS数据集
目前主要支持两种数据预处理方式
### 1. [BIRNAT](../configs/BIRNAT/README_cn.md)论文中描述的数据预处理方式
在训练模型前对DAVIS数据集进行裁剪，旋转等变换，并将变换后的数据进行存储（可点击 [DAVIS_mat]() 下载处理好的数据集，并修改configs/_base_/birnat_davis.py文件中data_root属性值，使其指向训练数据集路径）。

模型训练过程中一般不对数据做任何额外处理。具体实现代码位于 cacti/datasets/birnat_davis.py 文件中。

### 2. 训练过程中进行数据增强的方式
首先在 [DAVIS官网](https://davischallenge.org/) 下载DAVIS 2017数据集，并修改configs/_base_/davis.py文件中data_root属性值，使其指向训练数据集路径。

模型训练前不对数据做任何处理，在训练过程进行裁剪，缩放，翻转等变换。实现代码位于cacti/datasets/pipelines/augmentation.py文件中。

另外，可根据需要在该文件中添加新的数据预处理方式 （以随机缩放为例）
```
@PIPELINES.register_module
class RandomResize:
    def __init__(self,scale=(0.8,1.2)):
        pass
    def __call__(self, imgs):
        pass

```
在configs/\__base__/daivs.py文件中添加新的数据增强方法
```
train_pipeline = [
    ... 
    dict(type='RandomResize',scale=(0.6,1.4))
] #添加或删除数据增强方法
```

## 新增模型训练数据集
### 1.添加新增数据集处理方式 （以davis为例）

新建文件cacti/datasets/davis.py文件
```
from .builder import DATASETS

@DATASETS.register_module 
class DavisData(Dataset):
    def __init__(self,data_root,mask_path):
        pass
    def __getitem__(self,index):
        pass
```
注意：\__getitem__函数一般返回 gt(背景真实值)和 meas(测量值）。

## 2. 导入该模块
在cacti/datasets/\__init__.py文件中添加以下代码：
```
from .davis import  DavisData
```

## 3. 添加配置文件
新建configs/\__base__/daivs.py文件
```
train_pipeline = [
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Resize',resize_h=256,resize_w=256)
] #添加或删除数据增强方法

gene_meas = dict(type='GenerationGrayMeas') 

train_data = dict(
    type="DavisData",
    data_root="", #数据集路径
    mask_path="test_datasets/mask/mask.mat", #mask路径
    pipeline=train_pipeline, #数据增强
    gene_meas = gene_meas, #生成测量值和对应的gt
    mask_shape = None #如果mask_path值为None, 可在此设置需要的mask形状，生成随机mask用于模型训练。
)
```