# Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging

## Abstract
Snapshot compressive imaging (SCI) aims to record
three-dimensional signals via a two-dimensional camera.
For the sake of building a fast and accurate SCI recovery algorithm, we incorporate the interpretability of modelbased methods and the speed of learning-based ones and
present a novel dense deep unfolding network (DUN) with
3D-CNN prior for SCI, where each phase is unrolled from
an iteration of Half-Quadratic Splitting (HQS). To better
exploit the spatial-temporal correlation among frames and
address the problem of information loss between adjacent
phases in existing DUNs, we propose to adopt the 3D-CNN
prior in our proximal mapping module and develop a novel
dense feature map (DFM) strategy, respectively. Besides,
in order to promote network robustness, we further propose a dense feature map adaption (DFMA) module to allow inter-phase information to fuse adaptively. All the parameters are learned in an end-to-end fashion. Extensive
experiments on simulation data and real data verify the superiority of our method. The source code is available at
https://github.com/jianzhangcs/SCI3D.

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----: | :-----:|:----: |:----:|
|PSNR | 35.02| 31.78 | 40.92| 44.49 |  30.58 |  29.35 | 35.36| 
|SSIM |0.9681|0.9641 |0.9825|0.9940 |0.9411  |0.9532  |0.9672|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  1.7320   | 0.6833  |  0.5784 |  0.9441|  0.5518 |

## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](../../docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/DUN-3DUnet/dun3dunet.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES指定显卡编号  
* --nproc_per_node为使用显卡数量  
* --master_port代表主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/DUN-3DUnet/dun3dunet.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/DUN-3DUnet/dun3dunet.py --weights=checkpoints/dun3dunet/dun3dunet.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
## 真实数据集测试
TODO

## Citation 
```
@article{wu2021dense,
  title={Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging},
  author={Wu, Zhuoyuan and Zhang, Jian and Mou, Chong},
  journal={arXiv preprint arXiv:2109.06548},
  year={2021}
}
```