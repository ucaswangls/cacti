# Spectral Compressive Imaging Reconstruction Using Convolution and Contextual Transformer 
## Abstract
Spectral compressive imaging (SCI) is able to encode the high-dimensional hyperspectral image
into a two-dimensional (2D) snapshot measurement, and then use algorithms to reconstruct
the spatio-spectral data-cube. At present, the main bottleneck of SCI is the reconstruction
algorithm, and state-of-the-art (SOTA) reconstruction methods generally face the problems of
long reconstruction time and/or poor detail recovery. In this paper, we propose a novel hybrid
network module, namely CCoT (Convolution and Contextual Transformer) block, which can
simultaneously acquire the inductive bias ability of convolution and the powerful modeling
ability of Transformer, which is conducive to improving the quality of reconstruction to restore
fine details. We integrate the proposed CCoT block into a physics-driven deep unfolding
framework based on the generalized alternating projection algorithm, and further propose the
GAP-CCoT network. Finally, we apply the GAP-CCoT algorithm to SCI reconstruction. Through
experiments on a large amount of synthetic data and real data, our proposed model achieves
higher reconstruction quality (>2dB in PSNR on simulated benchmark datasets) and shorter
running time than existing SOTA algorithms by a large margin. The code and models are publicly
available at https://github.com/ucaswangls/GAP-CCoT.

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR | 32.58| 29.03| 39.12 | 42.54 |  29.40| 28.52|  33.53| 
|SSIM |0.9494|0.9378|0.9795|0.9922|0.9229|0.9411|0.9538|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX 8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.1079   | 0.0701  |  0.0650 |  0.0742 |  0.0642|

## 训练
根据cuda版本安装对应的cupy，具体请参考[cupy](https://cupy.dev/)官网。

支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](cacti/docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/GAP-CCoT/gapccot.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES指定显卡编号  
* --nproc_per_node为使用显卡数量  
* --master_port代表主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/GAP-CCoT/gapccot.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/GAP-CCoT/gapccot.py --weights=checkpoints/gapccot/gapccot.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。

## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_deeplearning.py configs/GAP-CCoT/gapccot_real_cr10.py --weights=checkpoints/gapccot/gapccot_real_cr10.pth

```
注意：
* 权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
* 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们需要首先修改gapccot_real_cr10.py文件中cr的属性值，并对模型进行重新训练。
## Citation
```
@article{wang2022snapshot,
  title={Snapshot spectral compressive imaging reconstruction using convolution and contextual Transformer},
  author={Wang, Lishun and Wu, Zongliang and Zhong, Yong and Yuan, Xin},
  journal={Photonics Research},
  volume={10},
  number={8},
  pages={1848--1858},
  year={2022},
  publisher={Optica Publishing Group}
}
```