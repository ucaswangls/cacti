# Memory-efficient network for large-scale video compressive sensing
## Abstract
To capture high-speed videos using a two-dimensional detector, video snapshot compressive imaging (SCI) is a
promising system, where the video frames are coded by different masks and then compressed to a snapshot measurement. Following this, efficient algorithms are desired to reconstruct the high-speed frames,
where the state-of-the-art results are achieved by deep learning networks. However,
these networks are usually trained for specific small-scale masks and often have high demands of training time and  GPU memory, which are hence not flexible to i) a new mask  with the same size and ii) a larger-scale mask. We address  these challenges by developing a Meta Modulated Convolutional Network for SCI reconstruction, dubbed MetaSCI.
MetaSCI is composed of a shared backbone for different masks, and light-weight meta-modulation parameters to evolve to different modulation parameters for each mask, thus having the properties of fast adaptation to new masks (or systems) and ready to scale to large data. Extensive simulation and real data results demonstrate the superior performance of our proposed approach. . Our code is available at https://github.com/xyvirtualgroup/MetaSCI-CVPR2021

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----: | :-----:|:----: |:----:|
|PSNR  | 29.90 | 25.94| 36.37 |  39.60| 28.58  | 27.71 | 31.35| 
|SSIM  |0.9111 |0.8841|0.9703 |0.9871 |0.9067  |0.9280 |0.9312|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.0160  | 0.0067  |  0.0056 |  0.0089 |  0.0052|

## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](cacti/docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/MetaSCI/metasci.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES 指定显卡编号  
* --nproc_per_node 表示使用显卡数量  
* --master_port 表示主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/MetaSCI/metasci.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/MetaSCI/metasci.py --weights=checkpoints/metasci/metasci.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_deeplearning.py configs/MetaSCI/metasci_real_cr10.py --weights=checkpoints/metasci/metasci_real_cr10.pth

```
注意：
* 权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
* 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们需要首先修改gapnet_metasci_cr10.py文件中cr的属性值，并对模型进行重新训练。

## Citation
```
@inproceedings{Wang2021e,
  author = {Wang, Zhengjue and Zhang, Hao and Cheng, Ziheng and Chen, Bo and Yuan, Xin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {2083--2092},
  title = {{MetaSCI: Scalable and adaptive reconstruction for video compressive sensing}},
  year = {2021}
}
```