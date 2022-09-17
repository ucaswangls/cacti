# Memory-efficient network for large-scale video compressive sensing
## Abstract
 Video snapshot compressive imaging (SCI) captures a sequence of video frames in a single shot using a 2D detector. The underlying principle is that during one exposure time, different masks are imposed on the high-speed scene to form a compressed measurement. With the knowledge of jmasks, optimization algorithms or deep learning methods are employed to reconstruct the desired high-speed video frames from this snapshot measurement. Unfortunately, though these methods can achieve decent results, the long running time of optimization algorithms or huge training memory occupation of deep networks still preclude them in practical applications. In this paper, we develop a memory-efficient network for large-scale video SCI based on multi-group reversible 3D convolutional neural networks. In addition to the basic model for the grayscale SCI system, we take one step further to combine demosaicing and SCI reconstruction to directly recover color video from Bayer measurements. Extensive results on both simulation and real data captured by SCI cameras demonstrate that our proposed model outperforms previous state-of-the-art with less memory and thus can be used in large-scale problems. The code is at https://github.com/BoChenGroup/RevSCI-net.

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR| 33.72| 30.02 | 39.40|  42.93|  29.35 | 28.13|  33.92| 
|SSIM|0.9572|0.9498|0.9775|0.9924|0.9245|0.9363|0.9563|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.3011  | 0.2851  |  0.2002 |  0.2252 |  0.2124|

## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](../../docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/RevSCI/revsci.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES 指定显卡编号  
* --nproc_per_node 表示使用显卡数量  
* --master_port 表示主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/RevSCI/revsci.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/RevSCI/revsci.py --weights=checkpoints/revsci/revsci.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_deeplearning.py configs/RevSCI/revsci_real_cr10.py --weights=checkpoints/revsci/revsci_real_cr10.pth

```
注意：
* 权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
* 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们需要首先修改revsci_real_cr10.py文件中cr的属性值，并对模型进行重新训练。

## Citation
```
@inproceedings{Cheng2021d,
  title = {Memory-efficient network for large-scale video compressive sensing}
  author = {Cheng, Ziheng and Chen, Bo and Liu, Guanliang and Zhang, Hao and Lu, Ruiying and Wang, Zhengjue and Yuan, Xin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {16246--16255},
  year = {2021}
}
```