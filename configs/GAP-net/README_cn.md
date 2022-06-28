# GAP-net for snapshot compressive imaging 
## Abstract
Snapshot compressive imaging (SCI) systems aim to capture high-dimensional (≥ 3D) images in a single shot using 2D detectors. SCI devices include two main parts: a hardware encoder and a software decoder. The hardware encoder typically consists of an (optical) imaging system designed to capture compressed measurements. The software decoder on the other hand refers to a reconstruction algorithm that retrieves the desired high-dimensional signal from those measurements. In this paper, using deep unfolding ideas, we propose an SCI recovery algorithm, namely GAP-net, which unfolds the generalized alternating projection (GAP) algorithm. At each stage, GAP-net passes its current estimate of the desired signal through a trained convolutional neural network (CNN). The CNN operates as a denoiser that projects the estimate back to the desired signal space. For the GAP-net that employs trained auto-encoder-based denoisers, we prove a probabilistic global convergence result. Finally, we investigate the performance of GAP-net in solving video SCI and spectral SCI problems. In both cases, GAP-net demonstrates competitive performance on both synthetic and real data. In addition to having high accuracy and high speed, we show that GAP-net is flexible with respect to signal modulation implying that a trained GAP-net decoder can be applied in different systems. Our code is at https://github.com/mengziyi64/ADMM-net.

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR | 31.38| 27.67| 37.59 | 41.11|  28.97| 28.00|  32.47| 
|SSIM |0.9362|0.9200|0.9754|0.9902|0.9165|0.9330|0.9452|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX 8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.0552   | 0.0207  |  0.0206 |  0.0290 |  0.0196|

## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](cacti/docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/GAP-net/gapnet.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES指定显卡编号  
* --nproc_per_node为使用显卡数量  
* --master_port代表主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tootls/train.py configs/GAP-net/gapnet.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tootls/test_deeplearning.py configs/GAP-net/gapnet.py --weights=checkpoints/gapnet/gapnet.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_deeplearning.py configs/GAP-net/gapnet_real_cr10.py --weights=checkpoints/gapnet/gapnet_real_cr10.pth

```
注意：
* 权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
* 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们需要首先修改gapnet_real_cr10.py文件中cr的属性值，并对模型进行重新训练。
## Citation
```
@article{Meng2020gap,
  title = {GAP-net for snapshot compressive imaging},
  author = {Meng, Ziyi and Jalali, Shirin and Yuan, Xin},
  journal = {arXiv preprint arXiv:2012.08364},
  year = {2020}
}
```