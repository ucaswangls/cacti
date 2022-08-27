# Spatial-Temporal Transformer for Video Snapshot Compressive Imaging
## Abstract
Video snapshot compressive imaging (SCI)  captures multiple sequential video frames by a single measurement using the idea of computational imaging. The underlying principle is to modulate high-speed frames through different masks and these modulated frames are summed to a single measurement captured by a low-speed 2D sensor (dubbed optical encoder); following this, algorithms are employed to reconstruct the desired high-speed frames (dubbed software decoder) if needed.
In this paper, we consider the reconstruction algorithm in video SCI, ie, recovering a series of video frames from a compressed measurement. Specifically, we propose a Spatial-Temporal transFormer (STFormer) to exploit the correlation in both spatial and temporal domains. STFormer network is composed of a token generation block, a video reconstruction block, and these two blocks are connected by a series of STFormer blocks. 
Each STFormer block consists of a spatial self-attention branch, a temporal self-attention branch and the outputs of these two branches are integrated by a fusion network.
Extensive results on both simulated and real data demonstrate the state-of-the-art performance of STFormer.

## 6个仿真数据集上的测试结果
|Dataset|Kobe |Traffic|Runner| Drop  |Aerial|Vehicle|Average|
|:----:|:----:|:----: |:-----:|:----:|:----:|:----:|:----:|
|PSNR  | 35.51| 32.12 | 42.69 | 44.99|31.49 | 31.17| 36.33| 
|SSIM  |0.9724|0.9674 |0.9882 |0.9944|0.9523|0.9707|0.9742|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  1.6774  | 0.5411  |  0.4500 |  0.9268 |  0.4871|

## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](cacti/docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/STFormer/stformer_base.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES 指定显卡编号  
* --nproc_per_node 表示使用显卡数量  
* --master_port 表示主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/STFormer/stformer_base.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/STFormer/stformer_base.py --weights=checkpoints/stformer_base/stformer_base.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_deeplearning.py configs/STFormer/stformer_base_real_cr10.py --weights=checkpoints/stformer_base/stformer_base_real_cr10.pth

```
注意：
* 权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
* 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们需要首先修改stformer_base_real_cr10.py文件中cr的属性值，并对模型进行重新训练。

## Citation
```

```