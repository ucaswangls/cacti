# BIRNAT: Bidirectional Recurrent Neural Networks with Adversarial Training for Video Snapshot Compressive Imaging (用于视频单曝光压缩成像的配合对抗训练的双向循环神经网络)
## Abstract 
本文研究的是视频单曝光压缩(Video Snapshot Compressive Imaging, Video SCI)成像中遇到的问题，Video SCI通过将多个高速帧经由不同掩模调制后被联合曝光在单一测量帧中。测量帧和调制用掩模被输入到文中的循环神经网络以获得高速帧的重建结果。本文的端到端采样和图像重建系统被称为配合对抗训练的双向循环神经网络(Bidirectional Recurrent Neural networks with Adversarial Training, BIRNAT)。据我们所知，这是首次将循环神经网络引入视频SCI。本文提出的BIRNAT通过利用视频帧间序列的潜在相关性，得到了表现优于当下最先进的基于深度学习和最优化的DeSCI算法结果。BIRNAT使用带有残差模块和特征图自注意力机制的深度卷积神经网络来重建第一帧，随后使用双向循环神经网络以顺序方式重建后续帧图像。为了进一步提升重建视频质量，除了使用常规的均方误差损失函数外，本文还为BIRNAT进行了对抗训练。大量的实验结果证明了本文BIRNAT系统的对于模拟和真实数据集的优异性能。代码于https://github.com/BoChenGroup/BIRNAT处提供。

## 6个仿真数据集上的测试结果
|Dataset|Kobe |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----:|:----: |:----:|:-----:|:----: | :-----:|:----: |
|PSNR  | 32.71| 29.33 | 38.70|  42.28|  28.99|  27.84 | 33.31 | 
|SSIM  |0.9505|0.9434 |0.9766| 0.9919| 0.9176|  0.9270|0.9512 |
## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.2311   | 0.1155  |  0.0988 |  0.1684 |  0.1179|

## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](../../docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/BIRNAT/birnat.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES指定显卡编号  
* --nproc_per_node为使用显卡数量  
* --master_port代表主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/BIRNAT/birnat.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/BIRNAT/birnat.py --weights=checkpoints/birnat/birnat.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
## 真实数据集测试
TODO
## Citation
```
@inproceedings{Cheng2020birnat,  
  title = {BIRNAT: Bidirectional recurrent neural networks with adversarial training for video snapshot compressive imaging},  
  author = {Cheng, Ziheng and Lu, Ruiying and Wang, Zhengjue and Zhang, Hao and Chen, Bo and Meng, Ziyi and Yuan, Xin},  
  booktitle = {European Conference on Computer Vision},  
  pages = {258--275},  
  publisher = {Springer},  
  year = {2020}
}
```
