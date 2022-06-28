# BIRNAT: Bidirectional Recurrent Neural Networks with Adversarial Training for Video Snapshot Compressive Imaging
## Abstract 
We consider the problem of video snapshot compressive imaging (SCI), where multiple high-speed frames are coded by different masks and then summed to a single measurement. This measurement and the modulation masks are fed into our Recurrent Neural Network (RNN) to reconstruct the desired high-speed frames. Our end-to-end sampling and reconstruction system is dubbed BIdirectional Recurrent Neural networks with Adversarial Training (BIRNAT). To our best knowledge, this is the first time that recurrent networks are employed to SCI problem. Our proposed BIRNAT outperforms other deep learning based algorithms and the state-of-the-art optimization based algorithm, DeSCI, through exploiting the underlying correlation of sequential video frames. BIRNAT employs a deep convolutional neural network with Resblock and feature map self-attention to reconstruct the first frame, based on which bidirectional RNN is utilized to reconstruct the following frames in a sequential manner. To improve the quality of the reconstructed video, BIRNAT is further equipped with the adversarial training besides the mean square error loss. Extensive results on both simulation and real data (from two SCI cameras) demonstrate the superior performance of our BIRNAT system. The codes are available at https://github.com/BoChenGroup/BIRNAT.

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
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](cacti/docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/BIRNAT/birnat.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES指定显卡编号  
* --nproc_per_node为使用显卡数量  
* --master_port代表主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tootls/train.py configs/BIRNAT/birnat.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tootls/test_deeplearning.py configs/BIRNAT/birnat.py --weights=checkpoints/birnat/birnat.pth
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
