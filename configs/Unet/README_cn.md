# Deep learning for video compressive sensing
## Abstract
We investigate deep learning for video compressive sensing within the scope of snapshot compressive imaging (SCI). In video SCI, multiple high-speed frames are modulated by different coding patterns and then a low-speed detector captures the integration of these modulated frames. In this manner, each captured measurement frame incorporates the information of all the coded frames, and reconstruction algorithms are then employed to recover the high-speed video. In this paper, we build a video SCI system using a digital micromirror device and develop both an end-to-end convolutional neural network (E2E-CNN) and a Plug-and-Play (PnP) framework with deep denoising priors to solve the inverse problem. We compare them with the iterative baseline algorithm GAP-TV and the state-of-the-art DeSCI on real data. Given a determined setup, a well-trained E2E-CNN can provide video-rate high-quality reconstruction. The PnP deep denoising method can generate decent results without task-specific pre-training and is faster than conventional iterative algorithms. Considering speed, accuracy, and flexibility, the PnP deep denoising method may serve as a baseline in video SCI reconstruction. To conduct quantitative analysis on these reconstruction algorithms, we further perform a simulation comparison on synthetic data. We hope that this study contributes to the applications of SCI cameras in our daily life

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |
|PSNR |  29.14| 24.94| 34.93|  38.13|  27.94|  26.97|  30.34 | 
|SSIM |0.8934|0.8489|0.9567|0.9615| 0.8856|0.8943|0.9067|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.0271   | 0.0081  |  0.0067 |   0.0132|  0.0077|


## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](../../docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/Unet/unet.py --distributed=True
```
* 其中 CUDA_VISIBLE_DEVICES 指定显卡编号  
* --nproc_per_node 表示使用显卡数量  
* --master_port 表示主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/Unet/unet.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/Unet/unet.py --weights=checkpoints/unet/unet.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_deeplearning.py configs/Unet/unet_real_cr10.py --weights=checkpoints/unet/unet_real_cr10.pth

```
注意：
* 权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
* 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们需要首先修改unet_real_cr10.py文件中cr的属性值，并对模型进行重新训练。

## Citation
```
@article{Qiao2020,
  title = {Deep learning for video compressive sensing},
  author = {Qiao, Mu and Meng, Ziyi and Ma, Jiawei and Yuan, Xin},
  issn = {2378-0967},
  journal = {APL Photonics},
  number = {3},
  pages = {30801},
  publisher = {AIP Publishing LLC},
  volume = {5},
  year = {2020}
}
```