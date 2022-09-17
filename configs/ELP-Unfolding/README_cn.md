# Ensemble Learning Priors Driven Deep Unfolding for Scalable Snapshot Compressive Imaging 
## Abstract
Snapshot compressive imaging (SCI) can record the
3D information by a 2D measurement and from this 2D
measurement to reconstruct the original 3D information by
reconstruction algorithm. As we can see, the reconstruction
algorithm plays a vital role in SCI. Recently, deep learning
algorithm show its outstanding ability, outperforming the
traditional algorithm. Therefore, to improve deep learning
algorithm reconstruction accuracy is an inevitable topic for
SCI. Besides, deep learning algorithms are usually limited
by scalability, and a well trained model in general can
not be applied to new systems if lacking the new training
process. To address these problems, we develop the ensemble learning priors to further improve the reconstruction 
accuracy and propose the scalable learning to empower
deep learning the scalability just like the traditional
algorithm. What’s more, our algorithm has achieved the
state-of-the-art results, outperforming existing algorithms.
Extensive results on both simulation and real datasets
demonstrate the superiority of our proposed algorithm.
The code and models will be released to the public.
https://github.com/integritynoble/ELP-Unfolding/tree/master

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR | 34.41| 31.58 | 41,16 | 44.99 |  30.68 | 29.65 |  35.41 | 
|SSIM |0.9657|0.9623 |0.9857|0.9947 |0.9438 |0.9589 |0.9685|

## 多个平台运行时间分析
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX 8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.9754   | 0.3974  |  0.3364 |  0.6250 |  0.2790|

## 训练
支持高效多GPU训练与单GPU训练, 首先根据 [模型训练数据集](../../docs/add_datasets_cn.md) 配置训练数据集。

多GPU训练可通过以下方式进行启动：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/ELP-Unfolding/.py --distributed=True
```
* 其中CUDA_VISIBLE_DEVICES指定显卡编号  
* --nproc_per_node为使用显卡数量  
* --master_port代表主节点端口号,主要用于通信

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：
```
python tools/train.py configs/ELP-Unfolding/elpunfolding.py
```

## 仿真数据集测试
指定权重参数路径，执行以下命令可在六个基准仿真数据集上进行测试。
```
python tools/test_deeplearning.py configs/ELP-Unfolding/elpunfolding.py --weights=checkpoints/elpunfolding/elpunfolding.pth
```
* --weights 权重参数路径  
注意：权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
<!-- ## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_deeplearning.py configs/ELP-Unfolding/elpunfolding_real_cr10.py --weights=checkpoints/elpunfolding/elpunfolding_real_cr10.pth

```
注意：
* 权重参数路径可以通过 --weight 进行指定，也可以修改配置文件中checkpoints值，相应权重可以在 [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 进行下载。
* 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们需要首先修改elpunfolding_real_cr10.py文件中cr的属性值，并对模型进行重新训练。 -->
## Citation
```
@article{yang2022ensemble,
  title={Ensemble learning priors unfolding for scalable Snapshot Compressive Sensing},
  author={Yang, Chengshuai and Zhang, Shiyu and Yuan, Xin},
  journal={arXiv preprint arXiv:2201.10419},
  year={2022}
}
```