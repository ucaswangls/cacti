# Plug-and-play algorithms for large-scale snapshot compressive imaging
## Abstract
Snapshot compressive imaging (SCI) aims to capture the high-dimensional (usually 3D) images using a 2D sensor (detector) in a single snapshot. Though enjoying the advantages of low-bandwidth, low-power and low-cost, applying SCI to large-scale problems (HD or UHD videos) in our daily life is still challenging. The bottleneck lies in the reconstruction algorithms; they are either too slow (iterative optimization algorithms) or not flexible to the encoding process (deep learning based end-to-end networks). In this paper, we develop fast and flexible algorithms for SCI based on the plug-and-play (PnP) framework. In addition to the widely used PnP-ADMM method, we further propose the PnP-GAP (generalized alternating projection) algorithm with a lower computational workload and prove the convergence1 of PnP-GAP under the SCI hardware constraints. By employing deep denoising priors, we first time show that PnP can recover a UHD color video (3840×1644×48 with PNSR above 30dB) from a snapshot 2D measurement. Extensive results on both simulation and real datasets verify the superiority of our proposed algorithm. The code is available at https://github.com/liuyang12/PnP-SCI

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |
|PSNR  | 30.39|23.89 |32.66| 39.82| 24.18| 24.57|29.25| 
|SSIM  |0.9241|0.8308|0.9356|0.9861|0.8191|0.8363|0.8887|

首先从[dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 中下载ffdnet文件夹放置于checkpoints文件夹中，然后进行仿真或真实数据重建。
## 灰度仿真数据集测试

执行以下命令可在6个基准灰度仿真数据集上进行测试。
```
python tools/test_iterative.py configs/PnP-FFDNet/ffdnet.py 

```
## 彩色仿真数据集测试
首先在 [BaiduNetdisk](https://pan.baidu.com/s/1wRMBsYoyVFFsEI5-lTPy6w?pwd=d2oi) 的datasets文件夹中下载middle_scale，并将其放置在test_datasets目录下。 

执行以下命令可在6个middle彩色仿真数据集上进行测试 (FFDNet_gray 版本）。
```
python tools/test_color_iterative.py configs/PnP-FFDNet/ffdnet_gray_mid_color.py 

```
执行以下命令可在6个middle彩色仿真数据集上进行测试 (FFDNet_color 版本）。
```
python tools/test_color_iterative.py configs/PnP-FFDNet/ffdnet_color_mid_color.py 

```
## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tootls/real_data/test_iterative.py configs/PnP-FFDNet/ffdnet_real_cr10.py 

```
* 注意： 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们只需要重新指定ffdnet_real_cr10.py文件中data_root和cr的属性值（以压缩率cr=20为例）
```

## Citation
```
@inproceedings{Yuan2020plug,  
  title = {{Plug-and-play algorithms for large-scale snapshot compressive imaging}},
  author = {Yuan, Xin and Liu, Yang and Suo, Jinli and Dai, Qionghai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {1447--1457},
  year = {2020}
}
```
