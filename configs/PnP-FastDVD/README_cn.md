# Plug-and-Play Algorithms for Video Snapshot Compressive Imaging
## Abstract
We consider the reconstruction problem of video snapshot compressive imaging (SCI), which captures high-speed videos using a low-speed 2D sensor (detector). The underlying principle of SCI is to modulate sequential high-speed frames with different masks and then these encoded frames are integrated into a snapshot on the sensor and thus the sensor can be of low-speed. On one hand, video SCI enjoys the advantages of low-bandwidth, low-power and low-cost. On the other hand, applying SCI to large-scale problems (HD or UHD videos) in our daily life is still challenging and one of the bottlenecks lies in the reconstruction algorithm. Exiting algorithms are either too slow (iterative optimization algorithms) or not flexible to the encoding process (deep learning based end-to-end networks). In this paper, we develop fast and flexible algorithms for SCI based on the plug-and-play (PnP) framework. In addition to the PnP-ADMM method, we further propose the PnP-GAP (generalized alternating projection) algorithm with a lower computational workload. We first employ the image deep denoising priors to show that PnP can recover a UHD color video with 30 frames from a snapshot measurement. Since videos have strong temporal correlation, by employing the video deep denoising priors, we achieve a significant improvement in the results. Furthermore, we extend the proposed PnP algorithms to the color SCI system using mosaic sensors, where each pixel only captures the red, green or blue channels. A joint reconstruction and demosaicing paradigm is developed for flexible and high quality reconstruction of color video SCI systems. Extensive results on both simulation and real datasets verify the superiority of our proposed algorithm.

## 6个仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:---:|
|PSNR | 32.33| 26.17 | 36.14|  41.93|  27.87 |  26.33 | 31.79 | 
|SSIM | 0.9431|0.9174|0.9616|0.9892 |0.8945  |0.9154 |0.9369| 

首先从[dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) 中下载fastdvd文件夹放置于checkpoints文件夹中，然后进行仿真或真实数据重建。

## 灰度仿真数据集测试
执行以下命令可在6个基准灰度仿真数据集上进行测试。
```
python tools/test_iterative.py configs/FastDVD/fastdvd.py 

```
## 彩色仿真数据集测试
首先在 [Dropbox](https://www.dropbox.com/sh/3cj7nv5l0hfqup9/AAAMbLQXmoVki98cqwuv754ia?dl=0) 的datasets文件夹中下载middle_scale，并将其放置在test_datasets目录下。 

执行以下命令可在6个middle彩色仿真数据集上进行测试 (FastDVDnet_gray 版本）。
```
python tools/test_iterative.py configs/FastDVD/fastdvd_gray_mid_color.py 

```
执行以下命令可在6个middle彩色仿真数据集上进行测试 (FastDVDnet_color 版本）。
```
python tools/test_iterative.py configs/FastDVD/fastdvd_color_mid_color.py 

```
## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_iterative.py configs/FastDVD/fastdvd_real_cr10.py 

```
* 注意： 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们只需要重新指定fastdvd_real_cr10.py文件中data_root和cr的属性值（以压缩率cr=20为例）
```
real_data = dict(
    data_root="test_datasets/real_data/cr20",
    cr=20
)
```
## Citation
```
@article{2021Plug,
  title={Plug-and-Play Algorithms for Video Snapshot Compressive Imaging},
  author={ Yuan, X.  and  Liu, Y.  and  Suo, J.  and  Durand, F.  and  Dai, Q. },
  year={2021},
}
```