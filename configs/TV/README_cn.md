# Generalized alternating projection based total variation minimization for compressive sensing
## Abstract
We consider the total variation (TV) minimization problem used for compressive sensing and solve it using the generalized alternating projection (GAP) algorithm. Extensive results demonstrate the high performance of proposed algorithm on compressive sensing, including two dimensional images, hyperspectral images and videos. We further derive the Alternating Direction Method of Multipliers (ADMM) framework with TV minimization for video and hyperspectral image compressive sensing under the CACTI and CASSI framework, respectively. Connections between GAP and ADMM are also provided.

## 6个基准灰度仿真数据集上的测试结果
|Dataset|Kobe  |Traffic|Runner | Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----: |:-----:|:----: | :-----:|:----: |:---:|
|PSNR |  26.64 | 20.65  | 30.13 | 34.50  | 25.02| 24.63 | 26.93| 
|SSIM |0.8401 |0.6965 |0.9142 |0.9668 |0.8259 |0.8255 |0.8448|

## 灰度仿真数据集测试
执行以下命令可在6个基准灰度仿真数据集上进行测试。
```
python tools/test_iterative.py configs/TV/tv.py 

```
## 彩色仿真数据集测试
首先在 [BaiduNetdisk](https://pan.baidu.com/s/1wRMBsYoyVFFsEI5-lTPy6w?pwd=d2oi) 的datasets文件夹中下载middle_scale，并将其放置在test_datasets目录下。 

执行以下命令可在6个middle彩色仿真数据集上进行测试。
```
python tools/test_iterative.py configs/TV/tv_mid_color.py 

```

## 真实数据集测试
执行以下命令可在真实数据集上进行测试。
```
python tools/real_data/test_iterative.py configs/TV/tv_real_cr10.py 

```
* 注意： 这里仅仅展示了压缩率为10的真实数据，对于其他压缩率我们只需要重新指定tv_real_cr10.py文件中data_root和cr的属性值（以压缩率cr=20为例）
```
real_data = dict(
    data_root="test_datasets/real_data/cr20",
    cr=20
)
```
## Citation
```
@inproceedings{Yuan2016,
  title = {Generalized alternating projection based total variation minimization for compressive sensing},
  author = {Yuan, Xin},
  booktitle = {2016 IEEE International Conference on Image Processing (ICIP)},
  isbn = {1467399612},
  pages = {2539--2543},
  publisher = {IEEE},
  year = {2016}
}
```
