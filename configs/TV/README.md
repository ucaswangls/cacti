# Generalized alternating projection based total variation minimization for compressive sensing
## Abstract
We consider the total variation (TV) minimization problem used for compressive sensing and solve it using the generalized alternating projection (GAP) algorithm. Extensive results demonstrate the high performance of proposed algorithm on compressive sensing, including two dimensional images, hyperspectral images and videos. We further derive the Alternating Direction Method of Multipliers (ADMM) framework with TV minimization for video and hyperspectral image compressive sensing under the CACTI and CASSI framework, respectively. Connections between GAP and ADMM are also provided.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner | Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----: |:-----:|:----: | :-----:|:----: |:---:|
|PSNR |  26.64 | 20.65  | 30.13 | 34.50  | 25.02| 24.63 | 26.93| 
|SSIM |0.8401 |0.6965 |0.9142 |0.9668 |0.8259 |0.8255 |0.8448|

## Testing GAP-TV in Simulation Dataset (Grayscale) 
Execute the statement below to launch GAP-TV in 6 benchmark grayscale simulation dataset

```
python tools/test_iterative.py configs/TV/tv.py 

```
## Testing GAP-TV in Simulation Dataset (Color)
First, download datasets/middle_scale folder on [BaiduNetdisk](https://pan.baidu.com/s/1wRMBsYoyVFFsEI5-lTPy6w?pwd=d2oi), and place it in the test_datasets directory.

Then execute the statement below to launch GAP-TV in 6 middle colored simulation dataset

```
python tools/test_iterative.py configs/TV/tv_mid_color.py 

```

## Testing GAP-TV in Real Dataset
Execute the statement below to launch GAP-TV in real dataset.

```
python tools/real_data/test_iterative.py configs/TV/tv_real_cr10.py 

```
* Notice: Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the cr value in file *data_root* and in *tv_real_cr10.py* 

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
