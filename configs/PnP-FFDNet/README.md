# Plug-and-play algorithms for large-scale snapshot compressive imaging
## Abstract
Snapshot compressive imaging (SCI) aims to capture the high-dimensional (usually 3D) images using a 2D sensor (detector) in a single snapshot. Though enjoying the advantages of low-bandwidth, low-power and low-cost, applying SCI to large-scale problems (HD or UHD videos) in our daily life is still challenging. The bottleneck lies in the reconstruction algorithms; they are either too slow (iterative optimization algorithms) or not flexible to the encoding process (deep learning based end-to-end networks). In this paper, we develop fast and flexible algorithms for SCI based on the plug-and-play (PnP) framework. In addition to the widely used PnP-ADMM method, we further propose the PnP-GAP (generalized alternating projection) algorithm with a lower computational workload and prove the convergence1 of PnP-GAP under the SCI hardware constraints. By employing deep denoising priors, we first time show that PnP can recover a UHD color video (3840×1644×48 with PNSR above 30dB) from a snapshot 2D measurement. Extensive results on both simulation and real datasets verify the superiority of our proposed algorithm. The code is available at https://github.com/liuyang12/PnP-SCI

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |
|PSNR  | 30.39|23.89 |32.66| 39.82| 24.18| 24.57|29.25| 
|SSIM  |0.9241|0.8308|0.9356|0.9861|0.8191|0.8363|0.8887|

First download fffdnet folder and place it in to checkpoints folder from [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0), then do the simulation or reconstruction.


## Testing PnP-FFDNet in Grayscale Simulation Dataset 

Execute the statement below to launch PnP-FFDNet in 6 benchmark grayscale simulation dataset

```
python tools/test_iterative.py configs/FFDNet/ffdnet.py 

```
## Testing PnP-FFDNet in Colored Simulation Dataset 
First, download datasets/middle_scale folder on [Dropbox](https://www.dropbox.com/sh/3cj7nv5l0hfqup9/AAAMbLQXmoVki98cqwuv754ia?dl=0), and place it in the test_datasets directory.

Then execute the statement below to launch PnP-FFDNet in 6 middle colored simulation dataset (run FFDNet_gray)

```
python tools/test_iterative.py configs/FFDNet/ffdnet_gray_mid_color.py 

```
Execute the statement below to launch PnP-FFDNet in 6 middle colored simulation dataset

```
python tools/test_iterative.py configs/FFDNet/ffdnet_color_mid_color.py 

```
## Testing PnP-FFDNet on Real Dataset 
Launch PnP-FFDNet on real dataset by executing the statement below.

```
python tootls/real_data/test_iterative.py configs/FFDNet/ffdnet_real_cr10.py 

```
* Notice: Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the cr value in file *data_root* and in *ffdnet_real_cr10.py* 

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