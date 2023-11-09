# Plug-and-Play Algorithms for Video Snapshot Compressive Imaging
## Abstract
We consider the reconstruction problem of video snapshot compressive imaging (SCI), which captures high-speed videos using a low-speed 2D sensor (detector). The underlying principle of SCI is to modulate sequential high-speed frames with different masks and then these encoded frames are integrated into a snapshot on the sensor and thus the sensor can be of low-speed. On one hand, video SCI enjoys the advantages of low-bandwidth, low-power and low-cost. On the other hand, applying SCI to large-scale problems (HD or UHD videos) in our daily life is still challenging and one of the bottlenecks lies in the reconstruction algorithm. Exiting algorithms are either too slow (iterative optimization algorithms) or not flexible to the encoding process (deep learning based end-to-end networks). In this paper, we develop fast and flexible algorithms for SCI based on the plug-and-play (PnP) framework. In addition to the PnP-ADMM method, we further propose the PnP-GAP (generalized alternating projection) algorithm with a lower computational workload. We first employ the image deep denoising priors to show that PnP can recover a UHD color video with 30 frames from a snapshot measurement. Since videos have strong temporal correlation, by employing the video deep denoising priors, we achieve a significant improvement in the results. Furthermore, we extend the proposed PnP algorithms to the color SCI system using mosaic sensors, where each pixel only captures the red, green or blue channels. A joint reconstruction and demosaicing paradigm is developed for flexible and high quality reconstruction of color video SCI systems. Extensive results on both simulation and real datasets verify the superiority of our proposed algorithm.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:---:|
|PSNR | 32.33| 26.17 | 36.14|  41.93|  27.87 |  26.33 | 31.79 | 
|SSIM | 0.9431|0.9174|0.9616|0.9892 |0.8945  |0.9154 |0.9369| 

First download fastdvd folder and place it in to checkpoints folder from [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0) , then do the simulation or reconstruction.


## Multi Platform Running Time Analysis 
Execute the statement below to launch PnP-FastDVDnet in 6 benchmark grayscale simulation dataset

```
python tools/test_iterative.py configs/FastDVD/fastdvd.py 

```
## Testing PnP-FastDVDnet in Colored Simulation Dataset 
First, download datasets/middle_scale folder on [Dropbox](https://www.dropbox.com/sh/3cj7nv5l0hfqup9/AAAMbLQXmoVki98cqwuv754ia?dl=0), and place it in the test_datasets directory.

Then execute the statement below to launch PnP-FastDVDnet in 6 middle colored simulation dataset (run FastDVDnet_gray)

```
python tools/test_color_iterative.py configs/PnP-FastDVD/fastdvd_gray_mid_color.py 

```
Execute the statement below to launch PnP-FastDVDnet in 6 middle colored simulation dataset

```
python tools/test_color_iterative.py configs/PnP-FastDVD/fastdvd_color_mid_color.py 

```
## Testing PnP-FastDVDnet on Real Dataset
Launch PnP-FastDVDnet on real dataset by executing the statement below.

```
python tools/real_data/test_iterative.py configs/PnP-FastDVD/fastdvd_real_cr10.py 

```
* Notice: Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the cr value in file *data_root* and in *fastdvd_real_cr10.py* 

```
real_data = dict(
    data_root="test_datasets/real_data/cr20",
    cr=20
)
```
## Citation 
```
@article{yuan2021plug,
  title={Plug-and-Play Algorithms for Video Snapshot Compressive Imaging.},
  author={Yuan, X and Liu, Y and Suo, J and Durand, F and Dai, Q},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
