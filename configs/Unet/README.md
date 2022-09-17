# Deep learning for video compressive sensing
## Abstract
We investigate deep learning for video compressive sensing within the scope of snapshot compressive imaging (SCI). In video SCI, multiple high-speed frames are modulated by different coding patterns and then a low-speed detector captures the integration of these modulated frames. In this manner, each captured measurement frame incorporates the information of all the coded frames, and reconstruction algorithms are then employed to recover the high-speed video. In this paper, we build a video SCI system using a digital micromirror device and develop both an end-to-end convolutional neural network (E2E-CNN) and a Plug-and-Play (PnP) framework with deep denoising priors to solve the inverse problem. We compare them with the iterative baseline algorithm GAP-TV and the state-of-the-art DeSCI on real data. Given a determined setup, a well-trained E2E-CNN can provide video-rate high-quality reconstruction. The PnP deep denoising method can generate decent results without task-specific pre-training and is faster than conventional iterative algorithms. Considering speed, accuracy, and flexibility, the PnP deep denoising method may serve as a baseline in video SCI reconstruction. To conduct quantitative analysis on these reconstruction algorithms, we further perform a simulation comparison on synthetic data. We hope that this study contributes to the applications of SCI cameras in our daily life

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |
|PSNR |  29.14| 24.94| 34.93|  38.13|  27.94|  26.97|  30.34 | 
|SSIM |0.8934|0.8489|0.9567|0.9615| 0.8856|0.8943|0.9067|

## Multi Platform Running Time Analysis
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.0271   | 0.0081  |  0.0067 |   0.0132|  0.0077|

## Training U-Net 
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](../../docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/Unet/unet.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/Unet/unet.py
```

## Testing U-Net on Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/Unet/unet.py --weights=checkpoints/unet/unet.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via  [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).

## Testing U-Net on Real Dataset
Launch U-Net on real dataset by executing the statement below.

```
python tools/real_data/test_deeplearning.py configs/Unet/unet_real_cr10.py --weights=checkpoints/unet/unet_real_cr10.pth

```
Notice:

* Path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).
* The results only show the real data under the compress ratio equals to 10, for other compress ratio, we need to modify the compress ratio value in the unet_real_cr10.py file, and then start a new model training process.

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