# GAP-net for snapshot compressive imaging  
## Abstract
Snapshot compressive imaging (SCI) systems aim to capture high-dimensional (≥ 3D) images in a single shot using 2D detectors. SCI devices include two main parts: a hardware encoder and a software decoder. The hardware encoder typically consists of an (optical) imaging system designed to capture compressed measurements. The software decoder on the other hand refers to a reconstruction algorithm that retrieves the desired high-dimensional signal from those measurements. In this paper, using deep unfolding ideas, we propose an SCI recovery algorithm, namely GAP-net, which unfolds the generalized alternating projection (GAP) algorithm. At each stage, GAP-net passes its current estimate of the desired signal through a trained convolutional neural network (CNN). The CNN operates as a denoiser that projects the estimate back to the desired signal space. For the GAP-net that employs trained auto-encoder-based denoisers, we prove a probabilistic global convergence result. Finally, we investigate the performance of GAP-net in solving video SCI and spectral SCI problems. In both cases, GAP-net demonstrates competitive performance on both synthetic and real data. In addition to having high accuracy and high speed, we show that GAP-net is flexible with respect to signal modulation implying that a trained GAP-net decoder can be applied in different systems. Our code is at https://github.com/mengziyi64/ADMM-net.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR | 31.38| 27.67| 37.59 | 41.11|  28.97| 28.00|  32.47| 
|SSIM |0.9362|0.9200|0.9754|0.9902|0.9165|0.9330|0.9452|

## Multi Platform Running Time Analysis 
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX 8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.0552   | 0.0207  |  0.0206 |  0.0290 |  0.0196|

## Training GAP-net
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](cacti/docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/GAP-net/gapnet.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/GAP-net/gapnet.py
```

## Testing GAP-net on Simulation Dataset
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/GAP-net/gapnet.py --weights=checkpoints/gapnet/gapnet.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via  [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).

## Testing GAP-net on Real Dataset
Launch GAP-net on real dataset by executing the statement below.

```
python tools/real_data/test_deeplearning.py configs/GAP-net/gapnet_real_cr10.py --weights=checkpoints/gapnet/gapnet_real_cr10.pth

```
Notice:

* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via dropbox.
* Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the cr value in file in *gapnet_real_cr10.py* and retrain the model.

## Citation 
```
@article{Meng2020gap,
  title = {GAP-net for snapshot compressive imaging},
  author = {Meng, Ziyi and Jalali, Shirin and Yuan, Xin},
  journal = {arXiv preprint arXiv:2012.08364},
  year = {2020}
}
```