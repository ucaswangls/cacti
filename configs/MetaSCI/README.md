# MetaSCI: Scalable and Adaptive Reconstruction for Video Compressive Sensing

## Abstract 
To capture high-speed videos using a two-dimensional detector, video snapshot compressive imaging (SCI) is a promising system, where the video frames are coded by different masks and then compressed to a snapshot measurement. Following this, efficient algorithms are desired to reconstruct the high-speed frames, where the state-of-the-art results are achieved by deep learning networks. However, these networks are usually trained for specific small-scale masks and often have high demands of training time and  GPU memory, which are hence not flexible to i) a new mask  with the same size and ii) a larger-scale mask. We address  these challenges by developing a Meta Modulated Convolutional Network for SCI reconstruction, dubbed MetaSCI. MetaSCI is composed of a shared backbone for different masks, and light-weight meta-modulation parameters to evolve to different modulation parameters for each mask, thus having the properties of fast adaptation to new masks (or systems) and ready to scale to large data. Extensive simulation and real data results demonstrate the superior performance of our proposed approach. Our code is available at https://github.com/xyvirtualgroup/MetaSCI-CVPR2021


## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----: | :-----:|:----: |:----:|
|PSNR  | 29.90 | 25.94| 36.37 |  39.60| 28.58  | 27.71 | 31.35| 
|SSIM  |0.9111 |0.8841|0.9703 |0.9871 |0.9067  |0.9280 |0.9312|

## Multi Platform Running Time Analysis
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.0160  | 0.0067  |  0.0056 |  0.0089 |  0.0052|

## Training MetaSCI 
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](../../docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/MetaSCI/metasci.py --distributed=True
```
* CUDA_VISIBLE_DEVICES: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/MetaSCI/metasci.py
```

## Testing MetaSCI on Simulation Dataset
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/MetaSCI/metasci.py --weights=checkpoints/metasci/metasci.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).

## Testing MetaSCI on Real Dataset 
Launch MetaSCI on real dataset by executing the statement below.

```
python tools/real_data/test_deeplearning.py configs/MetaSCI/metasci_real_cr10.py --weights=checkpoints/metasci/metasci_real_cr10.pth

```
Notice:

* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via dropbox.
* Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the cr value in file in *metasci_real_cr10.py* and retrain the model.

## Citation 
```
@inproceedings{Wang2021e,
  author = {Wang, Zhengjue and Zhang, Hao and Cheng, Ziheng and Chen, Bo and Yuan, Xin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {2083--2092},
  title = {{MetaSCI: Scalable and adaptive reconstruction for video compressive sensing}},
  year = {2021}
}
```