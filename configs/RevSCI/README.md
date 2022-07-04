# Memory-efficient network for large-scale video compressive sensing
## Abstract
 Video snapshot compressive imaging (SCI) captures a sequence of video frames in a single shot using a 2D detector. The underlying principle is that during one exposure time, different masks are imposed on the high-speed scene to form a compressed measurement. With the knowledge of jmasks, optimization algorithms or deep learning methods are employed to reconstruct the desired high-speed video frames from this snapshot measurement. Unfortunately, though these methods can achieve decent results, the long running time of optimization algorithms or huge training memory occupation of deep networks still preclude them in practical applications. In this paper, we develop a memory-efficient network for large-scale video SCI based on multi-group reversible 3D convolutional neural networks. In addition to the basic model for the grayscale SCI system, we take one step further to combine demosaicing and SCI reconstruction to directly recover color video from Bayer measurements. Extensive results on both simulation and real data captured by SCI cameras demonstrate that our proposed model outperforms previous state-of-the-art with less memory and thus can be used in large-scale problems. The code is at https://github.com/BoChenGroup/RevSCI-net.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR| 33.72| 30.02 | 39.40|  42.93|  29.35 | 28.13|  33.92| 
|SSIM|0.9572|0.9498|0.9775|0.9924|0.9245|0.9363|0.9563|

## Multi Platform Running Time Analysis 
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.3011  | 0.2851  |  0.2002 |  0.2252 |  0.2124|

## Training RevSCI 
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](cacti/docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/RevSCI/revsci.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/RevSCI/revsci.py
```

## Testing RevSCI on Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/RevSCI/revsci.py --weights=checkpoints/revsci/revsci.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).


## Testing RevSCI on Real Dataset 
Launch RevSCI on real dataset by executing the statement below.

```
python tools/real_data/test_deeplearning.py configs/RevSCI/revsci_real_cr10.py --weights=checkpoints/revsci/revsci_real_cr10.pth

```
Notice:

* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).
* Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the cr value in file in *revsci_real_cr10.py* and retrain the model.

## Citation
```
@inproceedings{Cheng2021d,
  title = {Memory-efficient network for large-scale video compressive sensing}
  author = {Cheng, Ziheng and Chen, Bo and Liu, Guanliang and Zhang, Hao and Lu, Ruiying and Wang, Zhengjue and Yuan, Xin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {16246--16255},
  year = {2021}
}
```