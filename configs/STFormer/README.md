# Spatial-Temporal Transformer for Video Snapshot Compressive Imaging
## Abstract
Video snapshot compressive imaging (SCI)  captures multiple sequential video frames by a single measurement using the idea of computational imaging. The underlying principle is to modulate high-speed frames through different masks and these modulated frames are summed to a single measurement captured by a low-speed 2D sensor (dubbed optical encoder); following this, algorithms are employed to reconstruct the desired high-speed frames (dubbed software decoder) if needed.
In this paper, we consider the reconstruction algorithm in video SCI, ie, recovering a series of video frames from a compressed measurement. Specifically, we propose a Spatial-Temporal transFormer (STFormer) to exploit the correlation in both spatial and temporal domains. STFormer network is composed of a token generation block, a video reconstruction block, and these two blocks are connected by a series of STFormer blocks. 
Each STFormer block consists of a spatial self-attention branch, a temporal self-attention branch and the outputs of these two branches are integrated by a fusion network.
Extensive results on both simulated and real data demonstrate the state-of-the-art performance of STFormer.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe |Traffic|Runner| Drop  |Aerial|Vehicle|Average|
|:----:|:----:|:----: |:-----:|:----:|:----:|:----:|:----:|
|PSNR  | 35.51| 32.12 | 42.69 | 44.99|31.49 | 31.17| 36.33| 
|SSIM  |0.9724|0.9674 |0.9882 |0.9944|0.9523|0.9707|0.9742|

## Multi Platform Running Time Analysis 
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  1.6774  | 0.5411   |  0.4500 |  0.9268 |  0.4871|

## Training STFormer 
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](../../docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/STFormer/stformer_base.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/STFormer/stformer_base.py
```

## Testing STFormer on Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/STFormer/stformer_base.py --weights=checkpoints/stformer_base/stformer_base.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).


## Testing STFormer on Real Dataset 
Launch STFormer on real dataset by executing the statement below.

```
python tools/real_data/test_deeplearning.py configs/STFormer/stformer_base_real_cr10.py --weights=checkpoints/stformer_base/stformer_base_real_cr10.pth

```
Notice:

* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).
* Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the cr value in file in *stformer_base_real_cr10.py* and retrain the model.

## Citation
```

```