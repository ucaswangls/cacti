# Ensemble Learning Priors Driven Deep Unfolding for Scalable Snapshot Compressive Imaging
## Abstract
Snapshot compressive imaging (SCI) can record the
3D information by a 2D measurement and from this 2D
measurement to reconstruct the original 3D information by
reconstruction algorithm. As we can see, the reconstruction
algorithm plays a vital role in SCI. Recently, deep learning
algorithm show its outstanding ability, outperforming the
traditional algorithm. Therefore, to improve deep learning
algorithm reconstruction accuracy is an inevitable topic for
SCI. Besides, deep learning algorithms are usually limited
by scalability, and a well trained model in general can
not be applied to new systems if lacking the new training
process. To address these problems, we develop the ensem-
ble learning priors to further improve the reconstruction
accuracy and propose the scalable learning to empower
deep learning the scalability just like the traditional
algorithm. What’s more, our algorithm has achieved the
state-of-the-art results, outperforming existing algorithms.
Extensive results on both simulation and real datasets
demonstrate the superiority of our proposed algorithm.
The code and models will be released to the public.
https://github.com/integritynoble/ELP-Unfolding/tree/master

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR | 34.41| 31.58 | 41,16 | 44.99 |  30.68 | 29.65 |  35.41 | 
|SSIM |0.9657|0.9623 |0.9857|0.9947 |0.9438 |0.9589 |0.9685|

## Multi Platform Running Time Analysis 
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX 8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.9754   | 0.3974  |  0.3364 |  0.6250 |  0.2790|

## Training ELP-Unfolding

Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](cacti/docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/ELP-Unfolding/elpunfolding.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/ELP-Unfolding/elpunfolding.py
```

## Testing ELP-Unfolding on Simulation Dataset
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/ELP-Unfolding/elpunfolding.py --weights=checkpoints/elpunfolding/elpunfolding.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via  [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).

## Testing ELP-Unfolding on Real Dataset
TODO
## Citation
```
@article{yang2022ensemble,
  title={Ensemble learning priors unfolding for scalable Snapshot Compressive Sensing},
  author={Yang, Chengshuai and Zhang, Shiyu and Yuan, Xin},
  journal={arXiv preprint arXiv:2201.10419},
  year={2022}
}
```