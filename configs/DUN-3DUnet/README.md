# Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging 

## Abstract
Snapshot compressive imaging (SCI) aims to record three-dimensional signals via a two-dimensional camera. For the sake of building a fast and accurate SCI recovery algorithm, we incorporate the interpretability of modelbased methods and the speed of learning-based ones and present a novel dense deep unfolding network (DUN) with 3D-CNN prior for SCI, where each phase is unrolled from an iteration of Half-Quadratic Splitting (HQS). To better exploit the spatial-temporal correlation among frames and address the problem of information loss between adjacent phases in existing DUNs, we propose to adopt the 3D-CNN prior in our proximal mapping module and develop a novel dense feature map (DFM) strategy, respectively. Besides, in order to promote network robustness, we further propose a dense feature map adaption (DFMA) module to allow inter-phase information to fuse adaptively. All the parameters are learned in an end-to-end fashion. Extensive experiments on simulation data and real data verify the superiority of our method. The source code is available at https://github.com/jianzhangcs/SCI3D.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----: | :-----:|:----: |:----:|
|PSNR | 35.02| 31.78 | 40.92| 44.49 |  30.58 |  29.35 | 35.36| 
|SSIM |0.9681|0.9641 |0.9825|0.9940 |0.9411  |0.9532  |0.9672|

## Multi Platform Running Time Analysis
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  1.7320   | 0.6833  |  0.5784 |  0.9441|  0.5518 |

## Training DUN-3DUnet 
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](../../docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/DUN-3DUnet/dun3dunet.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/DUN-3DUnet/dun3dunet.py
```
## Testing DUN-3DUnet on Simulation Dataset
```
python tools/test_deeplearning.py configs/DUN-3DUnet/dun3dunet.py --weights=checkpoints/dun3dunet/dun3dunet.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).

## Testing DUN-3DUnet on Real Dataset
TODO

## Citation 
```
@article{wu2021dense,
  title={Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging},
  author={Wu, Zhuoyuan and Zhang, Jian and Mou, Chong},
  journal={arXiv preprint arXiv:2109.06548},
  year={2021}
}
```