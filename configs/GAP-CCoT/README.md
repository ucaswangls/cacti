# Spectral Compressive Imaging Reconstruction Using Convolution and Contextual Transformer 
## Abstract
Spectral compressive imaging (SCI) is able to encode the high-dimensional hyperspectral image
into a two-dimensional (2D) snapshot measurement, and then use algorithms to reconstruct
the spatio-spectral data-cube. At present, the main bottleneck of SCI is the reconstruction
algorithm, and state-of-the-art (SOTA) reconstruction methods generally face the problems of
long reconstruction time and/or poor detail recovery. In this paper, we propose a novel hybrid
network module, namely CCoT (Convolution and Contextual Transformer) block, which can
simultaneously acquire the inductive bias ability of convolution and the powerful modeling
ability of Transformer, which is conducive to improving the quality of reconstruction to restore
fine details. We integrate the proposed CCoT block into a physics-driven deep unfolding
framework based on the generalized alternating projection algorithm, and further propose the
GAP-CCoT network. Finally, we apply the GAP-CCoT algorithm to SCI reconstruction. Through
experiments on a large amount of synthetic data and real data, our proposed model achieves
higher reconstruction quality (>2dB in PSNR on simulated benchmark datasets) and shorter
running time than existing SOTA algorithms by a large margin. The code and models are publicly
available at https://github.com/ucaswangls/GAP-CCoT.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe  |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----: |:----:|:-----:|:----:  | :-----:|:----: |:----:|
|PSNR | 32.58| 29.03| 39.12 | 42.54 |  29.40| 28.52|  33.53| 
|SSIM |0.9494|0.9378|0.9795|0.9922|0.9229|0.9411|0.9538|

## Multi Platform Running Time Analysis 
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX 8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.1079   | 0.0701  |  0.0650 |  0.0742 |  0.0642|

## Training GAP-CCoT
Install the corresponding cupy according to the cuda version, please refer to the [cupy](https://cupy.dev/) official website for details.

Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](../../docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/GAP-CCoT/gapccot.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/GAP-CCoT/gapccot.py
```

## Testing  on Simulation Dataset
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/GAP-CCoT/gapccot.py --weights=checkpoints/gapccot/gapccot.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via  [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).

## Testing GAP-CCoT on Real Dataset
TODO
## Citation
```
@article{wang2022snapshot,
  title={Snapshot spectral compressive imaging reconstruction using convolution and contextual Transformer},
  author={Wang, Lishun and Wu, Zongliang and Zhong, Yong and Yuan, Xin},
  journal={Photonics Research},
  volume={10},
  number={8},
  pages={1848--1858},
  year={2022},
  publisher={Optica Publishing Group}
}
```