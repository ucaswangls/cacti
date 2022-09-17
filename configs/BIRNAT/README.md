# BIRNAT: Bidirectional Recurrent Neural Networks with Adversarial Training for Video Snapshot Compressive Imaging 
## Abstract 
We consider the problem of video snapshot compressive imaging (SCI), where multiple high-speed frames are coded by different masks and then summed to a single measurement. This measurement and the modulation masks are fed into our Recurrent Neural Network (RNN) to reconstruct the desired high-speed frames. Our end-to-end sampling and reconstruction system is dubbed Bidirectional Recurrent Neural networks with Adversarial Training (BIRNAT). To our best knowledge, this is the first time that recurrent networks are employed to SCI problem. Our proposed BIRNAT outperforms other deep learning based algorithms and the state-of-the-art optimization based algorithm, DeSCI, through exploiting the underlying correlation of sequential video frames. BIRNAT employs a deep convolutional neural network with Resblock and feature map self-attention to reconstruct the first frame, based on which bidirectional RNN is utilized to reconstruct the following frames in a sequential manner. To improve the quality of the reconstructed video, BIRNAT is further equipped with the adversarial training besides the mean square error loss. Extensive results on both simulation and real data (from two SCI cameras) demonstrate the superior performance of our BIRNAT system. The codes are available at https://github.com/BoChenGroup/BIRNAT.

## Testing Result on Six Simulation Dataset
|Dataset|Kobe |Traffic|Runner| Drop  | Aerial | Vehicle|Average|
|:----:|:----:|:----: |:----:|:-----:|:----: | :-----:|:----: |
|PSNR  | 32.71| 29.33 | 38.70|  42.28|  28.99|  27.84 | 33.31 | 
|SSIM  |0.9505|0.9434 |0.9766| 0.9919| 0.9176|  0.9270|0.9512 |
## Multi Platform Running Time Analysis 
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|  0.2311   | 0.1155  |  0.0988 |  0.1684 |  0.1179|

## Training BIRNAT
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](../../docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/BIRNAT/birnat.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/BIRNAT/birnat.py
```

## Testing BIRNAT on Simulation Dataset
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test_deeplearning.py configs/BIRNAT/birnat.py --weights=checkpoints/birnat/birnat.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via dropbox.

## Testing BIRNAT on Real Dataset 
TODO
## Citation
```
@inproceedings{Cheng2020birnat,  
  title = {BIRNAT: Bidirectional recurrent neural networks with adversarial training for video snapshot compressive imaging},  
  author = {Cheng, Ziheng and Lu, Ruiying and Wang, Zhengjue and Zhang, Hao and Chen, Bo and Meng, Ziyi and Yuan, Xin},  
  booktitle = {European Conference on Computer Vision},  
  pages = {258--275},  
  publisher = {Springer},  
  year = {2020}
}
```