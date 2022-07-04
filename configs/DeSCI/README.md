# Rank minimization for snapshot compressive imaging
## Abstract
Snapshot compressive imaging (SCI) refers to compressive imaging systems where multiple frames are mapped into a single measurement, with video compressive imaging and hyperspectral compressive imaging as two representative applications. Though exciting results of high-speed videos and hyperspectral images have been demonstrated, the poor reconstruction quality precludes SCI from wide applications. This paper aims to boost the reconstruction quality of SCI via exploiting the high-dimensional structure in the desired signal. We build a joint model to integrate the nonlocal self-similarity of video/hyperspectral frames and the rank minimization approach with the SCI sensing process. Following this, an alternating minimization algorithm is developed to solve this non-convex problem. We further investigate the special structure of the sampling process in SCI to tackle the computational workload and memory issues in SCI reconstruction. Both simulation and real data (captured by four different SCI cameras) results demonstrate that our proposed algorithm leads to significant improvements compared with current state-of-the-art algorithms. We hope our results will encourage the researchers and engineers to pursue further in compressive imaging for real applications.
## Citation
```
@article{Liu2018rank,
  title = {Rank minimization for snapshot compressive imaging},
  author = {Liu, Yang and Yuan, Xin and Suo, Jinli and Brady, David J and Dai, Qionghai},
  issn = {0162-8828},
  journal = {IEEE transactions on pattern analysis and machine intelligence},
  number = {12},
  pages = {2990--3006},
  publisher = {IEEE},
  volume = {41},
  year = {2018}
}
```