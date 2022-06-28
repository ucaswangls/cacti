# CACTI代码库参考说明文档
## configs文件夹
* __base__文件夹：模型和数据的一些基本配置
* 各个重建算法的配置文件
## cacti文件夹
* datasets （数据预处理的具体实现）
* models （重建算法的具体的实现）
* utils （一些通用函数，如PSNR,SSIM值的计算）

## tools文件夹
* train.py （模型训练）
* test_deeplearning.py （在灰度仿真数据或彩色仿真数据集上测试端到端的深度学习算法或深度展开算法）
* test_iterative.py（在灰度仿真数据集上测试迭代优化算法或即插即用算法）
* test_color_iterative.py（在彩色仿真数据集上测试迭代优化算法或即插即用算法）
* real_data (对real data 进行测试)
* params_floats.py （模型参数量与计算复杂度的统计）
* video_gif （图像到视频与动态图的转换）
* onnx_tensorrt （onnx, tensorrt 模型转换与测试）

## test_datasets文件夹  
* mask （不同模型的mask值）
* simulation （六个基准灰度仿真数据）
* middle_scale （六个基准middle彩色仿真数据）
* real_data （真实数据，压缩率从10到50）


