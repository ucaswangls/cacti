# 图像到视频或gif转换

## 图像到视频的转换
执行以下命令将图片转换为视频 (以Unet测试结果为例)
```
python tools/video_gif/images2video.py --images_dir="work_dirs/unet/test_images" --fps=4 --size_h=256 --size_w=256
```
* --images_dir 需要转换的图片目录
* --fps 转换的视频帧率
* --size_h 转换后的视频高度
* --size_w 转换后的视频宽度

转换后的视频默认保存在 images_dir 同级目录的video目录下

## 图像到gif的转换
执行以下命令将图片转换为gif (以Unet测试结果为例)
```
python tools/video_gif/images2gif.py --images_dir="work_dirs/unet/test_images" 
```
* --images_dir 需要转换的图片目录

转换后的gif默认保存在 images_dir 同级目录的gif目录下