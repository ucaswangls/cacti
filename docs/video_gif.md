# Images to Video or to gif transfer

## Images to Video
Transfer images to video by executing following command

```
python tools/video_gif/images2video.py --images_dir="work_dirs/unet/test_images" --fps=4 --size_h=256 --size_w=256
```
* --images_dir  image directory 
* --fps video frame rate
* --size_h video pixel height
* --size_w video pixel width

Transferred video will be saved in *video* folder 

## Images to gif 
Transfer images to gif by executing following command

```
python tools/video_gif/images2gif.py --images_dir="work_dirs/unet/test_images" 
```
* --images_dir image directory 

Transferred gif will be saved in *gif* folder 