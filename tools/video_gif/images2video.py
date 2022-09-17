import os
import os.path as osp
import cv2
import argparse 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",type=str,default="work_dirs/stformer_base_mid_color/test_images")
    parser.add_argument("--fps",type=int,default=4)
    parser.add_argument("--size_h",type=int,default=512)
    parser.add_argument("--size_w",type=int,default=512)
    args = parser.parse_args()
    return args

def get_image_num(image_name):
    _num = osp.splitext(image_name)[0].split("_")[-1]
    return int(_num)

def image2video(imgs_path,fps,size):    
    img_names = os.listdir(imgs_path)
    img_names.sort(key=get_image_num)
    video_name = osp.basename(imgs_path)
    video_dir = osp.join(osp.dirname(osp.dirname(imgs_path)),"video")
    if not osp.exists(video_dir):
        os.makedirs(video_dir)
    video_path = osp.join(video_dir,video_name+".avi")
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    for idx in range(len(img_names)):      
        img_file = os.path.join(imgs_path, img_names[idx])
        print(img_file)
        img = cv2.imread(img_file)
        video.write(img)
    
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    args = parse_args()
    for class_dir in os.listdir(args.images_dir):
        images_path = osp.join(args.images_dir,class_dir)
        image2video(images_path,fps=args.fps,size=(args.size_h,args.size_w))