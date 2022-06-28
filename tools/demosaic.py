import cv2 
import os 
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import argparse 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",type=str,default="work_dirs/tv_large/test_images/football")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    image_dir = args.image_dir
    _name = image_dir.split("/")[1]
    dst_dir = image_dir+"_demosaic"
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    for image_name in os.listdir(image_dir):
        if "_d" in image_name:
            continue
        image_path = osp.join(image_dir,image_name)
        image = cv2.imread(image_path,0)
        demosaic_image = cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR)
        cv2.imwrite(osp.join(dst_dir,_name+"_"+image_name),demosaic_image)