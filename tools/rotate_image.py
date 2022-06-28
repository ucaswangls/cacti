import cv2 
import os 
import os.path as osp 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",type=str,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--rotate",type=int,default=0)
    args = parser.parse_args()
    return args

def rotate_image(image,rotate):
    image_shape = image.shape
    if len(image_shape)==2:
        h,w = image_shape
    else:
        h,w,c = image_shape
    rot_mat =  cv2.getRotationMatrix2D((w/2,h//2), -rotate, 1)
    image = cv2.warpAffine(image, rot_mat, (w,h))
    return image

if __name__=="__main__":
    args = parse_args()
    assert args.image_dir is not None, "Image directory is None!"
    if args.dst_dir is None:
        args.dst_dir = "rotate_images"
    if not osp.exists(args.dst_dir):
        os.makedirs(args.dst_dir)
    if args.image_dir is not None:
        for image_name in os.listdir(args.image_dir):
            image_path = osp.join(args.image_dir,image_name)
            image = cv2.imread(image_path)
            image = rotate_image(image,args.ratate)
            cv2.imwrite(osp.join(args.dst_dir,image_name),image)
