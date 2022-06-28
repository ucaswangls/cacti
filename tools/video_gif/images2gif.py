from PIL import Image
import os 
import os.path as osp
import argparse 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",type=str,default="work_dirs/meas/test_images")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    images_dir = args.images_dir
    for class_dir in os.listdir(images_dir):
        _name,_  = osp.splitext(class_dir)
        images_path = osp.join(images_dir,class_dir)
        gif_dir = osp.join(osp.dirname(images_dir),"gif")
        if not osp.exists(gif_dir):
            os.makedirs(gif_dir)
        gif_path = osp.join(gif_dir,_name+".gif")
        imgs = []
        img_name_list = os.listdir(images_path)
        img_name_list.sort(key = lambda x: int(osp.splitext(x)[0].split("_")[-1]))
        for image_name in img_name_list:
            image_path = osp.join(images_path,image_name)
            image = Image.open(image_path)
            imgs.append(image)
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print("{}.gif convert success!".format(images_path))