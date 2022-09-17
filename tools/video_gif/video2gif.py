import os
import os.path as osp
import argparse 
from moviepy.editor import VideoFileClip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,default="work_dirs/video/concat/ShakeNDry.avi")
    parser.add_argument("--dst_dir",type=str,default="work_dirs/gif/concat")
    parser.add_argument("--gif_name",type=str,default=None)
    parser.add_argument("--fps",type=int,default=8)
    args = parser.parse_args()
    return args 

if __name__=="__main__":
    args = parse_args()
    video_path = args.video_path
    dst_dir = args.dst_dir
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    gif_name = args.gif_name
    fps = args.fps
    if  gif_name is None:
        gif_name,_ = osp.splitext(osp.basename(video_path))
    video = VideoFileClip(video_path)
    video.write_gif("{}.gif".format(osp.join(dst_dir,gif_name)),fps=fps)