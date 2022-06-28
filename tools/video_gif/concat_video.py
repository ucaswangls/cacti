import numpy as np 
from PIL import Image,ImageDraw,ImageFont
import os 
import os.path as osp 
import einops
import cv2 
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",type=str,default="work_dirs/video/kobe")
    parser.add_argument("--dst_dir",type=str,default="work_dirs/video/concat")
    parser.add_argument("--h_pad",type=int,default=80)
    parser.add_argument("--w_pad",type=int,default=40)
    parser.add_argument("--font_size",type=int,default=34)
    parser.add_argument("--resize_h",type=int,default=512)
    parser.add_argument("--resize_w",type=int,default=512)
    args = parser.parse_args()
    return args

def readImg(image_path):
    cap = cv2.VideoCapture(image_path)
    frame_list = []
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    frame_list = np.array(frame_list)

    return frame_list 

def cv2ImgAddText(img, text, width, height, textColor, font_size):
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "C:/Windows/Fonts/Times New Roman/times.ttf", font_size)
    if "\n" in text:
        text1,text2 = text.split("\n")
        w1, h1 = fontStyle.getsize(text1)
        w2, h2 = fontStyle.getsize(text2)
        draw.text(((width-w1)/2, (height-h1-h2)/2-4), text1, textColor, font=fontStyle)
        draw.text(((width-w2)/2, (height-h1-h2)/2+h1+4), text2, textColor, font=fontStyle)
    else:
        w, h = fontStyle.getsize(text)
        draw.text(((width-w)/2, (height-h)/2), text, textColor, font=fontStyle)
    return np.asarray(img)

if __name__=="__main__":
    args = parse_args()
    video_dir = args.video_dir
    dst_dir = args.dst_dir
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    h_pad,w_pad = args.h_pad,args.w_pad
    font_size = args.font_size
    resize_h,resize_w = args.resize_h,args.resize_w

    _name = osp.basename(video_dir)
    # algorithms_name = [
    #     "Measurement",
    #     "Original",
    #     "GAP-TV",
    #     # "FFDNet-gray",
    #     "PnP-FFDNet-color",
    #     "PnP-FastDVDnet-gray",
    #     "PnP-FastDVDnet-color"
    # ]
    algorithms_name = [
        "Measurement",
        "Original",
        "GAP-TV",
        "PnP-FastDVDnet",
        "RevSCI",
        "DUN-3DUnet"
    ]
    video_list = []
    for video_name in algorithms_name:
        video_path = osp.join(video_dir,video_name.lower()+".avi")
        video = readImg(video_path)
        video_list.append(video)
    video_list[0] = einops.repeat(video_list[0],"f h w c->(f f2) h w c",f2=8)
    
    video_array = np.array(video_list)
    b,f,h,w,c= video_array.shape
    
    temp_array = np.zeros((b,f,resize_h,resize_w,c),dtype=np.uint8)
    
    temp_orig = []
    psnr_array = np.zeros((b,f))
    ssim_array = np.zeros((b,f))
    for i in range(b):
        for j in range(f):
            temp_img = video_array[i][j]
            if i==0:
                pass
            elif i==1:
                temp_orig.append(temp_img)
            else:
                psnr_array[i][j] = compare_psnr(temp_orig[j],temp_img)
                ssim_array[i][j] = compare_ssim(temp_orig[j],temp_img,channel_axis=-1)
            temp_img = cv2.resize(temp_img,(resize_w,resize_h))
            temp_array[i][j] = temp_img
    video_array = temp_array 
    b,f,h,w,c= video_array.shape
    w_blank = np.ones((b,f,h,w_pad,c),dtype=np.uint8)*255
    video_array = np.concatenate([video_array,w_blank],axis=3)
    b,f,h,w,c = video_array.shape
    h_blank = np.ones((b,f,h_pad,w,c),dtype=np.uint8)*255
    for i in range(b):
        for j in range(f):
            name=algorithms_name[i]
            if name == algorithms_name[0]:
                text = "{}: {}".format(name,j//8)
            elif name == algorithms_name[1]:
                text = "{}: {}".format(name,j+1)
            else:
                psnr_value = psnr_array[i][j]
                ssim_value = ssim_array[i][j]
                text = "{} \n(PSNR: {:.2f} dB, SSIM: {:.4f})".format(name,psnr_value,ssim_value)
            h_blank[i][j] = cv2ImgAddText(h_blank[i][j], text, height=h_pad,width=w-w_pad, textColor=(0), font_size=font_size)

    video_array = np.concatenate([h_blank,video_array],axis=2)
    b,f,h,w,c = video_array.shape
    video = cv2.VideoWriter("{}.avi".format(osp.join(dst_dir,_name)), cv2.VideoWriter_fourcc('I', '4', '2', '0'), 8, (w*b,h),True)
    for i in range(f):
        temp_array = einops.rearrange(video_array[:,i],"b h w c->h (b w) c")
        video.write(temp_array)
    video.release()
    cv2.destroyAllWindows()

