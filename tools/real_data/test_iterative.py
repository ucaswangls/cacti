import os
import os.path as osp
import sys 
BASE_DIR=osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(BASE_DIR)
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_real_masks
from cacti.utils.utils import At,save_single_image,get_device_info
from cacti.models.builder import build_model
from cacti.models.gap_denoise import GAP_denoise 
from cacti.models.admm_denoise import ADMM_denoise
from cacti.datasets.builder import build_dataset 
from cacti.utils.config import Config
from cacti.utils.logger import Logger
import torch
import numpy as np 
import argparse
import time 
import einops
import json 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])

    log_dir = osp.join(args.work_dir,"log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)

    dash_line = '-' * 80 + '\n' 
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line)
    
    logger.info('cfg info:\n'
            + dash_line + 
            json.dumps(cfg, indent=4)+'\n'+
            dash_line) 

    mask,mask_s = generate_real_masks(
        frames=cfg.real_data.cr,
        mask_path=osp.expanduser(cfg.real_data.mask_path)
    )
    test_data = build_dataset(cfg.real_data)
    data_loader = DataLoader(test_data,1)
    denoiser = build_model(cfg.model).to(device)
    if cfg.checkpoints is not None:
        denoiser.load_state_dict(torch.load(cfg.checkpoints))
    test_dir = osp.join(args.work_dir,"test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)
    denoiser.to(device)
    denoiser.eval()
    
    sigma_list = torch.tensor(cfg.sigma_list,dtype=torch.float32).to(device)
    iter_list = cfg.iter_list
    
    Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
    Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)

    Phi = torch.from_numpy(Phi).to(args.device)
    Phi_s = torch.from_numpy(Phi_s).to(args.device)
    
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    sum_time=0.0
    time_count = 0
    for data_iter,data in enumerate(data_loader):
        name = test_data.data_name_list[data_iter]
        logger.info("Reconstruction {}:".format(name))
        batch_output = []
        meas = data
        meas = meas[0].float().to(device)
        batch_size = meas.shape[0]
        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            y1 = torch.zeros_like(single_meas) 
            x = At(single_meas,Phi)
            b = torch.zeros_like(x)
            sum_iter = 0
            start_time = time.time()
            for iter,iter_num in enumerate(iter_list):
                sigma = sigma_list[iter]
                for i in range(iter_num):
                    if cfg.denoise_method =="ADMM":
                        x,b = ADMM_denoise(denoiser,x,single_meas,b,Phi,Phi_s,sigma)
                    elif cfg.denoise_method =="GAP":
                        x,y1 = GAP_denoise(denoiser,x,single_meas,y1,Phi,Phi_s,sigma)
                    else:
                        raise TypeError("denoise method undefined!")
                if cfg.show_flag:
                    sum_iter += iter_num
                    logger.info("Batch num: {}, iter num: {}.".format(ii,sum_iter))
            torch.cuda.synchronize()
            end_time = time.time()
            if ii>0:
                run_time = end_time-start_time
                sum_time += run_time
                time_count += 1
            
            # output = x[0].cpu().numpy()
            output = x.cpu().numpy()
            batch_output.append(output)
            # break
        if cfg.show_flag:
            logger.info(" ")   
        _name = name.split("_")[1]
        out = np.array(batch_output)
        for j in range(out.shape[0]):
            image_dir = osp.join(test_dir,_name)
            if not osp.exists(image_dir):
                os.makedirs(image_dir)
            save_single_image(out[j],image_dir,j,name=osp.basename(args.work_dir))
    if time_count==0:
        time_count=1
    logger.info('Average Run Time:\n' 
            + dash_line + 
            "{:.4f} s.".format(sum_time/time_count) + '\n' +
            dash_line)
