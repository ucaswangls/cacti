import os
import os.path as osp
import sys 
BASE_DIR=osp.dirname(osp.dirname(__file__))
sys.path.append(BASE_DIR)
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import At,save_single_image,get_device_info
from cacti.models.builder import build_model
from cacti.models.gap_denoise import GAP_denoise 
from cacti.models.admm_denoise import ADMM_denoise
from cacti.datasets.builder import build_dataset 
from cacti.utils.config import Config
from cacti.utils.logger import Logger
from cacti.utils.metrics import compare_psnr,compare_ssim
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

    mask, mask_s = generate_masks(cfg.test_data.mask_path)
    test_data = build_dataset(cfg.test_data,{"mask":mask})
    data_loader = DataLoader(test_data,1)
    denoiser = build_model(cfg.model).to(device)
    if cfg.checkpoints is not None:
        denoiser.load_state_dict(torch.load(cfg.checkpoints))
    denoiser.to(device)
    denoiser.eval()
    
    sigma_list = torch.tensor(cfg.sigma_list,dtype=torch.float32).to(device)
    iter_list = cfg.iter_list
    
    mask = torch.from_numpy(mask).to(device)
    frames,height,width = mask.shape
    mask_s = torch.from_numpy(mask_s).to(device)
    mask = einops.repeat(mask,"f h w-> b f h w",b=1)
    mask_s = einops.repeat(mask_s,"h w-> b f h w",b=1,f=1)
    mask_bayer = torch.zeros((1,frames, height//2, width//2, 4),device=mask.device)
    mask_s_bayer = torch.zeros((1,1,height//2, width//2, 4),device=mask_s.device)
    bayer = [[0,0], [0,1], [1,0], [1,1]]
    for ib in range(len(bayer)): 
        b = bayer[ib]
        mask_bayer[...,ib] = mask[:,:,b[0]::2, b[1]::2]
        mask_s_bayer[...,ib] =  mask_s[:,:,b[0]::2, b[1]::2]
    mask_bayer = einops.rearrange(mask_bayer,"b f h w ba->(b ba) f h w")
    mask_s_bayer = einops.rearrange(mask_s_bayer,"b f h w ba->(b ba) f h w")
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    sum_time=0.0
    time_count = 0

    color_denoiser=False
    use_cv2_demosaic=False
    if "color_denoiser" in cfg.keys() and cfg["color_denoiser"]:
        color_denoiser=True
    if "use_cv2_demosaic" in cfg.keys() and cfg["use_cv2_demosaic"]:
        use_cv2_demosaic=True
    for data_iter,data in enumerate(data_loader):
        logger.info("Reconstruction {}:".format(test_data.data_name_list[data_iter]))
        psnr,ssim = 0,0
        batch_output = []
        meas, gt = data
        gt = gt[0].cpu().numpy()
        meas = meas[0].float().to(device)
        batch_size = meas.shape[0]
        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            temp_meas = torch.zeros((1,1,height//2, width//2, 4),device=meas.device)
            for ib in range(len(bayer)): 
                b = bayer[ib]
                temp_meas[...,ib] = single_meas[:,:,b[0]::2, b[1]::2]
            single_meas = einops.rearrange(temp_meas,"b f h w ba->(b ba) f h w")
            y1 = torch.zeros_like(single_meas) 
            x = At(single_meas,mask_bayer)
            b_hat = torch.zeros_like(x)
            sum_iter = 0
            start_time = time.time()
            for iter,iter_num in enumerate(iter_list):
                sigma = sigma_list[iter]
                for i in range(iter_num):
                    if cfg.denoise_method =="ADMM":
                        x,b_hat = ADMM_denoise(denoiser,
                                            x,
                                            single_meas,
                                            b_hat,
                                            mask_bayer,
                                            mask_s_bayer,
                                            sigma,
                                            color_denoiser=color_denoiser,
                                            bayer=bayer,
                                            use_cv2_demosaic=use_cv2_demosaic
                                )
                    elif cfg.denoise_method =="GAP":
                        x,y1 = GAP_denoise(denoiser,
                                            x,
                                            single_meas,
                                            y1,
                                            mask_bayer,
                                            mask_s_bayer,
                                            sigma,
                                            color_denoiser=color_denoiser,
                                            bayer=bayer,
                                            use_cv2_demosaic=use_cv2_demosaic
                                )
                    else:
                        raise TypeError("denoise method undefined!")

                if cfg.show_flag:
                    single_psnr,single_ssim = 0,0
                    output = x.cpu().numpy()
                    temp_output = np.zeros_like(gt[ii])
                    for ib in range(len(bayer)): 
                        b = bayer[ib]
                        temp_output[:,b[0]::2, b[1]::2] = output[ib,...]
                    for jj in range(frames):
                        per_frame_out = temp_output[jj]
                        per_frame_gt = gt[ii,jj, :, :]
                        single_psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                        single_ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
                    sum_iter += iter_num
                    logger.info("Batch num: {}, iter num: {}, PSNR: {:.4f}, SSIM: {:.4f}.".format(ii,sum_iter,single_psnr/frames,single_ssim/frames))
            torch.cuda.synchronize()
            end_time = time.time()
            if ii>0:
                run_time = end_time-start_time
                sum_time += run_time
                time_count += 1
            if cfg.show_flag:
                logger.info(" ")
            output = x.cpu().numpy()
            temp_output = np.zeros_like(gt[ii])
            for ib in range(len(bayer)): 
                b = bayer[ib]
                temp_output[:,b[0]::2, b[1]::2] = output[ib,...]
            batch_output.append(temp_output)
            for jj in range(frames):
                per_frame_out = temp_output[jj]
                per_frame_gt = gt[ii,jj, :, :]
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
        test_dir = osp.join(args.work_dir,"test_images")
        name = test_data.data_name_list[data_iter]
        _name,_ = name.split("_")
        image_dir = osp.join(test_dir,_name)
        if not osp.exists(image_dir):
            os.makedirs(image_dir)
        out = np.array(batch_output)
        for j in range(out.shape[0]):
            if "demosaic" in cfg.keys() and cfg["demosaic"]:
                demosaic=True
            else:
                demosaic=False
            save_single_image(out[j],image_dir,j,name=osp.basename(args.work_dir),demosaic=demosaic)

        psnr = psnr / (batch_size * frames)
        ssim = ssim / (batch_size * frames)
        logger.info("{}, Mean PSNR: {:.4f} Mean SSIM: {:.4f}.\n".format(
                    test_data.data_name_list[data_iter],psnr,ssim))
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(out))
        gt_list.append(gt)

    if time_count==0:
        time_count=1
    logger.info('Average Run Time:\n' 
            + dash_line + 
            "{:.4f} s.".format(sum_time/time_count) + '\n' +
            dash_line)
    for i,name in enumerate(test_data.data_name_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
    psnr_dict["psnr_mean"] = np.round(np.mean(psnr_list),4)
    ssim_dict["ssim_mean"] = np.round(np.mean(ssim_list),4)

    psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
    ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])

    logger.info("Mean PSNR: \n"+
                dash_line + 
                "{}.\n".format(psnr_str)+
                dash_line)

    logger.info("Mean SSIM: \n"+
                dash_line + 
                "{}.\n".format(ssim_str)+
                dash_line)
