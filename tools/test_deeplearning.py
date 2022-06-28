import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import save_single_image,get_device_info,load_checkpoints
from cacti.utils.metrics import compare_psnr,compare_ssim
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.datasets.builder import build_dataset 
from cacti.utils.logger import Logger
import numpy as np 
import argparse 
import time
import einops 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--weights",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',config_name)
    mask,mask_s = generate_masks(cfg.test_data.mask_path)
    cr = mask.shape[0]
    if args.weights is None:
        args.weights = cfg.checkpoints

    log_dir = osp.join(args.work_dir,"test_log")
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
    test_data = build_dataset(cfg.test_data,{"mask":mask})
    data_loader = DataLoader(test_data,batch_size=1,shuffle=False)

    model = build_model(cfg.model).to(device)
    logger.info("Load pre_train model...")
    resume_dict = torch.load(cfg.checkpoints)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict,strict=True)

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
        psnr,ssim = 0,0
        batch_output = []
        meas, gt = data
        gt = gt[0].numpy()
        meas = meas[0].float().to(device)
        batch_size = meas.shape[0]

        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                outputs = model(single_meas, Phi, Phi_s)
                torch.cuda.synchronize()
                end = time.time()
                run_time = end - start
                if ii>0:
                    sum_time += run_time
                    time_count += 1
            if not isinstance(outputs,list):
                outputs = [outputs]
            output = outputs[-1][0].cpu().numpy()
            batch_output.append(output)
            for jj in range(cr):
                if output.shape[0]==3:
                    per_frame_out = output[:,jj]
                    per_frame_out = np.sum(per_frame_out*test_data.rgb2raw,axis=0)
                else:
                    per_frame_out = output[jj]
                per_frame_gt = gt[ii,jj, :, :]
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
        psnr = psnr / (batch_size * cr)
        ssim = ssim / (batch_size * cr)
        logger.info("{}, Mean PSNR: {:.4f} Mean SSIM: {:.4f}.".format(
                    test_data.data_name_list[data_iter],psnr,ssim))
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    logger.info('Average Run Time:\n' 
            + dash_line + 
            "{:.4f} s.".format(sum_time/time_count) + '\n' +
            dash_line)
    test_dir = osp.join(args.work_dir,"test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data.data_name_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        for j in range(out.shape[0]):
            image_dir = osp.join(test_dir,_name)
            if not osp.exists(image_dir):
                os.makedirs(image_dir)
            save_single_image(out[j],image_dir,j,name=config_name)

    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    
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

if __name__=="__main__":
    main()