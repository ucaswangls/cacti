import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(BASE_DIR)
import torch 
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import save_image
from cacti.utils.metrics import compare_psnr,compare_ssim
from cacti.utils.config import Config
from cacti.datasets.builder import build_dataset 
import numpy as np 
import argparse 
import onnxruntime
import time 

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = args.device
    model_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',model_name)
    mask,mask_s = generate_masks(cfg.test_data.mask_path)

    test_data = build_dataset(cfg.test_data)
    data_loader = DataLoader(test_data,batch_size=1,shuffle=False)

    out_onnx_name = osp.join(args.work_dir,model_name+".onnx")
    ort_session = onnxruntime.InferenceSession(out_onnx_name)
    ort_session.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}])
    
    inputs = ort_session.get_inputs()

    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]

    sum_time=0.0
    time_count = 0

    for iter,data in enumerate(data_loader):
        print("Computer {}.".format(test_data.data_name_list[iter]))
        psnr,ssim = 0,0
        batch_output = []
        meas, gt = data
        gt = gt[0].numpy()
        batch_size,frames,height,width = gt.shape
        meas = meas[0].float().to(device)
        mask = mask.float().to(device)
        mask_s = mask_s.float().to(device)
        mask = mask.expand(1,frames,height,width)
        mask_s = mask_s.expand(1,1,height,width)
        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)

            ort_inputs = {inputs[0].name: to_numpy(single_meas),
                        inputs[1].name: to_numpy(mask),
                        inputs[2].name: to_numpy(mask_s)}
            torch.cuda.synchronize()
            start = time.time()
            outputs = ort_session.run(None, ort_inputs)
            end = time.time()
            torch.cuda.synchronize()
            run_time = end-start
            if iter>0:
                sum_time += run_time
                time_count += 1
            if not isinstance(outputs,list):
                outputs = [outputs]
            output = outputs[-1][0]

            batch_output.append(output)
            for jj in range(frames):
                per_frame_out = output[jj]
                per_frame_gt = gt[ii,jj, :, :]
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
            
        psnr = psnr / (batch_size * frames)
        ssim = ssim / (batch_size * frames)
        psnr_list.append(np.round(psnr,4))
        ssim_list.append(np.round(ssim,4))
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    print("onnx runtime: {:.4f}".format(sum_time/time_count))
    test_dir = osp.join(args.work_dir,"test")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data.data_name_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        for j in range(out.shape[0]):
            image_name = osp.join(test_dir,_name+"_"+str(j)+".png")
            save_image(out[j],gt[j],image_name)
    psnr_dict["psnr_mean"] = np.round(np.mean(psnr_list),4)
    ssim_dict["ssim_mean"] = np.round(np.mean(ssim_list),4)
    print(psnr_dict)
    print(ssim_dict)

if __name__=="__main__":
    main()
