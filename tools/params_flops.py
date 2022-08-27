from thop import profile
import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from cacti.utils.mask import generate_masks
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.utils.logger import Logger
import argparse 

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
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])
    log_dir = osp.join(args.work_dir,"params_flops")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    dash_line = '-' * 80 + '\n'
    model = build_model(cfg.model).to(device)
    logger.info('Model Info:\n'
            + dash_line + 
            str(model)+'\n'+
            dash_line)

    mask,mask_s = generate_masks(cfg.test_data.mask_path,cfg.test_data.mask_shape)
    frames,height,width = mask.shape
    mask = torch.from_numpy(mask)
    mask_s = torch.from_numpy(mask_s)
    meas = torch.rand_like(mask_s)
    meas = meas.float().to(device)
    mask = mask.float().to(device)
    mask_s = mask_s.float().to(device)
    meas = meas.expand(1,1,height,width)
    Phi = mask.expand(1,frames,height,width)
    Phi_s= mask_s.expand(1,1,height,width)
    macs, params = profile(model, inputs=(meas,Phi,Phi_s))
    logger.info("\n")
    logger.info('Params and FLOPs Info:\n'
            + dash_line + 
            "Params: {:.2f} M.\n".format(params/1e6)+ 
            "FLOPs: {:.2f} G. \n".format(macs/1e9)+
            dash_line)
if __name__=="__main__":
    main()