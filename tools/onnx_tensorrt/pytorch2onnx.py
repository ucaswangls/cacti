import os
import os.path as osp
import torch
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(BASE_DIR)
from cacti.utils.mask import generate_masks
from cacti.utils.config import Config
from cacti.utils.loss_builder import build_model
import argparse 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--simple_flag",type=bool,default=True)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

if __name__=="__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    mask,mask_s = generate_masks(cfg.train_data.mask_path)
    model_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',model_name)
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(cfg.checkpoints))
    model = model.to(args.device)
    # input = torch.randn(4,3,56,336)
    frames,height,width = mask.shape
    mask = mask.expand(1,frames,height,width).to(args.device)
    mask_s = mask_s.expand(1,1,height,width).to(args.device)
    meas = torch.randn(1,1,height,width).to(args.device)
    if not osp.exists(args.work_dir):
        os.makedirs(args.work_dir)
    out_onnx_name = osp.join(args.work_dir,model_name+".onnx") 
    simple_onnx_name = osp.join(args.work_dir,model_name+"_simple.onnx")
    torch.onnx.export(model,(meas,mask,mask_s),out_onnx_name,input_names=["meas","mask","mask_s"],output_names=["output"],opset_version=13)
    
    print("{} to onnx model has been successfully transformed".format(model_name))

    if args.simple_flag:
        os.system("python -m onnxsim {} {}".format(out_onnx_name,simple_onnx_name))