import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np 
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import save_image
from cacti.utils.metrics import compare_psnr,compare_ssim
from cacti.utils.config import Config
from cacti.datasets.builder import build_dataset 
import argparse 
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit

def tensorrt_init():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network,TRT_LOGGER)
    with open("work_dirs/unet/unet.onnx", 'rb') as model:
        parser.parse(model.read())

    # output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
    max_batch_size=1
    builder.max_batch_size = max_batch_size

    builder.max_workspace_size = 1 <<  20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    # with trt.Builder(TRT_LOGGER) as builder:
    engine = builder.build_cuda_engine(network)
    serialized_engine = engine.serialize()

    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    # with open(“sample.engine”, “wb”) as f:
    #         f.write(engine.serialize())
    # with open(“sample.engine”, “rb”) as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         engine = runtime.deserialize_cuda_engine(f.read())
    meas = engine.get_binding_shape("meas")
    mask = engine.get_binding_shape("mask")
    mask_s = engine.get_binding_shape("mask_s")
    output = engine.get_binding_shape("output")
    h_meas = cuda.pagelocked_empty(trt.volume(meas), dtype=np.float32)
    h_mask = cuda.pagelocked_empty(trt.volume(mask), dtype=np.float32)
    h_mask_s = cuda.pagelocked_empty(trt.volume(mask_s), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output), dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_meas = cuda.mem_alloc(h_meas.nbytes)
    d_mask = cuda.mem_alloc(h_mask.nbytes)
    d_mask_s = cuda.mem_alloc(h_mask_s.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_meas,h_mask,h_mask_s,h_output,d_meas,d_mask,d_mask_s,d_output,engine,stream

import numpy as np 

def to_numpy(tensor):
    tensor = tensor.contiguous().view(-1)
    tensor_numpy = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return tensor_numpy



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])
    mask,mask_s = generate_masks(cfg.data.test.mask_path)

    test_data = build_dataset(cfg.data.test)
    data_loader = DataLoader(test_data,batch_size=1,shuffle=False)

    # model = build_model(cfg.model).to(device)
    # model.load_state_dict(torch.load(cfg.checkpoints))
    # ort_session = onnxruntime.InferenceSession("work_dirs/gap_net/gap_net.onnx")
    h_meas,h_mask,h_mask_s,h_output,d_meas,d_mask,d_mask_s,d_output,engine,stream  = tensorrt_init()
    
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]

    for iter,data in enumerate(data_loader):
        psnr,ssim = 0,0
        batch_output = []

        meas, gt = data
        gt = gt[0].numpy()
        batch_size,frames,height,width = gt.shape
        meas = meas[0].float()
        mask = mask.float()
        mask_s = mask_s.float()
        mask = mask.expand(1,frames,height,width)
        mask_s = mask_s.expand(1,1,height,width)
        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            with engine.create_execution_context() as context:
                # outputs = model(single_meas, mask, mask_s)
                np.copyto(h_meas,to_numpy(single_meas))
                np.copyto(h_mask,to_numpy(mask))
                np.copyto(h_mask_s,to_numpy(mask_s))

                cuda.memcpy_htod_async(d_meas, h_meas, stream)
                cuda.memcpy_htod_async(d_mask, h_mask, stream)
                cuda.memcpy_htod_async(d_mask_s, h_mask_s, stream)
                # Run inference.
                context.execute_async(bindings=[int(i) for i in [d_meas,d_mask,d_mask_s,d_output]], stream_handle=stream.handle)
                # Transfer predictions back from the GPU.
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                # outputs = np.zeros_like(h_output)
                outputs = h_output
                # np.copyto(outputs,h_output)
                outputs = np.reshape(outputs,(1,8,256,256))
                # Synchronize the stream
                stream.synchronize() 
                # ort_inputs = {inputs[0].name: to_numpy(single_meas),
                #             inputs[1].name: to_numpy(mask),
                #             inputs[2].name: to_numpy(mask_s)}
                # outputs = ort_session.run(None, ort_inputs)
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
