_base_=["../_base_/matlab_bayer.py"]

test_data = dict(
    data_root="test_datasets/middle_scale",
    mask_path="test_datasets/mask/mid_color_mask.mat",
    rot_flip_flag=True
)

model = dict(
    type='FFDNetColor',
    in_nc = 3, 
    out_nc=3,  
    nc = 96,
    nb = 12,
    act_mode='R'
)

denoise_method="ADMM"
checkpoints="checkpoints/ffdnet/ffdnet_color.pth"

sigma_list = [50/255, 25/255, 12/255]
iter_list = [20, 20, 20] 
show_flag=True
demosaic=True
color_denoiser=True
use_cv2_demosaic=False
