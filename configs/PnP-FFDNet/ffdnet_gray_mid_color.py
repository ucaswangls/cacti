_base_=["../_base_/matlab_bayer.py"]

test_data = dict(
    data_root="test_datasets/middle_scale",
    mask_path="test_datasets/mask/mid_color_mask.mat",
    rot_flip_flag=True
)

model = dict(
    type='FFDNet',
    num_input_channels=1
)

denoise_method="GAP"
checkpoints="checkpoints/ffdnet/ffdnet_gray.pth"

sigma_list = [50/255, 25/255, 12/255,6/255]
iter_list = [20, 20, 20, 10] 
show_flag=True
demosaic=True
color_denoiser=False
use_cv2_demosaic=True
