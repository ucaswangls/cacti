_base_=["../_base_/six_gray_sim_data.py"]

test_data = dict(
    mask_path="test_datasets/mask/ffdnet_mask.mat"
)

model = dict(
    type='FastDVDnet',
    num_input_frames=5, 
    num_color_channels=1
)

denoise_method="ADMM"
checkpoints="checkpoints/fastdvd/fastdvd_gray.pth"

sigma_list = [50/255, 25/255, 12/255]
iter_list = [30, 30, 30] 
show_flag=True
