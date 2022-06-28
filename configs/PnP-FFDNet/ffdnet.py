_base_=["../_base_/six_gray_sim_data.py"]

test_data = dict(
    mask_path="test_datasets/mask/ffdnet_mask.mat"
)

model = dict(
    type='FFDNet',
    num_input_channels=1
)

denoise_method="GAP"
checkpoints="checkpoints/ffdnet/ffdnet_gray.pth"

sigma_list = [50/255, 25/255, 12/255,6/255]
iter_list = [10, 10, 10, 10] 
show_flag=True
