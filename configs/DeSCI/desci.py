_base_=["../_base_/six_gray_sim_data.py"]

test_data = dict(
    mask_path="test_datasets/mask/ffdnet_mask.mat"
)

model = dict(
    type='DeSCI',
    image_size=256,
    local_win_size=7,
    patch_size=9,
    local_win_step=1,
    patch_step=3,
    sim_num=20,
)

denoise_method="ADMM"
checkpoints=None
sigma_list = [100/255,50/255, 25/255, 12/255,6/255]
iter_list = [5,5,5,5,5] 
show_flag=True
