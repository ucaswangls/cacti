_base_=["../_base_/matlab_bayer.py"]

model = dict(
    type='TV',
    tv_weight = 0.1,
    tv_iter_max = 5,
)
denoise_method="GAP"
checkpoints=None
sigma_list = [50/255, 25/255, 12/255,6/255]
iter_list = [10, 10, 10, 10] 
show_flag=True