_base_=[
    "../_base_/real_data.py",
]

real_data = dict(
    data_root="test_datasets/real_data/cr10",
    cr=10
)

model = dict(
    type='TV',
    tv_weight = 0.1,
    tv_iter_max = 5,
)
denoise_method="GAP"
checkpoints=None
# sigma_list = [50/255, 25/255, 12/255,6/255]
sigma_list = [100/255,75/255,50/255, 25/255]
iter_list = [10, 10, 10, 10] 

show_flag=True
