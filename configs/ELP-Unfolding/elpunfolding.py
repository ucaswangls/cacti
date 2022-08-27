_base_=[
        "../_base_/six_gray_sim_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]
test_data = dict(
    mask_path = "test_datasets/mask/elpunfolding_mask.mat"
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
model = dict(
    type='ELPUnfolding',
    in_ch = 8,
    pres_ch = 8,
    init_channels=512,
    iter_number=8,
    priors = 6,
)
eval=dict(
    flag=True,
    interval=1
)

checkpoints="checkpoints/elpunfolding/elpunfolding.pth"