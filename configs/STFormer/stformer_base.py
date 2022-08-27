_base_=[
        "../_base_/six_gray_sim_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

test_data = dict(
    mask_path="test_datasets/mask/random_mask.mat"
)

model = dict(
    type='STFormer',
    color_channels=1,
    units=4,
    dim=64
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints="checkpoints/stformer/stformer_base.pth"