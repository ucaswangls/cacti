_base_=[
        "../_base_/six_gray_sim_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

mask_path="test_datasets/mask/dun3dunet_mask.mat"
train_data = dict(mask_path=mask_path)
test_data = dict(mask_path=mask_path)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

model = dict(
    type='HQSNet',
    layer_num=10,
    n_channels=1
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints="checkpoints/dun3dunet/dun3dunet.pth"