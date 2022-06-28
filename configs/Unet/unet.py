_base_=[
        "../_base_/six_gray_sim_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

model = dict(
    type='Unet',
    in_ch=8,
    out_ch=64
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints="checkpoints/unet/unet.pth"