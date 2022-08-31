_base_=[
        "../_base_/real_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
)

model = dict(
    type='GAPNet',
    in_ch = 8
)

eval=dict(
    flag=False,
    interval=1
)

checkpoints="checkpoints/gapnet/gapnet.pth"
