_base_=["../_base_/davis.py"]

train_pipeline = [
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Resize',resize_h=256,resize_w=256)
]

gene_meas = dict(type='GenerationBayerMeas')

train_data = dict(
    type="DavisBayerData",
    mask_path="test_datasets/mask/mask.mat",
    pipeline=train_pipeline,
    gene_meas = gene_meas,
    mask_shape = None
)
