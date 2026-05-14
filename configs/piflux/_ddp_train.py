# ~65GB VRAM

model = dict(
    diffusion=dict(
        denoising=dict(
            freeze_exclude_autocast_dtype='bfloat16')))
train_cfg = dict(
    # grad_accum_batch_size=1,  # uncomment this line to reduce VRAM usage to ~45GB
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=100,
)
optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
    ),
}
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001)
runner = dict(
    type='DynamicIterBasedRunner',
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    gc_interval=20)
dist_params = dict(backend='nccl')
log_level = 'INFO'
module_wrapper = 'ddp'
cudnn_benchmark = True
mp_start_method = 'fork'
