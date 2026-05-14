# ~40GB VRAM on 2 GPUS, ~20GB VRAM on 8 GPUs

train_cfg = dict(
    grad_accum_batch_size=1,
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=100,
)
optimizer = {
    'diffusion': dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
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
    ckpt_bf16_optim=True,
    gc_interval=20)
dist_params = dict(backend='nccl')
log_level = 'INFO'
module_wrapper = 'fsdp'
fsdp_kwargs = dict(
    wrap_frozen_modules=True,  # shard all modules
    ignore_frozen_parameters=False,  # shard all parameters
    fsdp_modules=['diffusers.models.transformers.transformer_qwenimage.QwenImageTransformerBlock'],
    exclude_keys=['vae'],
    tie_key_mappings=['teacher->diffusion', 'teacher->diffusion_ema'],
)
cudnn_benchmark = True
mp_start_method = 'fork'
