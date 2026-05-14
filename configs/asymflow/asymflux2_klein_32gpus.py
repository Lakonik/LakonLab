name = 'asymflux2_klein_32gpus'


model = dict(
    type='LatentDiffusionTextImage',
    vae=dict(
        type='OklabColorEncoder',
        use_affine_norm=True,
        mean=(0.56, 0.0, 0.01),
        std=0.16),
    train_cached_latents_as_latents_2=True,
    diffusion=dict(
        type='AsymFlowVR',
        latent_patch_size=2,
        denoising=dict(
            type='AsymFlux2Transformer2DModel',
            freeze=True,
            freeze_exclude=[
                'x_embedder',
                'proj_out',
                'norm_out',
                'lora'],
            freeze_exclude_autocast_dtype='bfloat16',
            pretrained='huggingface://black-forest-labs/FLUX.2-klein-base-9B/transformer/diffusion_pytorch_model.safetensors.index.json',
            pretrained_linear_proj='checkpoints/asymflow_subspace_procrustes.pth',
            patch_size=16,
            in_channels=3,
            base_rank=128,
            num_layers=8,
            num_single_layers=24,
            attention_head_dim=128,
            num_attention_heads=32,
            joint_attention_dim=12288,
            timestep_guidance_channels=256,
            guidance_embeds=False,
            torch_dtype='bfloat16',
            checkpointing=True,
            use_lora=True,
            lora_target_modules=[
                'ff.linear_in',
                'ff.linear_out',
                'ff_context.linear_in',
                'ff_context.linear_out',
                'timestep_embedder.linear_1',
                'timestep_embedder.linear_2'
            ] + [
                f'single_transformer_blocks.{i}.attn.to_out' for i in range(24)
            ],
            lora_dropout=0.05,
            lora_rank=256),
        mse_loss_weight=10.0,  # LakonLab MSE loss has a internal 0.5 factor, so effective weight is 5.0 (rescaled)
        perceptual_loss=dict(
            type='LPIPSLoss',
            spatial=True,
            loss_weight=1.0),  # Relative weight is 0.2, so 5.0 * 0.2 = 1.0
        loss_shift=0.3,
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=17.0,
            logit_normal_enable=True),
        denoising_mean_mode='U',
        sigma_min=5e-2,
    ),
    diffusion_use_ema=True,
    teacher=dict(
        type='GaussianFlow',
        denoising=dict(
            type='AsymFlux2Transformer2DModel',
            freeze=True,
            pretrained='huggingface://black-forest-labs/FLUX.2-klein-base-9B/transformer/diffusion_pytorch_model.safetensors.index.json',
            pretrained_linear_proj='checkpoints/asymflow_subspace_procrustes.pth',
            patch_size=16,
            in_channels=3,
            base_rank=128,
            num_layers=8,
            num_single_layers=24,
            attention_head_dim=128,
            num_attention_heads=32,
            joint_attention_dim=12288,
            timestep_guidance_channels=256,
            guidance_embeds=False,
            torch_dtype='bfloat16'),
        num_timesteps=1,
        denoising_mean_mode='U',),
    tie_teacher=True,
)

save_interval = 250
must_save_interval = 1000  # interval to save regardless of max_keep_ckpts
eval_interval = 250
work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict(
    teacher_test_cfg=dict(
        guidance_scale=1.0),
    diffusion_grad_clip=200.0,
    diffusion_grad_clip_begin_iter=100,
)
test_cfg = dict(
    clamp_denoised=True,
)

optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
        paramwise_cfg=dict(custom_keys={
            'proj_out': dict(lr_mult=10.0),
        })
    ),
}
data = dict(
    workers_per_gpu=8,
    train=dict(
        type='ImagePrompt',
        data_root='data/laion-3m/',
        image_dir='images',
        image_datalist_path='data/laion-3m/images.jsonl',
        cache_dir='preproc_flux2_klein',
        cache_datalist_path='data/laion-3m/preproc_flux2_klein.jsonl.gz',
        ignore_cached_latents=False,  # load both images and latents
        negative_prompt_embeds_path='data/flux2_klein_empty_prompt_embeds.pth',
        image_scale_factor=1.0,
        image_scale_method='lanczos',
        latent_patch_size=2,  # latent patch size = 2
        vae_scale_factor=8,  # pixel patch size = 2 * 8 = 16
        bucketize=True,
        end_ind=-128),
    train_dataloader=dict(samples_per_gpu=8),
    val=dict(
        type='ImagePrompt',
        data_root='data/laion-3m/',
        image_dir='images',
        image_datalist_path='data/laion-3m/images.jsonl',
        cache_dir='preproc_flux2_klein',
        cache_datalist_path='data/laion-3m/preproc_flux2_klein.jsonl.gz',
        ignore_cached_latents=True,  # latents are disgarded
        negative_prompt_embeds_path='data/flux2_klein_empty_prompt_embeds.pth',
        image_scale_factor=1.0,
        latent_patch_size=16,  # this means pixel patch size
        vae_scale_factor=1,
        latent_size=(3, 768, 1344),
        start_ind=-128,
        repeat=2,
        test_mode=True,
    ),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=True,
    prefetch_factor=2,
    multiprocessing_context='fork',
)
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

step = 32
guidance_scale = 4.0

evaluation = []
for data_split in ['val']:
    prefix = f'unipc_g{guidance_scale:.2f}o1_step{step}'
    evaluation.append(
        dict(
            type='GenerativeEvalHook',
            data=data_split,
            prefix=prefix,
            sample_kwargs=dict(
                test_cfg_override=dict(
                    sampler='FlowAdapter',
                    sampler_kwargs=dict(
                        base_scheduler='UniPCMultistep'),
                    num_timesteps=step,
                    guidance_scale=guidance_scale,
                    orthogonal_guidance=1.0
                )),
            interval=eval_interval,
            metrics=[
                dict(type='HPSv2', hps_version='v2.1'),
                dict(type='HPSv3'),
                dict(type='ColorStats')
            ],
            viz_dir=f'viz/{name}/{data_split}_{prefix}',
            metric_cpu_offload=True,
            save_best_ckpt=False))

total_iters = 20000
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=100,
        momentum_policy='karras',
        momentum_cfg=dict(gamma=7.0),
        priority='VERY_HIGH'),
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    gc_interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = f'checkpoints/{name}/latest.pth'  # resume by default
workflow = [('train', save_interval)]

module_wrapper = 'ddp'

cudnn_benchmark = True
mp_start_method = 'fork'
