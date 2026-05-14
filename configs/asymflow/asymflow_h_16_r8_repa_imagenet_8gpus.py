name = 'asymflow_h_16_r8_repa_imagenet_8gpus'


steps_per_epoch = 1251
warmup_epochs = 5
epochs = 600
save_interval = steps_per_epoch * 2
must_save_interval = steps_per_epoch * 40
eval_interval = steps_per_epoch * 20

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='RGBColorEncoder',
    ),
    visual_encoder=dict(
        type='PretrainedDinoV2',
        model_name_or_path='facebook/dinov2-base',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='GaussianFlow',
        denoising=dict(
            type='AsymJiT',
            patch_size=16,
            in_channels=3,
            base_rank=8,
            num_timesteps=1,
            pretrained_linear_proj='checkpoints/asymflow_subspace_pca_dit.pth',
            input_size=256,
            hidden_size=1280,
            depth=32,
            num_heads=16,
            bottleneck_dim=256,
            in_context_len=32,
            in_context_start=10,
            num_classes=1000,
            attn_dropout=0.0,
            proj_dropout=0.2,
            torch_dtype='float32',
            autocast_dtype='bfloat16',
            upcast_attention=True,
            fused_attention=True,
            compile_forward=True,
            checkpointing=True,
            sigma_min=4e-2,  # AsymFlow inference clamp
        ),
        flow_loss=dict(
            type='DiffusionMSELoss',
            data_info=dict(pred='u_t_pred', target='u_t'),
            rescale_mode='constant',
            rescale_cfg=dict(scale=2.0),  # LakonLab MSE loss has a internal 0.5 factor, so use 2.0
        ),
        repa_loss=dict(
            type='REPALoss',
            input_dim=1280,
            hidden_dim=2048,
            output_dim=768,
            cache_config=dict(cache_after_block=7),  # corresponds to encoder_depth=8 in REPA-E
            loss_weight=0.5),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=1.0,
            logit_normal_enable=True,
            logit_normal_mean=0.8,
            logit_normal_std=0.8,
        ),
        denoising_mean_mode='U',
        sigma_min=5e-2,  # training loss weight clamp (same as JiT's official 5e-2)
    ),
    diffusion_use_ema=True,
)

work_dir = f'work_dirs/{name}'
train_cfg = dict(
    prob_class=0.9,
)
test_cfg = dict()

optimizer = {
    'diffusion': dict(
        type='AdamW',
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    ),
}

data = dict(
    workers_per_gpu=8,
    train=dict(
        type='ImageNet',
        data_root='data/imagenet/train/',
        datalist_path='data/imagenet/train.txt',
        negative_label=1000,
        image_size=256),
    train_dataloader=dict(samples_per_gpu=128),
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train/',
        datalist_path='data/imagenet/train.txt',
        negative_label=1000,
        image_size=256,
        latent_size=(3, 256, 256),
        test_label_sampling='equal',  # class-balanced sampling
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    persistent_workers=True,
    prefetch_factor=32,
    multiprocessing_context='fork',
)

lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=steps_per_epoch * warmup_epochs,
    warmup_ratio=1e-8,
    by_epoch=False,
)

checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir=f'checkpoints/',
)

step = 50
guidance_scale = 2.2
guidance_interval = [0, 0.88]

prefix = f'heun_g{guidance_scale}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        interval=eval_interval,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
                num_timesteps=step,
            ),
        ),
        feed_batch_size=32,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=50000,
                reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl',
                resize=False),
        ],
        save_best_ckpt=False,
        # viz_num=256,
        # viz_dir=f'viz/{name}/{prefix}',
    )
]

total_iters = steps_per_epoch * epochs
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='fixed',
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH'),
]

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    ckpt_bf16_optim=True,
    gc_interval=100)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = f'checkpoints/{name}/latest.pth'
workflow = [('train', save_interval)]
module_wrapper = 'ddp'
cudnn_benchmark = True
mp_start_method = 'fork'
