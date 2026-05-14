# 512 samples per gpu, requires 32GB VRAM
name = 'gmrepa_k32_imagenet_piid_2step_8gpus'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        model_name_or_path='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='float16'),
    diffusion=dict(
        type='PiFlowImitation',
        policy_type='GMFlow',
        denoising=dict(
            type='GMDiTTransformer2DModelV2',
            pretrained='huggingface://Lakonik/pi-Flow-ImageNet/teachers/repa_imagenet.pth',
            num_gaussians=32,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='float32',
            autocast_dtype='bfloat16',
            checkpointing=True),
        flow_loss=dict(
            type='DiffusionMSELoss',
            data_info=dict(pred='u_t_pred', target='u_t'),
            rescale_mode='constant',
            rescale_cfg=dict(scale=3.0)),
        num_timesteps=1,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    teacher=dict(
        type='GaussianFlow',
        denoising=dict(
            type='DiTTransformer2DModelMod',
            freeze=True,
            pretrained='huggingface://Lakonik/pi-Flow-ImageNet/teachers/repa_imagenet.pth',
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='bfloat16'),
        num_timesteps=1,
        denoising_mean_mode='U')
)

save_interval = 400
must_save_interval = 800  # interval to save regardless of max_keep_ckpts
eval_interval = 400
work_dir = f'work_dirs/{name}'

train_cfg = dict(
    policy_dropout=0.05,
    num_intermediate_states=2,
    teacher_test_cfg=dict(
        guidance_scale=2.8,
        guidance_interval=[0, 0.7],
        orthogonal_guidance=False,
    ),
    nfe=2,
    total_substeps=128,
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=400,
)
test_cfg = dict(
    nfe=2,
    total_substeps=128,
)

optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=5e-5, betas=(0.9, 0.95), weight_decay=0.0,
    ),
}
data = dict(
    workers_per_gpu=4,
    train=dict(
        type='ImageNet',
        data_root='data/imagenet/train_cache/',
        datalist_path='data/imagenet/train_cache.txt',
        negative_label=1000),
    train_dataloader=dict(samples_per_gpu=512),
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train_cache/',
        datalist_path='data/imagenet/train_cache.txt',
        negative_label=1000,
        latent_size=(4, 32, 32),
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=125),
    test_dataloader=dict(samples_per_gpu=125),
    persistent_workers=True,
    prefetch_factor=256)
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.001)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

prefix = 'step2'
evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        interval=eval_interval,
        feed_batch_size=32,
        viz_num=256,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=50000,
                resize=False,
                reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
            ],
        viz_dir=f'viz/{name}/{prefix}',
        save_best_ckpt=False)]

total_iters = 24000
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
        start_iter=400,
        momentum_policy='karras',
        momentum_cfg=dict(gamma=7.0),
        priority='VERY_HIGH'),
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    gc_interval=20)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = f'checkpoints/{name}/latest.pth'  # resume by default
workflow = [('train', save_interval)]
module_wrapper = 'ddp'
cudnn_benchmark = True
mp_start_method = 'fork'
