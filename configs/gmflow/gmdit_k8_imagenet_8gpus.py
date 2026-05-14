# 512 samples per gpu, requires 40GB VRAM
name = 'gmdit_k8_imagenet_8gpus'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        model_name_or_path='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='float16'),
    diffusion=dict(
        type='GMFlow',
        denoising=dict(
            type='GMDiTTransformer2DModel',
            num_gaussians=8,
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
        spectrum_net=dict(
            type='SpectrumMLP',
            base_size=(4, 32, 32),
            layers=[64, 8],
            torch_dtype='float32'),
        flow_loss=dict(
            type='GMFlowNLLLoss',
            log_cfgs=dict(type='quartile', prefix_name='loss_trans', total_timesteps=1000),
            data_info=dict(
                pred_means='means',
                target='x_t_low',
                pred_logstds='logstds',
                pred_logweights='logweights'),
            rescale_mode='constant',
            rescale_cfg=dict(scale=8.0)),
        spectral_loss_weight=4.0,
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
)

save_interval = 1000
must_save_interval = 40000  # interval to save regardless of max_keep_ckpts
eval_interval = 20000
work_dir = f'work_dirs/{name}'

train_cfg = dict(
    trans_ratio=0.5,
    prob_class=0.9,
    diffusion_grad_clip=10.0,
    diffusion_grad_clip_begin_iter=1000,
)
test_cfg = dict()

optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=2e-4, betas=(0.9, 0.95), weight_decay=0.05,
        paramwise_cfg=dict(custom_keys={
            'bias': dict(decay_mult=0.0),
        })
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
    warmup_iters=1000,
    warmup_ratio=0.001)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

step = 16
substep = 8
guidance_scale = 0.04

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=f'gmode2_g{guidance_scale:.2f}_step{step}',
        sample_kwargs=dict(
            test_cfg_override=dict(
                output_mode='mean',
                sampler='FlowEulerODE',
                guidance_scale=guidance_scale,
                order=2,
                num_timesteps=step,
                num_substeps=substep,
            )),
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
        viz_dir=f'viz/{name}/gmode2_g{guidance_scale:.2f}_step{step}',
        save_best_ckpt=False)]

total_iters = 200000
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
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=30000, ema_rampup=0.05, batch_size=4096, eps=1e-8),
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
