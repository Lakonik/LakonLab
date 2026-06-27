name = 'flowdit_imagenet_test'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        model_name_or_path='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='GaussianFlow',
        denoising=dict(
            type='DiTTransformer2DModelMod',
            pretrained='huggingface://Lakonik/pi-Flow-ImageNet/teachers/flowdit_imagenet.pth',
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='bfloat16'),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    inference_only=True)

work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict()
test_cfg = dict()

data = dict(
    workers_per_gpu=4,
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train_cache/',
        datalist_path='data/imagenet/train_cache.txt',
        negative_label=1000,
        latent_size=(4, 32, 32),
        test_mode=True),
    test_dataloader=dict(samples_per_gpu=125),
    persistent_workers=True,
    prefetch_factor=64)


guidance_scales = [2.5]
guidance_starts = [600]

method_name = 'euler'
method_config = dict(
    sampler='FlowEulerODE',
)

step = 12

evaluation = []

for guidance_start in guidance_starts:
    for guidance_scale in guidance_scales:
        test_cfg_override = dict(
            guidance_scale=guidance_scale,
            guidance_interval=[0, guidance_start],
            num_timesteps=step)
        test_cfg_override.update(method_config)
        prefix = f'{method_name}_g{guidance_scale:.2f}(0-{guidance_start})_step{step}'
        evaluation.append(
            dict(
                type='GenerativeEvalHook',
                data='val',
                prefix=prefix,
                sample_kwargs=dict(
                    test_cfg_override=test_cfg_override),
                feed_batch_size=32,
                viz_num=256,
                metrics=[
                    dict(
                        type='InceptionMetrics',
                        num_images=50000,
                        reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
                ],
                viz_dir=f'viz/{name}/{prefix}',
                save_best_ckpt=False))

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
