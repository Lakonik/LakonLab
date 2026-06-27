name = 'gmdit_k8_imagenet_test'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        model_name_or_path='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='GMFlow',
        denoising=dict(
            type='GMDiTTransformer2DModel',
            pretrained='huggingface://Lakonik/gmflow_imagenet_k8_ema/transformer/diffusion_pytorch_model.bf16.safetensors',
            num_gaussians=8,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='bfloat16'),
        spectrum_net=dict(
            type='SpectrumMLP',
            base_size=(4, 32, 32),
            layers=[64, 8],
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

guidance_scales = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.13, 0.16, 0.19, 0.23, 0.27, 0.33, 0.39, 0.47, 0.55, 0.65, 0.75]

methods = dict(
    gmode2=dict(
        output_mode='mean',
        sampler='FlowEulerODE',
        order=2),
    gmsde2=dict(
        output_mode='sample',
        sampler='FlowSDE',
        order=2)
)

evaluation = []

for step, substep in [(8, 16), (32, 4)]:
    for method_name, method_config in methods.items():
        for guidance_scale in guidance_scales:
            test_cfg_override = dict(
                orthogonal_guidance=1.0,
                guidance_scale=guidance_scale,
                num_timesteps=step)
            if 'gmode' in method_name:
                test_cfg_override.update(num_substeps=substep)
            test_cfg_override.update(method_config)
            prefix = f'{method_name}_g{guidance_scale:.2f}_step{step}'
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
