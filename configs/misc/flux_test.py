_base_ = ['../piflux/_data_test.py']

name = 'flux_test'

model = dict(
    type='LatentDiffusionTextImage',
    vae=dict(
        type='PretrainedVAEDecoder',
        model_name_or_path='black-forest-labs/FLUX.1-dev',
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='GaussianFlow',
        denoising=dict(
            type='FluxTransformer2DModel',
            patch_size=2,
            pretrained='huggingface://black-forest-labs/FLUX.1-dev/transformer/diffusion_pytorch_model.safetensors.index.json',
            in_channels=64,
            num_layers=19,
            num_single_layers=38,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            torch_dtype='bfloat16'),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=3.0,
            logit_normal_enable=False,
            use_dynamic_shifting=True,
            base_seq_len=256 * 4,
            max_seq_len=4096 * 4,
            base_logshift=0.5,
            max_logshift=1.15),
        denoising_mean_mode='U'))

work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict()
test_cfg = dict()

data = dict(
    workers_per_gpu=1,
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=True,
    prefetch_factor=2
)

methods = {
    'euler_g3.5_step50': dict(
        sampler='FlowEulerODE',
        num_timesteps=50,
        distilled_guidance_scale=3.5),
}

evaluation = []
for method_name, method_config in methods.items():
    for data_split in ['test', 'test2']:
        prefix = method_name
        num_images = None
        metrics = []
        if data_split == 'test':
            num_images = 3200
            metrics.extend([
                dict(
                    type='InceptionMetrics',
                    num_images=num_images,
                    use_kid=False,
                    use_pr=False,
                    use_is=False,
                    reference_pkl='huggingface://Lakonik/inception_feats/flux_hpsv2_inception.pkl'),
                dict(
                    type='InceptionMetrics',
                    num_images=num_images,
                    center_crop=True,
                    use_kid=False,
                    use_pr=False,
                    use_is=False,
                    prefix='patch',
                    reference_pkl='huggingface://Lakonik/inception_feats/flux_hpsv2_patch_inception.pkl'),
            ])
        elif data_split == 'test2':
            num_images = 10000
            metrics.extend([
                dict(
                    type='InceptionMetrics',
                    num_images=num_images,
                    use_kid=False,
                    use_pr=False,
                    use_is=False,
                    reference_pkl='huggingface://Lakonik/inception_feats/coco10k_inception.pkl'),
                dict(
                    type='InceptionMetrics',
                    num_images=num_images,
                    center_crop=True,
                    use_kid=False,
                    use_pr=False,
                    use_is=False,
                    prefix='patch',
                    reference_pkl='huggingface://Lakonik/inception_feats/coco10k_patch_inception.pkl'),
            ])
        metrics.extend([
            dict(
                type='HPSv2',
                num_images=num_images,
                hps_version='v2.1'),
            dict(
                type='VQAScore',
                num_images=num_images),
            dict(
                type='CLIPSimilarity',
                num_images=num_images),
        ])
        evaluation.append(
            dict(
                type='GenerativeEvalHook',
                data=data_split,
                prefix=prefix,
                sample_kwargs=dict(
                    test_cfg_override=method_config),
                metrics=metrics,
                viz_dir=f'viz/{name}/{data_split}_{prefix}',
                save_best_ckpt=False))

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
