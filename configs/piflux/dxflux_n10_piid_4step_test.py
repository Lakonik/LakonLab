_base_ = ['./_data_test.py']

name = 'dxflux_n10_piid_4step_test'

model = dict(
    type='LatentDiffusionTextImage',
    vae=dict(
        type='PretrainedVAE',
        model_name_or_path='black-forest-labs/FLUX.1-dev',
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='PiFlowImitation',
        policy_type='DX',
        policy_kwargs=dict(
            segment_size=1 / 3.5,  # 1 / (nfe - 1 + final_step_size_scale)
            shift=3.2),
        denoising=dict(
            type='DXFluxTransformer2DModel',
            patch_size=2,
            pretrained='huggingface://black-forest-labs/FLUX.1-dev/transformer/diffusion_pytorch_model.safetensors.index.json',
            pretrained_adapter='huggingface://Lakonik/pi-FLUX.1/dxflux_n10_piid_4step/diffusion_pytorch_model.safetensors',
            n_grid=10,
            in_channels=64,
            out_channels=64,
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
            shift=3.2,
            logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    inference_only=True
)

work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict()
test_cfg = dict(
    distilled_guidance_scale=3.5,
    nfe=4,
    final_step_size_scale=0.5,
    total_substeps=128,
)

data = dict(
    workers_per_gpu=1,
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=True,
    prefetch_factor=2
)

evaluation = []
for data_split in ['test']:
    prefix = 'step4'
    num_images = None
    metrics = []
    if data_split == 'test':
        num_images = 3200
        metrics.extend([
            dict(
                type='InceptionMetrics',
                num_images=num_images,
                resize=True,
                use_kid=False,
                use_pr=False,
                use_is=False,
                reference_pkl='huggingface://Lakonik/inception_feats/flux_hpsv2_inception.pkl'),
            dict(
                type='InceptionMetrics',
                num_images=num_images,
                center_crop=True,
                resize=False,
                use_kid=False,
                use_pr=False,
                use_is=False,
                prefix='patch',
                reference_pkl='huggingface://Lakonik/inception_feats/flux_hpsv2_patch_inception.pkl'),
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
            metrics=metrics,
            viz_dir=f'viz/{name}/{data_split}_{prefix}',
            save_best_ckpt=False))

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
