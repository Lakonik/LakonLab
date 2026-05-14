_base_ = ['./_ddp_train.py', './_data_trainval.py']

name = 'gmflux_k8_datafree_piid_4step_32gpus'

model = dict(
    type='LatentDiffusionTextImage',
    vae=dict(
        type='PretrainedVAEDecoder',
        model_name_or_path='black-forest-labs/FLUX.1-dev',
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='PiFlowImitationDataFree',
        policy_type='GMFlow',
        denoising=dict(
            type='GMFluxTransformer2DModel',
            patch_size=2,
            freeze=True,
            freeze_exclude=[
                'proj_out_means',
                'proj_out_logweights',
                'proj_out_logstds',
                'norm_out',
                'lora'],
            pretrained='huggingface://black-forest-labs/FLUX.1-dev/transformer/diffusion_pytorch_model.safetensors.index.json',
            num_gaussians=8,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            logweights_channels=4,
            in_channels=64,
            num_layers=19,
            num_single_layers=38,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            torch_dtype='bfloat16',
            checkpointing=True,
            use_lora=True,
            lora_target_modules=[
                'proj_mlp',
                'proj_out',
                'ff.net.0.proj',
                'ff.net.2',
                'ff_context.net.0.proj',
                'ff_context.net.2',
                'timestep_embedder.linear_1',
                'timestep_embedder.linear_2'],
            lora_dropout=0.05,
            lora_rank=256),
        flow_loss=dict(
            type='DiffusionMSELoss',
            data_info=dict(pred='u_t_pred', target='u_t'),
            rescale_mode='constant',
            rescale_cfg=dict(scale=30.0)),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=3.2,
            logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    teacher=dict(
        type='GaussianFlow',
        denoising=dict(
            type='FluxTransformer2DModel',
            patch_size=2,
            freeze=True,
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
        denoising_mean_mode='U'),
    tie_teacher=True,
)

save_interval = 100
must_save_interval = 200  # interval to save regardless of max_keep_ckpts
eval_interval = 100
work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict(
    num_decay_iters=2000,
    window_substeps=3,
    policy_dropout=0.1,
    num_intermediate_states=4,
    distilled_guidance_scale=3.5,
    teacher_test_cfg=dict(distilled_guidance_scale=3.5),
    nfe=4,
    final_step_size_scale=0.5,
    total_substeps=128,
)
test_cfg = dict(
    distilled_guidance_scale=3.5,
    nfe=4,
    final_step_size_scale=0.5,
    total_substeps=128,
)

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2),
    test=dict(
        type='ImagePrompt',
        data_root='data/balancia_morgan_flux_embeds/',
        cache_dir='',
        cache_datalist_path='data/balancia_morgan_cache.json',
        end_ind=128,
        latent_size=(16, 96, 170),
        repeat=2,
        test_mode=True,
    ),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=True,
    num_threads=1,
    prefetch_factor=2
)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

evaluation = []
for data_split in ['val2', 'test']:
    for temperature in [0.3]:
        prefix = f'step4_temp{temperature}'
        evaluation.append(
            dict(
                type='GenerativeEvalHook',
                data=data_split,
                prefix=prefix,
                sample_kwargs=dict(
                    test_cfg_override=dict(
                        temperature=temperature,
                    )),
                interval=eval_interval,
                metrics=[dict(
                    type='HPSv2',
                    hps_version='v2.1')],
                viz_dir=f'viz/{name}/{data_split}_{prefix}',
                metric_cpu_offload=True,
                save_best_ckpt=False))

total_iters = 3000
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

load_from = None
resume_from = f'checkpoints/{name}/latest.pth'  # resume by default
workflow = [('train', save_interval)]
