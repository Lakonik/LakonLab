_base_ = ['./_ddp_train.py', './_data_trainval.py']

name = 'dxqwen_n10_piid_4step_32gpus'

model = dict(
    type='LatentDiffusionTextImage',
    vae=dict(
        type='PretrainedVAEQwenImage',
        model_name_or_path='Qwen/Qwen-Image',
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
            type='DXQwenImageTransformer2DModel',
            patch_size=2,
            freeze=True,
            freeze_exclude=[
                'proj_out',
                'norm_out',
                'lora'],
            pretrained='huggingface://Qwen/Qwen-Image/transformer/diffusion_pytorch_model.safetensors.index.json',
            n_grid=10,
            in_channels=64,
            out_channels=64,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            axes_dims_rope=(16, 56, 56),
            torch_dtype='bfloat16',
            checkpointing=True,
            use_lora=True,
            lora_target_modules=[
                'img_mlp.net.0.proj',
                'img_mlp.net.2',
                'timestep_embedder.linear_1',
                'timestep_embedder.linear_2'
            ] + [
                f'transformer_blocks.{i}.txt_mlp.net.0.proj' for i in range(59)
            ] + [
                f'transformer_blocks.{i}.txt_mlp.net.2' for i in range(59)
            ],
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
            type='QwenImageTransformer2DModel',
            patch_size=2,
            freeze=True,
            pretrained='huggingface://Qwen/Qwen-Image/transformer/diffusion_pytorch_model.safetensors.index.json',
            in_channels=64,
            out_channels=64,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            axes_dims_rope=(16, 56, 56),
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
    window_substeps=3,
    num_intermediate_states=2,
    teacher_test_cfg=dict(guidance_scale=4.0),
    nfe=4,
    final_step_size_scale=0.5,
    total_substeps=128,
)
test_cfg = dict(
    nfe=4,
    final_step_size_scale=0.5,
    total_substeps=128,
)

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=8),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=True,
    prefetch_factor=4
)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

evaluation = []
for data_split in ['val']:
    prefix = 'step4'
    evaluation.append(
        dict(
            type='GenerativeEvalHook',
            data=data_split,
            prefix=prefix,
            interval=eval_interval,
            viz_dir=f'viz/{name}/{data_split}_{prefix}',
            metric_cpu_offload=True,
            save_best_ckpt=False))

total_iters = 9000
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
