# Difference to the official SiT:
# Official SiT uses 3-channel CFG, this is not a well-established design and is not implemented in LakonLab.
# So we use full-channel CFG with a lower guidance scale of 1.4.

# val_sde_em_g1.4_step250_fid = 2.0499460087625567
# val_sde_em_g1.4_step250_precision = 0.8225599527359009
# val_sde_em_g1.4_step250_recall = 0.5842999815940857
# val_sde_em_g1.4_step250_is = 269.79620361328125

name = 'sit_imagenet_test'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        model_name_or_path='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='GaussianFlow',
        flip_model_timesteps=True,
        denoising=dict(
            type='DiTTransformer2DModelMod',
            pretrained='huggingface://Lakonik/pi-Flow-ImageNet/teachers/sit_imagenet.safetensors',
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='bfloat16',
            compile_forward=True),
        num_timesteps=1,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    inference_only=True)

work_dir = f'work_dirs/{name}'
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
    prefetch_factor=64,
    multiprocessing_context='fork',
)

evaluation = []
step = 250
guidance_scale = 1.4

prefix = f'sde_em_g{guidance_scale}_step{step}'
evaluation.append(
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowSDE',
                sampler_kwargs=dict(
                    terminal_sigma=0.04,
                    h='sqrt(1 - sigma)',
                    solver_type='euler-maruyama',
                ),
                guidance_scale=guidance_scale,
                num_timesteps=step,
            ),
        ),
        feed_batch_size=64,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=50000,
                resize=False,
                reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
        ],
        save_best_ckpt=False,
        # viz_num=256,
        # viz_dir=f'viz/{name}/{prefix}',
    )
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
