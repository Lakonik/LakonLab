# val_sde_em_g1.8(0.0-0.7)_step250_fid = 1.3779400267236106
# val_sde_em_g1.8(0.0-0.7)_step250_precision = 0.796019971370697
# val_sde_em_g1.8(0.0-0.7)_step250_recall = 0.6351999640464783
# val_sde_em_g1.8(0.0-0.7)_step250_is = 306.3859558105469

name = 'repa_imagenet_test'

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
            pretrained='huggingface://Lakonik/pi-Flow-ImageNet/teachers/repa_imagenet.pth',
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
guidance_scale = 1.8
guidance_interval = [0.0, 0.7]

prefix = f'sde_em_g{guidance_scale}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'
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
                    use_fp64=True
                ),
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
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
