# Difference to official JiT:
# - We use ADM evaluation, which is the ImageNet standard.
# - We use sigma_min=4e-2 which works better than JiT's official 5e-2.

name = 'jit_h_16_imagenet_test'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='RGBColorEncoder',
    ),
    diffusion=dict(
        type='GaussianFlow',
        flip_model_timesteps=True,
        denoising=dict(
            type='JiT',
            input_size=256,
            patch_size=16,
            in_channels=3,
            hidden_size=1280,
            depth=32,
            num_heads=16,
            bottleneck_dim=256,
            in_context_len=32,
            in_context_start=10,
            num_classes=1000,
            pretrained='huggingface://Lakonik/pi-Flow-ImageNet/teachers/jit_h_16_imagenet.safetensors',
            torch_dtype='bfloat16',
            upcast_attention=True,
            fused_attention=False,
            compile_forward=True,
        ),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=1.0,
            logit_normal_enable=True,
            logit_normal_mean=0.8,
            logit_normal_std=0.8,
        ),
        denoising_mean_mode='X0',
        sigma_min=4e-2,  # works better than JiT's official 5e-2
    ),
    diffusion_use_ema=True,
    inference_only=True,
)

work_dir = f'work_dirs/{name}'
train_cfg = dict()
test_cfg = dict()

data = dict(
    workers_per_gpu=4,
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train/',
        datalist_path='data/imagenet/train.txt',
        negative_label=1000,
        image_size=256,
        latent_size=(3, 256, 256),
        test_label_repeat=50,  # JiT-style class sampling
        test_label_sampling='equal',  # class-balanced sampling
        test_mode=True),
    test_dataloader=dict(samples_per_gpu=125),
    persistent_workers=True,
    prefetch_factor=64,
    multiprocessing_context='fork',
)

step = 50
guidance_scale = 2.2
guidance_interval = [0.0, 0.9]

prefix = f'heun_g{guidance_scale}_step{step}'

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
                num_timesteps=step,
            ),
        ),
        feed_batch_size=32,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=50000,
                is_shuffle=True,  # True when using test_label_repeat
                reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl',
                resize=False),
        ],
        save_best_ckpt=False,
        # viz_num=256,
        # viz_dir=f'viz/{name}/{prefix}',
    )
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
