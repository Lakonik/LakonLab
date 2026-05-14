name = 'asymflow_h_16_r8_repa_imagenet_test'


model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='RGBColorEncoder',
    ),
    diffusion=dict(
        type='GaussianFlow',
        denoising=dict(
            type='AsymJiT',
            patch_size=16,
            in_channels=3,
            base_rank=8,
            num_timesteps=1,
            pretrained='huggingface://Lakonik/AsymFlow-ImageNet/asymflow_h_16_r8_repa_imagenet.safetensors',
            input_size=256,
            hidden_size=1280,
            depth=32,
            num_heads=16,
            bottleneck_dim=256,
            in_context_len=32,
            in_context_start=10,
            num_classes=1000,
            torch_dtype='bfloat16',
            upcast_attention=True,
            fused_attention=False,
            compile_forward=True,
            sigma_min=4e-2,  # AsymFlow inference clamp
        ),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=1.0,
            logit_normal_enable=True,
            logit_normal_mean=0.8,
            logit_normal_std=0.8,
        ),
        denoising_mean_mode='U',
    ),
    diffusion_use_ema=True,
    inference_only=True,
)

work_dir = f'work_dirs/{name}'
train_cfg = dict()
test_cfg = dict()

data = dict(
    workers_per_gpu=8,
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
    test_dataloader=dict(samples_per_gpu=64),
    persistent_workers=True,
    prefetch_factor=16,
    multiprocessing_context='fork',
)

step = 50
guidance_scale = 2.2
guidance_interval = [0, 0.88]

prefix = f'heun_g{guidance_scale}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'

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
