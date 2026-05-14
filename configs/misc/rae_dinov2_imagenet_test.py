# val_euler_g1_step50_fid = 1.4973859142647876
# val_euler_g1_step50_precision = 0.7874799966812134
# val_euler_g1_step50_recall = 0.6385999917984009
# val_euler_g1_step50_is = 254.70004272460938

name = 'rae_dinov2_imagenet_test'

latent_size = (768, 16, 16)
time_dist_shift_base = 4096
time_dist_shift_dim = latent_size[0] * latent_size[1] * latent_size[2]
time_dist_shift = (time_dist_shift_dim / time_dist_shift_base) ** 0.5

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedRAE',
        model_name_or_path='nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08',
        freeze=True,
        torch_dtype='float32',
    ),
    diffusion=dict(
        type='GaussianFlow',
        denoising=dict(
            type='LightningDDT',
            pretrained='huggingface://nyu-visionx/RAE-collections/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt',
            input_size=16,
            patch_size=1,
            in_channels=768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads=[16, 16],
            mlp_ratio=4.0,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed=True,
            torch_dtype='float32',
            compile_forward=True,
        ),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=time_dist_shift,
            logit_normal_enable=False,
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
    workers_per_gpu=4,
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train/',
        datalist_path='data/imagenet/train.txt',
        negative_label=1000,
        latent_size=latent_size,
        test_label_sampling='equal',
        test_mode=True),
    test_dataloader=dict(samples_per_gpu=125),
    persistent_workers=True,
    prefetch_factor=64,
    multiprocessing_context='fork',
)

evaluation = []
step = 50
guidance_scale = 1.0

prefix = f'euler_g{guidance_scale}_step{step}'
evaluation.append(
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowEulerODE',
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
        viz_num=256,
        viz_dir=f'viz/{name}/{prefix}',
    )
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
