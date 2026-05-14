name = 'asymflux2_klein_test'

HPSV3_BENCHMARK_FILES = [
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Characters.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Arts.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Design.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Architecture.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Animals.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Natural Scenery.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Transportation.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Products.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Others.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Plants.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Food.json',
    'hf://datasets/MizzenAI/HPDv3/benchmark/benchmark_Science.json',
]
HPSV3_BENCHMARK_DATASET_KWARGS = dict(
    path='json',
    data_files=HPSV3_BENCHMARK_FILES,
    split='train')

model = dict(
    type='LatentDiffusionTextImage',
    text_encoder=dict(
        type='PretrainedFlux2KleinTextEncoder'
    ),
    vae=dict(
        type='OklabColorEncoder',
        use_affine_norm=True,
        mean=(0.56, 0.0, 0.01),
        std=0.16),
    diffusion=dict(
        type='AsymFlowVR',
        latent_patch_size=2,
        denoising=dict(
            type='AsymFlux2Transformer2DModel',
            freeze=True,
            pretrained='huggingface://black-forest-labs/FLUX.2-klein-base-9B/transformer/diffusion_pytorch_model.safetensors.index.json',
            pretrained_adapter='huggingface://Lakonik/AsymFLUX.2-klein-9B/diffusion_pytorch_model.safetensors',
            patch_size=16,
            in_channels=3,
            base_rank=128,
            num_layers=8,
            num_single_layers=24,
            attention_head_dim=128,
            num_attention_heads=32,
            joint_attention_dim=12288,
            timestep_guidance_channels=256,
            guidance_embeds=False,
            torch_dtype='bfloat16'),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=17.0,
            logit_normal_enable=True),
        denoising_mean_mode='U',
    ),
    diffusion_use_ema=True,
    inference_only=True)

work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict()
test_cfg = dict(
    clamp_denoised=True,
)

data = dict(
    workers_per_gpu=4,
    test2=dict(
        type='ImagePrompt',
        data_root='data/dpgbench-prompts/',
        prompt_dataset_kwargs=dict(
            path='json',
            data_files='data/dpgbench-prompts/prompts.jsonl',
            split='train'),
        negative_prompt_embeds_path='data/flux2_klein_empty_prompt_embeds.pth',
        vae_scale_factor=1,
        latent_size=(3, 1024, 1024),
        repeat=4,
        test_mode=True),
    test3=dict(
        type='ImagePrompt',
        data_root='data/geneval-prompts/',
        prompt_dataset_kwargs=dict(
            path='json',
            data_files='data/geneval-prompts/prompts.jsonl',
            split='train'),
        negative_prompt_embeds_path='data/flux2_klein_empty_prompt_embeds.pth',
        vae_scale_factor=1,
        latent_size=(3, 1024, 1024),
        repeat=4,
        test_mode=True),
    test4=dict(
        type='ImagePrompt',
        data_root='data/',
        prompt_dataset_kwargs=HPSV3_BENCHMARK_DATASET_KWARGS,
        prompt_key='caption',
        negative_prompt_embeds_path='data/flux2_klein_empty_prompt_embeds.pth',
        vae_scale_factor=1,
        latent_size=(3, 1024, 1024),
        test_mode=True),
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=True,
    prefetch_factor=2,
    multiprocessing_context='fork',
)

methods = {
    'unipc_g4o1_step32': dict(
        sampler='FlowAdapter',
        sampler_kwargs=dict(
            base_scheduler='UniPCMultistep'),
        num_timesteps=32,
        guidance_scale=4.0,
        orthogonal_guidance=1.0),
}

evaluation = []
for method_name, method_config in methods.items():
    for data_split in ['test2', 'test3', 'test4']:
        prefix = method_name
        metrics = []
        if data_split == 'test2':
            metrics.append(dict(
                type='DPGBenchExport',
                prompt_path='data/dpgbench-prompts/prompts.jsonl',
                img_root_dir=f'viz/{name}/{data_split}_{prefix}/dpgbench'))
        elif data_split == 'test3':
            metrics.append(dict(
                type='GenEvalExport',
                prompt_path='data/geneval-prompts/prompts.jsonl',
                metadata_path='data/geneval-prompts/evaluation_metadata.jsonl',
                img_root_dir=f'viz/{name}/{data_split}_{prefix}/geneval'))
        elif data_split == 'test4':
            metrics.append(dict(
                type='HPSv3BenchmarkExport',
                benchmark_dataset_kwargs=HPSV3_BENCHMARK_DATASET_KWARGS,
                img_root_dir=f'viz/{name}/{data_split}_{prefix}/hpsv3'))
        evaluation.append(
            dict(
                type='GenerativeEvalHook',
                data=data_split,
                prefix=prefix,
                sample_kwargs=dict(
                    test_cfg_override=method_config),
                metrics=metrics,
                # viz_dir=f'viz/{name}/{data_split}_{prefix}',
                save_best_ckpt=False))

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
