import torch
from diffusers import ComponentsManager
from diffusers.utils import encode_video, load_image
from lakonlab.models.architectures.diffusers.attention import default_attention_backend, set_attention_backends
from lakonlab.models.architectures.diffusers.minimax_h3 import MiniMaxH3SolAttnProcessor
from lakonlab.models.architectures.diffusers.pretrained import AutoencoderKLMiniMaxH3
from lakonlab.models.diffusions.schedulers import FlowEulerODEScheduler
from lakonlab.ops import quantize_frozen_linears
from lakonlab.pipelines.pipeline_minimax_h3 import MiniMaxH3BlocksMod
from lakonlab.pipelines.utils import load_lakonlab_adapter

NUM_STEPS = 4  # 2 or 4
ADAPTER_NAME = f'pi_h3_{NUM_STEPS}step_adapter'
MAX_RAW_T = {2: 0.93, 4: 0.96}[NUM_STEPS]

# load base pipeline with batched VAE decoding
manager = ComponentsManager()
pipe = MiniMaxH3BlocksMod().init_pipeline('multimodalart/MiniMax-H3-Pruned', components_manager=manager)
vae_spec = pipe.get_component_spec('vae')
vae_spec.type_hint = AutoencoderKLMiniMaxH3
pipe.update_components(vae=vae_spec.load(dtype=torch.float16))
pipe.load_components(
    names=['text_encoder', 'tokenizer', 'processor', 'audio_vae', 'transformer'],
    dtype=torch.bfloat16, trust_remote_code=True)
pipe.vae.decode_tile_batch_size = 28
# load adapter
adapter_name = load_lakonlab_adapter(
    pipe, 'Lakonik/pi-H3', subfolder=ADAPTER_NAME)

# FP8 W8A8
pipe.transformer.requires_grad_(False)
quantize_frozen_linears(
    pipe.transformer, exclude_keys=['lora', 'self.norm_out', 'self.proj_out', 'self.audio_proj_out'])
pipe.text_encoder.requires_grad_(False)
quantize_frozen_linears(pipe.text_encoder, exclude_keys=['lm_head'])
torch.cuda.empty_cache()
# the fastest installed attention kernels for the GPU
set_attention_backends(pipe.transformer, backend=default_attention_backend('dense'))
# Sol sparse attention after the first 2 blocks (requires lakonlab[sol]; remove this loop for dense attention)
for block in pipe.transformer.transformer_blocks[2:]:
    block.attn.set_processor(MiniMaxH3SolAttnProcessor())
    block.attn.set_attention_backend(default_attention_backend('short'))  # for the short prefix query rows

pipe.update_components(
    scheduler=FlowEulerODEScheduler(
        num_train_timesteps=1, shift=12.0, max_raw_t=MAX_RAW_T, final_step_size_scale=0.6),
    audio_scheduler=FlowEulerODEScheduler(
        num_train_timesteps=1, shift=3.0, max_raw_t=MAX_RAW_T, final_step_size_scale=0.6))

# manager.enable_auto_cpu_offload(device='cuda', memory_reserve_margin='20GB')

# Text-to-video+audio example
prompt = 'A red fox trotting through a snowy pine forest, snow crunching underfoot, birds chirping in the distance'
out = pipe(
    prompt=prompt,
    height=768,
    width=1344,
    num_frames=243,  # 10 s at H3's fixed 24 fps, rounded up to 17n + 5
    num_inference_steps=NUM_STEPS,
    generator=torch.Generator().manual_seed(42),
    output=['videos', 'audio', 'sampling_rate'])
encode_video(
    out['videos'][0], fps=24, output_path='minimax_h3_t2va.mp4',
    audio=out['audio'][0], audio_sample_rate=out['sampling_rate'])

# Image-to-video+audio example
prompt = 'The cat looks around curiously and meows softly'
cat_image = load_image('https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png')
out = pipe(
    prompt=prompt,
    image=cat_image,
    num_frames=243,
    num_inference_steps=NUM_STEPS,
    generator=torch.Generator().manual_seed(42),
    output=['videos', 'audio', 'sampling_rate'])
encode_video(
    out['videos'][0], fps=24, output_path='minimax_h3_i2va.mp4',
    audio=out['audio'][0], audio_sample_rate=out['sampling_rate'])
