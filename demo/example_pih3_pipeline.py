import torch
from diffusers import ComponentsManager
from diffusers.utils import encode_video, load_image
from lakonlab.models.architectures import RGBColorEncoder
from lakonlab.models.architectures.diffusers.attention import default_attention_backend, set_attention_backends
from lakonlab.models.architectures.diffusers.minimax_h3 import MiniMaxH3SolAttnProcessor
from lakonlab.models.diffusions.schedulers import FlowEulerODEScheduler
from lakonlab.ops import quantize_frozen_linears
from lakonlab.pipelines.pipeline_minimax_h3 import MiniMaxH3DWDBlocks
from lakonlab.pipelines.utils import load_lakonlab_adapter

NUM_STEPS = 4  # 2 or 4
ADAPTER_NAME = f'pi_h3_{NUM_STEPS}step_dwd_adapter'
MAX_RAW_T = {2: 0.93, 4: 0.96}[NUM_STEPS]

# load base pipeline (the VAE only encodes the keyframes, the DWD head decodes the pixels)
manager = ComponentsManager()
pipe = MiniMaxH3DWDBlocks().init_pipeline('multimodalart/MiniMax-H3-Pruned', components_manager=manager)
pipe.load_components(
    names=['text_encoder', 'tokenizer', 'processor', 'vae', 'audio_vae', 'transformer'],
    dtype=torch.bfloat16, trust_remote_code=True)
# load adapter
load_lakonlab_adapter(pipe, 'Lakonik/pi-H3', subfolder=ADAPTER_NAME)

# FP8 W8A8
pipe.transformer.requires_grad_(False)
quantize_frozen_linears(pipe.transformer, exclude_keys=['lora', 'norm_out', 'proj_out', 'dwd_linear_head'])
pipe.text_encoder.requires_grad_(False)
quantize_frozen_linears(pipe.text_encoder, exclude_keys=['lm_head'])
torch.cuda.empty_cache()
# the fastest installed attention kernels for the GPU (the DWD pixel refiner takes variable-length windows)
set_attention_backends(pipe.transformer, backend=default_attention_backend('dense'), overrides={
    'self.dwd_pixel_refiner': default_attention_backend('varlen')})
# Sol sparse attention after the first 2 blocks (requires lakonlab[sol]; remove this loop for dense attention)
for block in pipe.transformer.transformer_blocks[2:]:
    block.attn.set_processor(MiniMaxH3SolAttnProcessor())
    block.attn.set_attention_backend(default_attention_backend('short'))  # for the short prefix query rows

pipe.update_components(
    scheduler=FlowEulerODEScheduler(
        num_train_timesteps=1, shift=12.0, max_raw_t=MAX_RAW_T, final_step_size_scale=0.6),
    audio_scheduler=FlowEulerODEScheduler(
        num_train_timesteps=1, shift=3.0, max_raw_t=MAX_RAW_T, final_step_size_scale=0.6))

pipe.to('cuda')

color_encoder = RGBColorEncoder(
    use_affine_norm=True, mean=(-0.03, -0.088, -0.188), std=(0.458, 0.448, 0.45)).cuda()

# Text-to-video+audio example
prompt = 'A red fox trotting through a snowy pine forest, snow crunching underfoot, birds chirping in the distance'
out = pipe(
    prompt=prompt,
    height=768,
    width=1344,
    num_frames=243,  # 10 s at H3's fixed 24 fps, rounded up to 17n + 5
    num_inference_steps=NUM_STEPS,
    generator=torch.Generator().manual_seed(42),
    output=['dwd_pixels', 'audio', 'sampling_rate'])
video = out.pop('dwd_pixels')
video = color_encoder.decode(video).add_(1).mul_(0.5)
encode_video(
    video[0].permute(1, 2, 3, 0).cpu().numpy(), fps=24, output_path='minimax_h3_dwd_t2va.mp4',
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
    output=['dwd_pixels', 'audio', 'sampling_rate'])
video = out.pop('dwd_pixels')
video = color_encoder.decode(video).add_(1).mul_(0.5)
encode_video(
    video[0].permute(1, 2, 3, 0).cpu().numpy(), fps=24, output_path='minimax_h3_dwd_i2va.mp4',
    audio=out['audio'][0], audio_sample_rate=out['sampling_rate'])
