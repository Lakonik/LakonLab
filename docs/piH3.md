# pi-H3: Video Generation in 2–4 Steps Without a VAE Decoder

Official PyTorch implementation of:

**pi-H3: Video Generation in 2–4 Steps Without a VAE Decoder**
<br>
Blog post, 2026
<br>
[Hansheng Chen](https://lakonik.github.io/)<sup>1*</sup>,
[Shengqu Cai](https://primecai.github.io/)<sup>1</sup>,
[Jiaming Song](https://tsong.me/)<sup>†</sup>,
[Gordon Wetzstein](http://web.stanford.edu/~gordonwz/)<sup>1</sup>,
[Xinchen Yan](https://sites.google.com/site/skywalkeryxc)<sup>†</sup>,
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)<sup>1</sup><br>
<sup>1</sup>Stanford University<br>
<sup>*</sup>Part of work done during an internship at Luma AI, <sup>†</sup>Work done while at Luma AI
<br>
[Blog](https://hanshengchen.com/blog/pi-h3/) | [ComfyUI](https://github.com/Lakonik/ComfyUI-piFlow) | [pi-H3 Demo🤗](https://huggingface.co/spaces/Lakonik/pi-H3)

<img src="https://hanshengchen.com/blog/pi-h3/teaser.jpg" width="500" alt=""/>

## Highlights

pi-H3 accelerates [MiniMax H3](https://www.minimax.io/blog/minimax-h3) video generation with audio by combining two techniques:

- **pi-Flow v2**: A full upgrade of [pi-Flow](piFlow.md) for data-free few-step distillation. pi-H3 generates video in 2 or 4 sampling steps.
- **[DWD](https://hanshengchen.com/dwd/) (Decoding Without a Decoder)**: Instead of passing the final latents through a VAE decoder, DWD re-uses the denoiser itself to predict clean pixels at the last step, removing the video decoder from the pipeline.

pi-H3 supports text-to-video, image-to-video, and first-and-last-frame-to-video generation. For inference, this codebase also provides FP8 quantization and [Sol-Attn](../lakonlab/ops/sol_attn) sparse attention. See the [blog](https://hanshengchen.com/blog/pi-h3/) for speed and quality comparisons.

## Installation

Follow the [root README](../README.md#installation) to install LakonLab, including the CUDA build prerequisites. The [`sol` extra](../README.md#sol-attn) is optional but highly recommended: the Gradio app uses Sol sparse attention when it is installed and the GPU supports it. pi-H3 inference also uses:

- [SageAttention](https://github.com/thu-ml/SageAttention) (optional, recommended): the backend of the dense blocks on all GPUs except SM100/SM103. The source build targets the GPUs of the build machine; set `TORCH_CUDA_ARCH_LIST` to build for others.

  ```bash
  git clone https://github.com/thu-ml/SageAttention.git
  (cd SageAttention && uv pip install --no-build-isolation .)
  ```

- [FlashAttention 3](https://github.com/Dao-AILab/flash-attention) (optional, recommended on SM90): the backend of the Sol prefix rows and the DWD pixel refiner, which otherwise use PyTorch SDPA. Its build needs `wheel`:

  ```bash
  uv pip install wheel
  git clone https://github.com/Dao-AILab/flash-attention.git
  (cd flash-attention/hopper && uv pip install --no-build-isolation .)
  ```

  pi-H3 inference only needs the BF16 forward kernels with head dimensions 64 and 128. To build just those, which is much faster, replace the last line above with:

  ```bash
  (
    cd flash-attention/hopper || exit 1
    for f in SM80 FP8 FP16 HDIM96 HDIM192 HDIM256 HDIMDIFF64 HDIMDIFF192 BACKWARD PAGEDKV APPENDKV SOFTCAP LOCAL; do
      export FLASH_ATTENTION_DISABLE_$f=TRUE
    done
    FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --no-build-isolation .
  )
  ```

## Inference: Diffusers Pipeline

### [pi-H3](../demo/example_pih3_pipeline.py)

The following code generates a 10-second video with audio in 4 steps; set `NUM_STEPS = 2` for the 2-step model.

```python
import torch
from diffusers import ComponentsManager
from diffusers.utils import encode_video
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
```

To condition on a first frame or a last frame, pass `image=...` or `last_image=...`. The [example script](../demo/example_pih3_pipeline.py) includes an image-to-video example.

### [Non-DWD pi-H3](../demo/example_pih3_vae_pipeline.py)

We also release 2-step and 4-step non-DWD pi-H3 adapters (`pi_h3_2step_adapter` and `pi_h3_4step_adapter`), whose latents are decoded by the native H3 VAE. See [example_pih3_vae_pipeline.py](../demo/example_pih3_vae_pipeline.py) for usage.

## Inference: Gradio App

The official app is available on HuggingFace Spaces: [pi-H3 Demo🤗](https://huggingface.co/spaces/Lakonik/pi-H3).

Run the following command from the repository root to launch the app locally:

```bash
PYTHONPATH=. python demo/gradio_pih3.py --share
```

The app takes a text prompt and optional first and last frames, and generates video with audio in 2 or 4 steps. Without arguments, it selects the [attention backends](#attention-backends) for the GPU and the installed packages. **Remove decoder** is enabled by default, which generates with pi-H3. Disabling it switches to non-DWD pi-H3, decoded by the native H3 VAE. For prompt writing, see the [MiniMax H3 prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

With two GPUs visible, the text encoder runs on `cuda:1` and the transformer on `cuda:0`. On a single GPU with insufficient memory, `--offload` keeps the text encoder on the CPU when it is not in use. `--crf` sets the output video quality (lower is better; default 23).

## Accelerated Inference

### Attention Backends

[Sol-Attn](../lakonlab/ops/sol_attn) is a forward-only sparse attention kernel for SM90, SM100/SM103, and SM120 GPUs; on SM120, it uses block-sparse attention. The Gradio app selects its attention backends automatically, and the following options override the selection:

| Option | Choices | Default |
|---|---|---|
| `--attention` | `sol`, or `dense` for dense attention in all blocks | `sol` if the `sol` extra is installed and the GPU supports it, otherwise `dense` |
| `--dense-backend` | `sage` (SageAttention), `_flash_3` (FlashAttention 3), or `native` (PyTorch SDPA) | `sage` if installed, except on SM100/SM103; otherwise `_flash_3` if installed and on SM90; otherwise `native` |
| `--sol-prefix-backend` | `_flash_3`, `native`, or `sage` | `_flash_3` if installed and on SM90, otherwise `native` |
| DWD pixel refiner | not configurable | FlashAttention 3 if installed and on SM90, otherwise PyTorch SDPA |

The app prints the selected backends at startup.

With Sol, the first `--sol-dense-blocks` transformer blocks (default 2) keep dense attention. The prefix rows (text, conditioning, and audio tokens) of the remaining blocks also get dense queries, from `--sol-prefix-backend`; these queries are short, which SageAttention handles inefficiently. Additional Sol options:

- `--sol-tau` and `--sol-thresh-type`: the routing settings (defaults `1.0` and `exact`).
- `--sol-kv-splits`: the number of KV splits. By default, SM90 uses 4 splits from 65,536 tokens and 1 split below. SM120 supports only 1 split.

Sol kernels are compiled on the first use of each layout. `--sol-precompile` compiles all token buckets up to 131,072 tokens at startup instead.

### AOT Compilation

`--aot` compiles each distinct transformer block layout once with AOTInductor and caches the packages under `$LAKONLAB_CACHE/aot` (default `~/.cache/lakonlab/aot`). The Sol blocks of the DWD pixel refiner, the decoder layers and vision blocks of the text encoder (except with `--offload`), and the audio VAE decoder are compiled the same way. The cache key covers the PyTorch version, the GPU, and the block and attention settings. The video VAE runs eagerly.

### Offline Sol Bundles

On SM90 and SM120 GPUs, the Sol kernels can be exported to an offline bundle, so that serving requires no attention kernel compilation. Export the bundle on the target GPU in the serving environment:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/export_sol_attention.py /path/to/sol-bundle
```

Then load it in the app, together with AOT compilation:

```bash
PYTHONPATH=. python demo/gradio_pih3.py --aot --sol-aot /path/to/sol-bundle
```

`--sol-aot` cannot be combined with `--sol-precompile`. The app's `--sol-tau`, `--sol-thresh-type`, and `--sol-kv-splits` must match the bundle; on SM90, the exporter includes 1 and 4 KV splits by default. A bundle is rejected if the Sol sources, the Python or dependency versions, or the GPU architecture differ from those at export time, in which case it must be re-exported. Other custom kernels, such as FP8 quantization, are not part of the bundle.

## Model Weights

The pi-H3 adapters are hosted at [Lakonik/pi-H3](https://huggingface.co/Lakonik/pi-H3) and are distributed under the [MiniMax H3 Community License Agreement](https://huggingface.co/Lakonik/pi-H3/blob/main/LICENSE.md).

## Training

The training code and configs will be released later.

## Citation

```
@misc{chen2026pih3,
  title        = {{pi-H3: Video Generation in 2–4 Steps Without a VAE Decoder}},
  author       = {Hansheng Chen and Shengqu Cai and Jiaming Song and Gordon Wetzstein and Xinchen Yan and Leonidas Guibas},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://hanshengchen.com/blog/pi-h3/}}
}

@article{chen2026decoding,
  title={Decoding Without a Decoder: Direct Pixel Prediction from Diffusion Transformers},
  author={Hansheng Chen and Shengqu Cai and Jan Ackermann and Rayhan Zirvi and Gordon Wetzstein and Xinchen Yan and Leonidas Guibas},
  journal={arXiv preprint arXiv:2611.22287},
  url={https://arxiv.org/abs/2611.22287},
  year={2026},
}
```
