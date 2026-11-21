from lakonlab.utils.env import apply_default_env
apply_default_env()

import argparse
import enum
import functools
import hashlib
import inspect
import json
import math
import os
import tempfile

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from accelerate import cpu_offload_with_hook
from diffusers.models.attention_dispatch import (
    AttentionBackendName, _AttentionBackendRegistry, dispatch_attention_fn)
from PIL import Image, ImageOps
from lakonlab.models.architectures import RGBColorEncoder
from lakonlab.models.architectures.diffusers.attention import (
    check_attention_backend, default_attention_backend, set_attention_backends)
from lakonlab.models.architectures.diffusers.pretrained import AutoencoderKLMiniMaxH3
from lakonlab.models.architectures.diffusers.minimax_h3 import MiniMaxH3SolAttnProcessor
from lakonlab.models.diffusions.schedulers import FlowEulerODEScheduler
from lakonlab.ops import quantize_frozen_linears
from lakonlab.ops import attention as attention_ops
from lakonlab.ops.quant import FrozenFP8Linear
from lakonlab.ops.sol_attn import get_sol_attn_backend, load_sol_attention_aot
from lakonlab.pipelines.pipeline_minimax_h3 import MiniMaxH3BlocksMod, MiniMaxH3DWDBlocks
from lakonlab.pipelines.utils import load_lakonlab_adapter
from lakonlab.ui.gradio.create_img_to_video import create_interface_img_to_video
from lakonlab.utils.io_utils import LAKONLAB_CACHE, FileClient, rgb_to_yuv420p, save_video_audio


def parse_args():
    parser = argparse.ArgumentParser(description='pi-H3 Gradio Demo')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    parser.add_argument(
        '--offload', action='store_true',
        help='Keep the text encoder on the CPU and move it to the GPU only to encode the prompt (for GPUs with '
             'less memory)')
    parser.add_argument(
        '--crf', type=int, default=23,
        help='libx264 constant rate factor of the output video (lower is higher quality, e.g. 18)')
    parser.add_argument(
        '--aot', action='store_true',
        help='Run the transformer blocks, the DWD refiner blocks, the text encoder (unless --offload) and the audio '
             'decoder through AOTInductor-compiled kernels (compiled once and cached on disk)')
    parser.add_argument(
        '--attention', choices=['sol', 'dense'],
        help='Sol sparse attention after the dense blocks, or dense attention in all blocks '
             '(default: sol if the GPU and lakonlab[sol] support it, otherwise dense)')
    parser.add_argument(
        '--dense-backend', choices=['sage', '_flash_3', 'native'],
        help='Backend of the dense blocks (default: sage, else _flash_3 on SM90, else native SDPA)')
    parser.add_argument(
        '--sol-prefix-backend', choices=['_flash_3', 'native', 'sage'],
        help='Backend of the short prefix query rows of the Sol blocks '
             '(default: _flash_3 on SM90, else native SDPA)')
    parser.add_argument('--sol-tau', type=float, default=1.0)
    parser.add_argument('--sol-thresh-type', choices=['exact', 'diag'], default='exact')
    parser.add_argument('--sol-dense-blocks', type=int, default=2)
    parser.add_argument('--sol-kv-splits', type=int, choices=[1, 2, 4], default=None,
                        help='KV splits; by default select four on SM90 at >=65536 tokens, otherwise one')
    parser.add_argument('--sol-precompile', action='store_true',
                        help='Warm every Sol token bucket in the supported AOT range before serving')
    parser.add_argument('--sol-aot', help='Load an offline Sol artifact bundle from tools/export_sol_attention.py')
    return parser.parse_args()


ADAPTER_REPO = 'Lakonik/pi-H3'
MAX_TOKENS = 2 ** 17

DEFAULT_PROMPT = ('integrated_multimodal_description: [Shot 1] Live-action, handheld smartphone selfie video, a close '
                  "medium shot with the camera held at arm's length slightly above eye level, a young man in an orange "
                  'beanie and a green rain jacket fills the left 40 percent of the frame and looks straight into the '
                  'lens. Behind him, a lakeside campsite glows at golden hour, with a yellow tent at the right edge of '
                  'the background and a dark treeline above the shore. The camera shakes slightly. The young man with '
                  'a cheerful, breathy voice (S1) says: <d>[English] Day two of camping, and the vibes are '
                  'immaculate.</d> While he speaks without noticing, a large brown bear steps out of the treeline in '
                  'the right background and walks along the shore toward the camera, growing larger in the frame. It '
                  'stops directly behind him and lowers its huge head onto his shoulder nearest the right edge of the '
                  'frame, so that its snout reaches into the right half of the frame beside his ear. His smile freezes '
                  'in place and his eyes slide sideways toward the bear without his head moving.\n\noverall_soundscape: '
                  'Wind hisses across the lake while gravel crunches under heavy, slow paws. The bear lets out a long, '
                  'warm huff beside the microphone and his nylon jacket rustles as he goes rigid.\n\nnon_diegetic_music: '
                  'N/A')


def canvas_size(aspect_ratio, num_pixels, multiple=32):
    """(height, width) of the aspect ratio at num_pixels, both rounded up to the multiple, so 1 MP gives H3's native
    768x1344 for 16:9 and 672x1536 for 21:9."""
    aspect_width, aspect_height = map(float, aspect_ratio.split(':'))
    height = (num_pixels * aspect_height / aspect_width) ** 0.5
    width = (num_pixels * aspect_width / aspect_height) ** 0.5
    return math.ceil(height / multiple) * multiple, math.ceil(width / multiple) * multiple


_source_hash = functools.cache(lambda path: hashlib.sha1(open(path, 'rb').read()).hexdigest())


def aot_compile_block(block, example_inputs, dynamic_shapes, name):
    """Regional AOTInductor compilation of a block: one export (dynamic in dynamic_shapes) and compile per distinct
    block layout, cached on disk, run by the block on its own weights in place. A layout covers what the export bakes
    in: the source files of the block's module classes, of its attention backend and of this function, the scalar
    attributes (and containers of scalars, e.g. LoRA scaling) of its modules and attention processor, its tensors'
    shapes and dtypes, and the example inputs' non-tensor values, tensor shapes and dtypes and dynamic shapes. Python
    helpers that a forward calls from other files are not covered."""
    scalar = lambda v: isinstance(v, (bool, int, float, str, enum.Enum, type(None))) or (  # noqa: E731
        isinstance(v, (tuple, list, dict)) and all(map(scalar, v.values() if isinstance(v, dict) else v)))
    constants = {**dict(block.named_parameters()), **dict(block.named_buffers())}
    device = next(iter(constants.values())).device  # the block's GPU, not necessarily the current one
    processors = [m.processor for m in block.modules() if hasattr(m, 'processor')]  # of diffusers attention modules
    modules = [*block.modules(), *processors]
    backends = {_AttentionBackendRegistry._backends[AttentionBackendName(
        getattr(p, '_attention_backend', None) or _AttentionBackendRegistry.get_active_backend()[0])]
        for p in processors}
    files = {inspect.getsourcefile(cls) for m in modules for cls in type(m).__mro__ if cls is not object}
    files |= {inspect.getsourcefile(backend) for backend in backends}
    files |= {inspect.getsourcefile(dispatch_attention_fn), inspect.getsourcefile(attention_ops)}
    settings = [(type(m).__qualname__, sorted((k, repr(v)) for k, v in vars(m).items() if scalar(v)))
                for m in modules]
    tensors = [v for v in torch.utils._pytree.tree_leaves(example_inputs) if isinstance(v, torch.Tensor) and v.numel()]
    # the export ties example inputs that share memory, which the runtime inputs need not do
    assert len({v.untyped_storage().data_ptr() for v in tensors}) == len(tensors), 'example inputs share memory'
    inputs = [(tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else repr(v)
              for v in torch.utils._pytree.tree_leaves(example_inputs)]
    layout = [torch.__version__, torch.cuda.get_device_name(device), inspect.getsource(aot_compile_block), settings,
              sorted((f, _source_hash(f)) for f in files), inputs, repr(dynamic_shapes)] + [
        (k, tuple(v.shape), str(v.dtype)) for k, v in sorted(constants.items())]
    path = os.path.join(LAKONLAB_CACHE, 'aot', f'{name}_{hashlib.sha1(repr(layout).encode()).hexdigest()[:16]}.pt2')
    if not os.path.exists(path):
        with torch.no_grad(), torch.cuda.device(device):
            exported = torch.export.export(block, example_inputs, dynamic_shapes=dynamic_shapes)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch._inductor.aoti_compile_and_package(
                exported, package_path=path, inductor_configs={'aot_inductor.package_constants_in_so': False})
    compiled = torch._inductor.aoti_load_package(path, device_index=device.index)
    compiled.load_constants(
        {k: constants[k] for k in compiled.get_constant_fqns()}, check_full_update=True, user_managed=True)
    block.forward = compiled


def aot_compile_transformer_blocks(transformers):
    """AOTInductor-compile the MiniMax H3 transformer blocks, dynamic in the token count."""
    transformer = transformers[0]
    position_ids = torch.zeros(4096, 3, dtype=torch.long, device='cuda')
    example_inputs = (
        torch.randn(1, 4096, transformer.config.hidden_size, dtype=torch.bfloat16, device='cuda'),
        torch.zeros(2, transformer.config.adaln_rank, dtype=torch.bfloat16, device='cuda'),
        torch.zeros(4096, dtype=torch.long, device='cuda'),
        transformer.rope(position_ids),
        dict(sol_prefix=torch.tensor(1, device='cpu')))
    tokens = torch.export.Dim('tokens', min=256, max=MAX_TOKENS)
    levels = torch.export.Dim('levels', min=2, max=16)  # distinct noise levels of the AdaLN table
    dynamic_shapes = ({1: tokens}, {0: levels}, {0: tokens}, ({0: tokens}, {0: tokens}), dict(sol_prefix={}))
    for transformer in transformers:
        for block in transformer.transformer_blocks:
            aot_compile_block(block, example_inputs, dynamic_shapes, 'minimax_h3_block')


def aot_compile_refiner_blocks(refiner):
    """AOTInductor-compile the Sol blocks of the DWD pixel refiner, one layout per layer parity, dynamic in the
    latent frames and the token grid."""
    x = torch.randn(1, 12, 48, 84, refiner.norm_out.normalized_shape[0], dtype=torch.bfloat16, device='cuda')
    g = x.new_zeros(1, 3, refiner.register_tokens.size(1) + 1, x.size(-1))  # global tokens of 3 native blocks
    frames, height, width = (torch.export.Dim(name, min=2 if name == 'frames' else 16, max=256)
                             for name in ('frames', 'height', 'width'))
    blocks = torch.export.Dim('blocks', min=1, max=64)
    for block, args in refiner.block_args(x.shape, g.shape, x.device):
        _, global_indices, window_indices = args
        global_shapes = tuple(
            {0: blocks, 1: torch.export.Dim.AUTO} if rows.ndim == 2 else {0: torch.export.Dim.AUTO}
            for rows in global_indices)
        window_shapes = tuple({0: torch.export.Dim.AUTO} for _ in window_indices)
        dynamic_shapes = ({1: frames, 2: height, 3: width}, {1: blocks}, None, global_shapes, window_shapes)
        aot_compile_block(block, (x, g, *args), dynamic_shapes, 'sol_h3_block')


class _Qwen3VLTextLayer(torch.nn.Module):
    """A Qwen3-VL text decoder layer as its text model calls it, without a mask (causal) or a cache."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, cos, sin):
        return self.layer(hidden_states, position_embeddings=(cos, sin))


class _Qwen3VLVisionBlock(torch.nn.Module):
    """A Qwen3-VL vision block on the patches of one image."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, cu_seqlens, cos, sin):
        return self.block(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))


def aot_compile_text_encoder(text_encoder):
    """AOTInductor-compile the Qwen3-VL text encoder: its decoder layers, dynamic in the token count, and its vision
    blocks, dynamic in the patch count of one image. Attention never crosses images, so the vision blocks run the
    images of a request one at a time."""
    model = text_encoder.model
    text_config, vision_config = model.language_model.config, model.visual.config
    x = torch.zeros(1, 64, text_config.hidden_size, dtype=text_encoder.dtype, device=text_encoder.device)
    cos, sin = (x.new_zeros(1, 64, text_config.head_dim) for _ in range(2))
    tokens = torch.export.Dim('tokens', min=2, max=16384)
    for layer in model.language_model.layers:
        wrapper = _Qwen3VLTextLayer(layer)
        aot_compile_block(wrapper, (x, cos, sin), ({1: tokens},) * 3, 'qwen3vl_text_layer')

        def forward(hidden_states, position_embeddings, attention_mask=None, _wrapper=wrapper, **kwargs):
            assert attention_mask is None, 'the compiled layer attends causally, without a padding mask'
            return _wrapper(hidden_states, *position_embeddings)
        layer.forward = forward

    x = torch.zeros(1024, vision_config.hidden_size, dtype=text_encoder.dtype, device=text_encoder.device)
    cu_seqlens = torch.tensor([0, x.size(0)], dtype=torch.int32, device=x.device)
    cos, sin = (x.new_zeros(x.size(0), vision_config.hidden_size // vision_config.num_heads, dtype=torch.float32)
                for _ in range(2))
    patches = torch.export.Dim('patches', min=16, max=65536)
    for block in model.visual.blocks:
        wrapper = _Qwen3VLVisionBlock(block)
        aot_compile_block(
            wrapper, (x, cu_seqlens, cos, sin), ({0: patches}, None, {0: patches}, {0: patches}),
            'qwen3vl_vision_block')

        def forward(hidden_states, cu_seqlens, position_embeddings, _wrapper=wrapper, **kwargs):
            if cu_seqlens.numel() == 2:  # one image
                return _wrapper(hidden_states, cu_seqlens, *position_embeddings)
            bounds = cu_seqlens.tolist()
            return torch.cat([
                _wrapper(hidden_states[a:b], cu_seqlens.new_tensor([0, b - a]), *(r[a:b] for r in position_embeddings))
                for a, b in zip(bounds, bounds[1:])])
        block.forward = forward


def aot_compile_audio_decoder(audio_vae):
    """AOTInductor-compile the audio VAE decoder, dynamic in the latent length."""
    x = torch.zeros(2, audio_vae.dec_in_proj.out_channels, 200, dtype=audio_vae.dtype, device=audio_vae.device)
    aot_compile_block(
        audio_vae.decoder, (x,), ({2: torch.export.Dim('frames', min=16, max=4096)},), 'minimax_h3_audio_decoder')


def main():
    args = parse_args()
    # unspecified backends are selected from the GPU and the installed packages
    sm_major = torch.cuda.get_device_capability()[0]
    sol_supported = get_sol_attn_backend() in ('cute_sm90', 'cute_sm100', 'cute_sm120')
    if args.attention is None:
        args.attention = 'sol' if sol_supported else 'dense'
    elif args.attention == 'sol' and not sol_supported:
        raise RuntimeError('Sol attention requires an SM90, SM100/SM103 or SM120 GPU and lakonlab[sol].')
    args.dense_backend = args.dense_backend or default_attention_backend('dense')
    args.sol_prefix_backend = args.sol_prefix_backend or default_attention_backend('short')
    refiner_backend = default_attention_backend('varlen')  # the Sol refiner's variable-length windows
    for backend in {args.dense_backend, args.sol_prefix_backend}:  # explicit choices, before loading the weights
        check_attention_backend(backend)
    print(f'Attention: {args.attention}, dense backend: {args.dense_backend}, '
          f'Sol prefix backend: {args.sol_prefix_backend}, DWD refiner backend: {refiner_backend}')
    if args.attention != 'sol' and (args.sol_aot or args.sol_precompile):
        raise ValueError('--sol-aot and --sol-precompile require Sol attention.')
    if args.sol_aot:
        if args.sol_precompile:
            raise ValueError('--sol-aot cannot be combined with --sol-precompile.')
        load_sol_attention_aot(args.sol_aot)

    # load base pipeline with batched VAE decoding
    base_pipe = MiniMaxH3BlocksMod().init_pipeline('multimodalart/MiniMax-H3-Pruned')
    vae_spec = base_pipe.get_component_spec('vae')
    vae_spec.type_hint = AutoencoderKLMiniMaxH3
    base_pipe.update_components(vae=vae_spec.load(dtype=torch.float16))
    base_pipe.load_components(
        names=['text_encoder', 'tokenizer', 'processor', 'audio_vae', 'transformer'],
        dtype=torch.bfloat16, trust_remote_code=True)
    base_pipe.vae.decode_tile_batch_size = 28
    base_pipe.text_encoder.requires_grad_(False)
    quantize_frozen_linears(base_pipe.text_encoder, exclude_keys=['lm_head'])  # layer by layer, before .cuda()
    base_pipe.transformer.requires_grad_(False).cuda()  # the adapters are loaded onto the base weights' device

    scheduler_2step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=12.0, max_raw_t=0.93, final_step_size_scale=0.6)
    audio_scheduler_2step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=3.0, max_raw_t=0.93, final_step_size_scale=0.6)
    scheduler_4step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=12.0, max_raw_t=0.96, final_step_size_scale=0.6)
    audio_scheduler_4step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=3.0, max_raw_t=0.96, final_step_size_scale=0.6)
    components = {name: getattr(base_pipe, name) for name in ['vae', 'audio_vae', 'transformer']}

    # the text encoder runs in its own pipeline, so it can sit on a second GPU
    blocks = MiniMaxH3BlocksMod()
    dwd_blocks = MiniMaxH3DWDBlocks()
    text_blocks = MiniMaxH3BlocksMod.from_blocks_dict(
        {name: blocks.sub_blocks[name] for name in ('before_encode', 'text_encoder')})
    blocks.sub_blocks.pop('text_encoder')
    dwd_blocks.sub_blocks.pop('text_encoder')

    text_pipe = text_blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    text_pipe.update_components(
        text_encoder=base_pipe.text_encoder, tokenizer=base_pipe.tokenizer, processor=base_pipe.processor)

    pipe_2step = blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_2step.update_components(**components, scheduler=scheduler_2step, audio_scheduler=audio_scheduler_2step)
    load_lakonlab_adapter(pipe_2step, ADAPTER_REPO, subfolder='pi_h3_2step_adapter')

    pipe_4step = blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_4step.update_components(**components, scheduler=scheduler_4step, audio_scheduler=audio_scheduler_4step)
    load_lakonlab_adapter(pipe_4step, ADAPTER_REPO, subfolder='pi_h3_4step_adapter')

    pipe_2step_dwd = dwd_blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_2step_dwd.update_components(**components, scheduler=scheduler_2step, audio_scheduler=audio_scheduler_2step)
    load_lakonlab_adapter(pipe_2step_dwd, ADAPTER_REPO, subfolder='pi_h3_2step_dwd_adapter')

    pipe_4step_dwd = dwd_blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_4step_dwd.update_components(**components, scheduler=scheduler_4step, audio_scheduler=audio_scheduler_4step)
    load_lakonlab_adapter(pipe_4step_dwd, ADAPTER_REPO, subfolder='pi_h3_4step_dwd_adapter')

    pipe_2step.transformer.requires_grad_(False)
    pipe_4step.transformer.requires_grad_(False)
    pipe_2step_dwd.transformer.requires_grad_(False)
    pipe_4step_dwd.transformer.requires_grad_(False)

    # a DWD adapter is trained on top of its frozen piFlow adapter and shares its LoRA layers
    for dwd_pipe, piflow_pipe in ((pipe_2step_dwd, pipe_2step), (pipe_4step_dwd, pipe_4step)):
        for name, m in list(dwd_pipe.transformer.named_modules()):
            if name.endswith(('.lora_A', '.lora_B')):
                dwd_pipe.transformer.set_submodule(name, piflow_pipe.transformer.get_submodule(name))

    # the base transformer is quantized in place; every pipe takes its fp8 linears wherever it holds the base weight,
    # then quantizes the linears its adapter replaced
    quant_exclude_keys = ['lora', 'adaln_proj', 'norm_out', 'proj_out']
    base_linears = {m.weight.data_ptr(): m for m in base_pipe.transformer.modules() if isinstance(m, torch.nn.Linear)}
    quantize_frozen_linears(base_pipe.transformer, exclude_keys=quant_exclude_keys)
    fp8_linears = {ptr: m for ptr, m in base_linears.items() if isinstance(m, FrozenFP8Linear)}
    for _pipe in (pipe_2step, pipe_4step, pipe_2step_dwd, pipe_4step_dwd):
        for name, m in list(_pipe.transformer.named_modules()):
            if isinstance(m, torch.nn.Linear) and m.weight.data_ptr() in fp8_linears:
                _pipe.transformer.set_submodule(name, fp8_linears[m.weight.data_ptr()])
        quantize_frozen_linears(_pipe.transformer, exclude_keys=quant_exclude_keys)

    text_device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    if args.offload:  # the text encoder returns to the CPU as soon as a transformer runs
        _, text_encoder_hook = cpu_offload_with_hook(base_pipe.text_encoder, execution_device=text_device)
        for _pipe in (pipe_2step, pipe_4step, pipe_2step_dwd, pipe_4step_dwd):
            _pipe.transformer.register_forward_pre_hook(lambda *_: text_encoder_hook.offload())
    else:
        base_pipe.text_encoder.to(text_device)
    base_pipe.vae.cuda()
    base_pipe.audio_vae.cuda()

    del base_pipe, components, base_linears, fp8_linears
    torch.cuda.empty_cache()

    for _pipe in (pipe_2step, pipe_4step):
        set_attention_backends(_pipe.transformer, backend=args.dense_backend)
    for _pipe in (pipe_2step_dwd, pipe_4step_dwd):
        set_attention_backends(
            _pipe.transformer, backend=args.dense_backend, overrides={'self.dwd_pixel_refiner': refiner_backend})
    if args.attention == 'sol':  # Sol after the dense blocks
        for _pipe in (pipe_2step, pipe_4step, pipe_2step_dwd, pipe_4step_dwd):
            for block in _pipe.transformer.transformer_blocks[args.sol_dense_blocks:]:
                processor = MiniMaxH3SolAttnProcessor()
                processor.tau, processor.thresh_type = args.sol_tau, args.sol_thresh_type
                processor.kv_splits = args.sol_kv_splits or 0
                block.attn.set_processor(processor)
                block.attn.set_attention_backend(args.sol_prefix_backend)
    if args.aot:
        aot_compile_transformer_blocks([
            pipe_2step.transformer, pipe_4step.transformer,
            pipe_2step_dwd.transformer, pipe_4step_dwd.transformer])
        for _pipe in (pipe_2step_dwd, pipe_4step_dwd):
            aot_compile_refiner_blocks(_pipe.transformer.dwd_pixel_refiner)
        if not args.offload:  # the packages hold the weights' GPU addresses
            aot_compile_text_encoder(text_pipe.text_encoder)
        aot_compile_audio_decoder(pipe_2step.audio_vae)
    if args.sol_precompile:
        attn = pipe_2step.transformer.transformer_blocks[0].attn
        with torch.no_grad():
            bucket_size = attention_ops.SOL_COMPILE_BUCKET_SIZE
            start = (args.sol_kv_splits or 1) * bucket_size
            for tokens in range(start, MAX_TOKENS + 1, bucket_size):
                qkv = torch.zeros(
                    (1, tokens, attn.heads, attn.head_dim), device='cuda', dtype=torch.bfloat16)
                attention_ops.sol_attention(
                    qkv, qkv, qkv,
                    torch.tensor(0, device='cpu'),
                    args.sol_tau,
                    args.sol_thresh_type,
                    0 if args.sol_kv_splits is None else args.sol_kv_splits,
                    'sdpa')
                # The 65536-token bucket straddles the automatic split-count transition.
                if args.sol_kv_splits is None and tokens == 65536 and sm_major == 9:
                    attention_ops.sol_attention(
                        qkv, qkv, qkv,
                        torch.tensor(0, device='cpu'),
                        args.sol_tau,
                        args.sol_thresh_type,
                        1,
                        'sdpa')
                del qkv
        torch.cuda.empty_cache()

    color_encoder = RGBColorEncoder(
        use_affine_norm=True, mean=(-0.03, -0.088, -0.188), std=(0.458, 0.448, 0.45)).cuda()

    def generate(
            seed, prompt, in_image, in_last_image, aspect_ratio, megapixels, duration, steps, remove_decoder,
            progress=gr.Progress(track_tqdm=True)):
        height, width = canvas_size(aspect_ratio, megapixels * 1e6)
        # H3 generates 17n + 5 frames at 24 fps, from 124 (5.2 s) to 345 (14.4 s)
        num_frames = min(max(round((duration * 24 - 5) / 17), 7), 20) * 17 + 5
        steps = 2 if steps < 3 else 4
        if remove_decoder:
            pipe = pipe_2step_dwd if steps == 2 else pipe_4step_dwd
        else:
            pipe = pipe_2step if steps == 2 else pipe_4step
        # keyframes are center-cropped and resized onto the requested canvas
        image = None if in_image is None else ImageOps.fit(
            in_image, (width, height), Image.Resampling.LANCZOS)
        last_image = None if in_last_image is None else ImageOps.fit(
            in_last_image, (width, height), Image.Resampling.LANCZOS)
        with torch.cuda.device(text_device):  # the fp8 triton kernels launch on the current device
            text = text_pipe(
                prompt=prompt, image=image, last_image=last_image, height=height, width=width,
                num_frames=num_frames, output=['prompt_embeds', 'text_token_tags'])
        out = pipe(
            prompt_embeds=text['prompt_embeds'].to('cuda:0'),
            text_token_tags=text['text_token_tags'],
            image=image,
            last_image=last_image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
            output=['dwd_pixels' if remove_decoder else 'videos', 'audio', 'sampling_rate'],
            **({} if remove_decoder else dict(output_type='pt')),  # the DWD pipeline has no decode step
        )
        if remove_decoder:  # RGB (C, T, H, W) in [-1, 1] to (T, C, H, W) in [0, 1]
            out['videos'] = color_encoder.decode(out.pop('dwd_pixels')).div_(2).add_(0.5).clamp_(
                min=0, max=1).transpose(1, 2)
        # generation settings for reproduction, as JSON in the mp4 comment tag (keyframes as hashes of the fitted images)
        settings = dict(
            seed=seed, prompt=prompt, steps=steps, remove_decoder=remove_decoder, aspect_ratio=aspect_ratio,
            megapixels=megapixels, height=height, width=width, num_frames=num_frames, fps=24,
            first_frame=None if image is None else hashlib.sha256(image.tobytes()).hexdigest(),
            last_frame=None if last_image is None else hashlib.sha256(last_image.tobytes()).hexdigest(),
            adapter=f'{ADAPTER_REPO}/pi_h3_{steps}step{"_dwd" if remove_decoder else ""}_adapter',
            attention=args.attention, dense_backend=args.dense_backend, refiner_backend=refiner_backend, aot=args.aot,
            **({k: v for k, v in vars(args).items() if k.startswith('sol_') and k not in ('sol_aot', 'sol_precompile')}
               if args.attention == 'sol' else {}))
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        save_video_audio(
            rgb_to_yuv420p(out['videos'][0]).cpu().numpy(), out['audio'][0].float().cpu().numpy(), output_path,
            FileClient.infer_client(uri=output_path), 24, out['sampling_rate'], pix_fmt='yuv420p',
            options={'preset': 'veryfast', 'crf': str(args.crf)}, metadata={'comment': json.dumps(settings)})
        return output_path

    with gr.Blocks(analytics_enabled=False,
                   title='pi-H3 Demo'
                   ) as demo:

        md_txt = '# pi-H3 Demo 1.0\n\n' \
                 'The fastest MiniMax H3 video model: 2 or 4 steps, direct pixel output without a video VAE decoder. ' \
                 'Accelerated by **pi-Flow V2** and **[DWD](https://hanshengchen.com/dwd/)** ' \
                 '(**D**ecoding **W**ithout a **D**ecoder). ' \
                 'Read the [blog post](https://hanshengchen.com/blog/pi-h3/) for more details.\n' \
                 '<br> MiniMax H3 **prompting guide**: [https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)'
        gr.Markdown(md_txt)

        create_interface_img_to_video(
            generate,
            prompt=DEFAULT_PROMPT,
            steps=2, min_steps=2, max_steps=4, steps_slider_step=2, remove_decoder=True,
            aspect_ratios=('21:9', '16:9', '4:3', '1:1', '3:4', '9:16'),
            examples=json.load(open('demo/examples/minimax_h3/examples.json')),
            args=['last_seed', 'prompt', 'in_image', 'in_last_image', 'aspect_ratio', 'megapixels', 'duration',
                  'steps', 'remove_decoder'])
        demo.queue().launch(
            share=args.share, ssr_mode=False,
            css_paths='lakonlab/ui/gradio/style.css', head_paths='lakonlab/ui/gradio/head.html')


if __name__ == "__main__":
    main()
