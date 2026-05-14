# ComfyUI-AsymFlux2

ComfyUI nodes for **Lakonik's AsymFLUX.2-klein-9B**
([HuggingFace](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B),
[paper](https://hanshengchen.com/asymflow)).

AsymFLUX.2 is a *pixel-space* finetune of FLUX.2-klein-9B that uses
"rank-asymmetric flow parameterization" — the 707 MB adapter file is a hybrid
of (a) full-weight replacements for the I/O projections and (b) rank-256 LoRA
deltas for 58 internal modules. It is **not** a plain LoRA, and it
**changes the model's input shape** from a 128-channel VAE latent to a
3-channel pixel image.

## Install

Clone or copy this folder into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes/
git clone <this repo>/comfyui/ComfyUI-AsymFlux2
```

Place the adapter safetensors in `ComfyUI/models/asymflux2/`:

```bash
# in your ComfyUI install:
mkdir -p models/asymflux2
huggingface-cli download Lakonik/AsymFLUX.2-klein-9B \
    diffusion_pytorch_model.safetensors \
    --local-dir models/asymflux2/
```

You also need the base model already working: **FLUX.2-klein-base-9B** must be
downloaded and loadable by ComfyUI's standard `UNETLoader` /
`CheckpointLoaderSimple`. ComfyUI core has supported FLUX.2 since commit
[`6b573ae`](https://github.com/comfyanonymous/ComfyUI/commit/6b573ae)
(Nov 25 2025).

## Nodes

| Node | Inputs | Output |
|---|---|---|
| **AsymFlux2 Load Adapter** | `MODEL`, `adapter_name` | `MODEL` (patched) |
| **AsymFlux2 Empty Latent (pixel)** | width, height, batch | `LATENT` (3 × H × W) |
| **AsymFlux2 Sigmas (sqrt-shift)** | steps, width, height | `SIGMAS` |
| **Oklab Encode** | `IMAGE` | `LATENT` |
| **Oklab Decode** | `LATENT` | `IMAGE` |

### Sample graph

```
[UNETLoader (FLUX.2-klein-base-9B)] ─┐
                                     ▼
                          [AsymFlux2 Load Adapter] ──► MODEL
[CLIPLoader] ──► CLIP                                   │
[CLIPTextEncode] ──► positive ──┐                       │
[CLIPTextEncode] ──► negative ──┤                       │
[AsymFlux2 Empty Latent]   ─────┤                       │
[AsymFlux2 Sigmas]         ─────┤                       │
[KSamplerSelect (uni_pc)]  ─────┴── [SamplerCustom] ────┘
                                          │
                                       LATENT
                                          ▼
                                   [Oklab Decode]
                                          ▼
                                       [SaveImage]
```

## How the adapter loader works

The 707 MB safetensors contains exactly **121 tensors**, split as:

- 5 full-weight overrides:
  `x_embedder.weight`, `proj_out.weight`, `norm_out.linear.weight`,
  `proj_buffer`, `scale_buffer`.
- 116 LoRA tensors (rank 256, alpha 256 → scaling 1.0) targeting 58 modules
  across `transformer_blocks` (8), `single_transformer_blocks` (24), and
  `time_guidance_embed.timestep_embedder`.

Diffusers-style names are translated to ComfyUI's BFL-style FLUX names by
`key_map.LORA_MAP` and `key_map.FULL_OVERRIDES`. The loader (`adapter_loader.apply_adapter`):

1. Replaces `img_in` with a fresh `nn.Linear(768, 4096)` and copies in the
   adapter's `x_embedder.weight`. Replaces `final_layer.linear` similarly.
2. Overwrites `final_layer.adaLN_modulation.1.weight` from `norm_out.linear.weight`.
3. Registers `proj_buffer` (768 × 128) and `scale_buffer` (scalar) on the
   transformer.
4. For each of the 58 LoRA targets, fuses `delta = B @ A * scaling` into the
   existing weight (math done in fp32, then cast back).
5. Sets `transformer.patch_size = 16`, `in_channels = out_channels = 768`
   so `Flux.process_img` patchifies the pixel-resolution input correctly.

## Testing

Three test suites that run without ComfyUI:

```bash
python tests/test_key_translation.py    # 5/5 — ground truth: real 121-key dump
python tests/test_scheduler.py          # 5/5 — vs LakonLab reference math
python tests/test_apply_adapter.py      # 6/6 — surgery primitives
python tests/test_oklab.py              # 4/4 — color round-trips
```

## Known unknowns / things to verify on first real run

These can't be verified without the real 9 B base model + a GPU, so they're
flagged here for the first end-to-end run:

1. **`final_layer.adaLN_modulation.1` is assumed to be the right target for
   `norm_out.linear.weight`.** ComfyUI's `LastLayer` source wasn't inspectable
   via static fetch; the layout (`Sequential(SiLU, Linear)` with the linear at
   index 1) is the canonical FLUX layout but worth double-checking on the
   first error.
2. **The model's `process_img` may not respect the post-hoc `patch_size = 16`
   change cleanly** if it caches anything at construction time. If you see a
   shape mismatch in the first denoise step, the fix is to construct the
   `Flux` module with `patch_size=16, in_channels=3` from the start (i.e. a
   custom model-config registration) instead of patching it after.
3. **Positional embedding (`pe_embedder`) is shape-agnostic** in ComfyUI's
   `EmbedND`, but a 60×80 token grid at patch 16 has 4× the token count of
   the equivalent latent FLUX.2 run. Performance/memory will rise accordingly.
4. **The `scale_buffer` is registered but not yet consumed** by anything in
   this package — LakonLab's pipeline reads it inside `prepare_latents` for
   scaling the initial noise. If image contrast looks off on the first run,
   that's the likely culprit; the fix is to multiply the initial empty latent
   by `scale_buffer`.
5. **ComfyUI's `ModelPatcher` may dislike module replacement.** The loader
   does `unpatch_model()` before the surgery to keep its bookkeeping happy,
   but low-VRAM users may need to disable smart-offloading.

If any of these bite, the fixes are small and localized — file an issue with
the traceback.

## License

Code in this folder: Apache 2.0.
Adapter weights from `Lakonik/AsymFLUX.2-klein-9B`: see upstream license.
Base model `FLUX.2-klein-base-9B`: subject to Black Forest Labs' license.
