# Copyright (c) 2026 Hansheng Chen

import inspect
from typing import Callable

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3FL2VAPrepareLatentsStep, MiniMaxH3NoKeyframeAnchorsStep, MiniMaxH3PrepareConditionLatentsStep,
    MiniMaxH3PrepareLatentsStep, MiniMaxH3PrepareLayoutStep, MiniMaxH3Ref2VAPrepareLatentsStep,
    MiniMaxH3Ref2VAPrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)
from diffusers.modular_pipelines.minimax_h3.decoders import (
    MiniMaxH3AfterDenoiseStep, MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep)
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3DenoiseStep, MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep, MiniMaxH3Ref2VADenoiseStep)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoBeforeEncodeStep, MiniMaxH3AutoDenoiseStep, MiniMaxH3AutoTextEncoderStep,
    MiniMaxH3AutoVaeEncoderStep, MiniMaxH3Blocks, MiniMaxH3CoreDenoiseStep, MiniMaxH3DecodeStep,
    MiniMaxH3FL2VACoreDenoiseStep, MiniMaxH3Ref2VACoreDenoiseStep)
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.utils.torch_utils import randn_tensor


class MiniMaxH3PrepareConditionLatentsStepMod(MiniMaxH3PrepareConditionLatentsStep):
    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        device = components._execution_device
        patch_size = components.patch_size

        packed = []
        for condition in block_state.condition_latents:
            noise = randn_tensor(condition.shape, generator=block_state.generator, device=device, dtype=torch.float32)
            # This is the transformer's t rather than the scheduler's, so the native `scale_noise` is inlined.
            noised = components.keyframe_noise_aug * condition.to(device) + (1 - components.keyframe_noise_aug) * noise
            packed.append(patchify_video_latents(noised, patch_size))
        block_state.condition_rows = torch.cat(packed)
        if block_state.condition_rows.shape[0] != block_state.num_condition_video_rows:
            raise ValueError(
                f'The layout reserved {block_state.num_condition_video_rows} conditioning rows but the encoded '
                f'conditioning latents pack into {block_state.condition_rows.shape[0]}. The canvas the layout was '
                'built from and the one the conditioning was encoded at do not agree.'
            )

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3SetTimestepsStepMod(MiniMaxH3SetTimestepsStep):
    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        device = components._execution_device

        components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        components.audio_scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps

        block_state.row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in self.build_row_timesteps(
                    block_state.video_indices,
                    block_state.audio_indices,
                    block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows,
                    block_state.text_indices.numel(),
                    float(timestep),
                    float(audio_timestep),
                    min(float(timestep), 1 - components.keyframe_noise_aug),
                    0.0,
                )
            )
            for timestep, audio_timestep in zip(block_state.timesteps, block_state.audio_timesteps)
        ]

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3LoopDenoiserMod(MiniMaxH3LoopDenoiser):
    @property
    def inputs(self):
        return super().inputs + [InputParam(name='num_condition_video_rows', type_hint=int, default=0)]

    @property
    def intermediate_outputs(self):
        return super().intermediate_outputs + [
            OutputParam('features', type_hint=torch.Tensor),
            OutputParam('dwd_head', type_hint=Callable),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        transformer = getattr(components, self.transformer_name)
        # Only a DWD transformer's features are decoded, from the final step
        return_features = i == len(block_state.row_timestep_plan) - 1 and hasattr(transformer, 'dwd_linear_head')
        unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
        unique_timesteps = 1 - unique_timesteps
        # The class signature, since ahead-of-time compilation replaces the instance's forward
        layout_kwargs = {
            name: value
            for name, value in block_state.denoiser_input_fields.items()
            if name in inspect.signature(type(transformer).forward).parameters
        }
        output = transformer(
            hidden_states=block_state.latents[None].to(transformer.dtype),
            audio_hidden_states=block_state.audio_latents[None].to(transformer.dtype),
            encoder_hidden_states=block_state.prompt_embeds.to(transformer.dtype),
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            text_attention_kwargs=None,
            # sol_prefix: rows before the target video (text, conditions, audio), read by the Sol attention processor
            attention_kwargs=dict(
                block_state.attention_kwargs or {},
                sol_prefix=torch.tensor(
                    block_state.prompt_embeds.shape[1]
                    + block_state.num_condition_video_rows
                    + block_state.audio_latents.shape[0],
                    device='cpu'
                )),
            return_features=return_features,
            **layout_kwargs)
        block_state.features = block_state.dwd_head = None
        if return_features:
            output, features = output
            block_state.features = features['image']
            block_state.dwd_head = transformer.forward_dwd_head
        block_state.noise_pred, block_state.audio_noise_pred = -output[0], -output[1]
        return components, block_state


class MiniMaxH3Ref2VALoopDenoiserMod(MiniMaxH3LoopDenoiserMod):
    def __init__(self):
        super().__init__(transformer_name='transformer_ref')


class MiniMaxH3DenoiseStepMod(MiniMaxH3DenoiseStep):
    block_classes = [MiniMaxH3LoopDenoiserMod, MiniMaxH3LoopSchedulerStep]


class MiniMaxH3Ref2VADenoiseStepMod(MiniMaxH3Ref2VADenoiseStep):
    block_classes = [MiniMaxH3Ref2VALoopDenoiserMod, MiniMaxH3LoopSchedulerStep]


class MiniMaxH3DWDVideoDecodeStep(ModularPipelineBlocks):
    @property
    def inputs(self):
        return [
            InputParam(name='features', type_hint=torch.Tensor, required=True),
            InputParam(name='dwd_head', type_hint=Callable, required=True),
            InputParam(name='num_latent_frames', type_hint=int, required=True),
            InputParam(name='latent_height', type_hint=int, required=True),
            InputParam(name='latent_width', type_hint=int, required=True),
            InputParam(name='num_condition_video_rows', type_hint=int, default=0),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam('dwd_pixels', type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        features = block_state.features[:, block_state.num_condition_video_rows:]
        patch_t, patch_h, patch_w = components.patch_size
        block_state.dwd_pixels = block_state.dwd_head(features, (
            block_state.num_latent_frames // patch_t,
            block_state.latent_height // patch_h,
            block_state.latent_width // patch_w))
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3CoreDenoiseStepMod(MiniMaxH3CoreDenoiseStep):
    block_classes = [
        MiniMaxH3NoKeyframeAnchorsStep,
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3SetTimestepsStepMod,
        MiniMaxH3DenoiseStepMod,
        MiniMaxH3AfterDenoiseStep,
    ]
    block_names = [
        'no_keyframe_anchors',
        'prepare_layout',
        'prepare_latents',
        'set_timesteps',
        'denoise',
        'after_denoise',
    ]


class MiniMaxH3FL2VACoreDenoiseStepMod(MiniMaxH3FL2VACoreDenoiseStep):
    block_classes = [
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3PrepareConditionLatentsStepMod,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3FL2VAPrepareLatentsStep,
        MiniMaxH3SetTimestepsStepMod,
        MiniMaxH3DenoiseStepMod,
        MiniMaxH3AfterDenoiseStep,
    ]
    block_names = [
        'prepare_layout',
        'prepare_condition_latents',
        'prepare_latents',
        'prepare_latents_fl2va',
        'set_timesteps',
        'denoise',
        'after_denoise',
    ]


class MiniMaxH3Ref2VACoreDenoiseStepMod(MiniMaxH3Ref2VACoreDenoiseStep):
    block_classes = [
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3PrepareConditionLatentsStepMod,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3Ref2VAPrepareLatentsStep,
        MiniMaxH3SetTimestepsStepMod,
        MiniMaxH3Ref2VADenoiseStepMod,
        MiniMaxH3AfterDenoiseStep,
    ]
    block_names = [
        'prepare_layout',
        'prepare_condition_latents',
        'prepare_latents',
        'prepare_latents_ref2va',
        'set_timesteps',
        'denoise',
        'after_denoise',
    ]


class MiniMaxH3AutoDenoiseStepMod(MiniMaxH3AutoDenoiseStep):
    block_classes = [MiniMaxH3Ref2VACoreDenoiseStepMod, MiniMaxH3FL2VACoreDenoiseStepMod, MiniMaxH3CoreDenoiseStepMod]


class MiniMaxH3VideoDecodeStepMod(MiniMaxH3VideoDecodeStep):
    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        device = components._execution_device

        if block_state.output_type not in ('pil', 'np', 'pt'):
            raise ValueError(
                f"`output_type` must be one of 'pil', 'np' or 'pt', got {block_state.output_type!r}. To keep the "
                'latents instead of decoding them, run a pipeline that does not include the decode blocks.'
            )

        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = block_state.latents * latents_std + latents_mean

        video = components.vae.decode(latents, return_dict=False)[0]
        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3DecodeStepMod(MiniMaxH3DecodeStep):
    block_classes = [MiniMaxH3VideoDecodeStepMod, MiniMaxH3AudioDecodeStep]


class MiniMaxH3DWDDecodeStep(MiniMaxH3DecodeStep):
    block_classes = [MiniMaxH3DWDVideoDecodeStep, MiniMaxH3AudioDecodeStep]


class MiniMaxH3BlocksMod(MiniMaxH3Blocks):
    block_classes = [
        MiniMaxH3AutoBeforeEncodeStep,
        MiniMaxH3AutoTextEncoderStep,
        MiniMaxH3AutoVaeEncoderStep,
        MiniMaxH3AutoDenoiseStepMod,
        MiniMaxH3DecodeStepMod,
    ]


class MiniMaxH3DWDBlocks(MiniMaxH3BlocksMod):
    block_classes = [
        MiniMaxH3AutoBeforeEncodeStep,
        MiniMaxH3AutoTextEncoderStep,
        MiniMaxH3AutoVaeEncoderStep,
        MiniMaxH3AutoDenoiseStepMod,
        MiniMaxH3DWDDecodeStep,
    ]
