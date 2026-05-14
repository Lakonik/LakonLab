# Copyright (c) 2026 Hansheng Chen

from typing import Dict, List, Optional, Union, Any, Callable
from functools import partial

import torch

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.utils import is_torch_xla_available
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline, FluxPipelineOutput)
from lakonlab.models.diffusions.schedulers.flow_map_sde import FlowMapSDEScheduler
from lakonlab.models.diffusions.piflow_policies import POLICY_CLASSES
from .utils import LakonLabMixin


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class PiFluxPipeline(FluxPipeline, LakonLabMixin):
    r"""
    The policy-based Flux pipeline for text-to-image generation.

    Reference: https://arxiv.org/abs/2510.14974

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMapSDEScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
        policy_type (`str`, *optional*, defaults to `"GMFlow"`):
            The type of flow policy to use. Currently supports `"GMFlow"` and `"DX"`.
        policy_kwargs (`Dict`, *optional*):
            Additional keyword arguments to pass to the policy class.
    """

    def __init__(
        self,
        scheduler: FlowMapSDEScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        policy_type: str = 'GMFlow',
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            text_encoder_2,
            tokenizer_2,
            transformer,
            image_encoder,
            feature_extractor
        )
        assert policy_type in POLICY_CLASSES, f'Invalid policy: {policy_type}. Supported policies are {list(POLICY_CLASSES.keys())}.'
        self.policy_type = policy_type
        self.policy_class = partial(
            POLICY_CLASSES[policy_type], **policy_kwargs
        ) if policy_kwargs else POLICY_CLASSES[policy_type]

    def _unpack_gm(self, gm, height, width, num_channels_latents, patch_size=2, gm_patch_size=1):
        c = num_channels_latents * patch_size * patch_size
        h = (int(height) // (self.vae_scale_factor * patch_size))
        w = (int(width) // (self.vae_scale_factor * patch_size))
        bs = gm['means'].size(0)
        k = self.transformer.num_gaussians
        scale = patch_size // gm_patch_size
        gm['means'] = gm['means'].reshape(
            bs, h, w, k, c // (scale * scale), scale, scale
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, c // (scale * scale), h * scale, w * scale)
        gm['logweights'] = gm['logweights'].reshape(
            bs, h, w, k, 1, scale, scale
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, 1, h * scale, w * scale)
        gm['logstds'] = gm['logstds'].reshape(bs, 1, 1, 1, 1)
        return gm

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, patch_size=1, target_patch_size=2):
        scale = target_patch_size // patch_size
        latents = latents.view(
            batch_size,
            num_channels_latents * patch_size * patch_size,
            height // target_patch_size, scale, width // target_patch_size, scale)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size,
            (height // target_patch_size) * (width // target_patch_size),
            num_channels_latents * target_patch_size * target_patch_size)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor, patch_size=2, target_patch_size=1):
        batch_size, num_patches, channels = latents.shape
        scale = patch_size // target_patch_size

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = (int(height) // (vae_scale_factor * patch_size))
        width = (int(width) // (vae_scale_factor * patch_size))

        latents = latents.view(
            batch_size, height, width, channels // (scale * scale), scale, scale)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (scale * scale), height * scale, width * scale)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        total_substeps: int = 128,
        temperature: Union[float, str] = 'auto',
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            total_substeps (`int`, *optional*, defaults to 128):
                The total number of substeps for policy-based flow integration.
            temperature (`float` or `"auto"`, *optional*, defaults to `"auto"`):
                The tmperature parameter for the flow policy.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare prompt embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        image_seq_len = latents.shape[1]
        self.scheduler.set_timesteps(num_inference_steps, seq_len=image_seq_len, device=self._execution_device)
        timesteps = self.scheduler.timesteps
        timesteps_dst = self.scheduler.timesteps_dst
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=self.num_timesteps) as progress_bar:
            for i, (t_src, t_dst) in enumerate(zip(timesteps, timesteps_dst)):
                if self.interrupt:
                    continue

                self._current_timestep = t_src
                time_scaling = self.scheduler.config.num_train_timesteps
                sigma_t_src = t_src / time_scaling
                sigma_t_dst = t_dst / time_scaling

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                with self.transformer.cache_context("cond"):
                    denoising_output = self.transformer(
                        hidden_states=latents.to(dtype=self.transformer.dtype),
                        timestep=t_src.expand(latents.shape[0]) / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                    )

                # unpack and create policy
                latents = self._unpack_latents(
                    latents, height, width, self.vae_scale_factor, target_patch_size=1)
                if self.policy_type == 'GMFlow':
                    denoising_output = self._unpack_gm(
                        denoising_output, height, width, num_channels_latents, gm_patch_size=1)
                    denoising_output = {k: v.to(torch.float32) for k, v in denoising_output.items()}
                    policy = self.policy_class(
                        denoising_output, latents, sigma_t_src)
                    if i < self.num_timesteps - 1:
                        if temperature == 'auto':
                            temperature = min(max(0.1 * (num_inference_steps - 1), 0), 1)
                        else:
                            assert isinstance(temperature, (float, int))
                        policy.temperature_(temperature)
                elif self.policy_type == 'DX':
                    denoising_output = denoising_output[0]
                    denoising_output = self._unpack_latents(
                        denoising_output, height, width, self.vae_scale_factor, target_patch_size=1)
                    denoising_output = denoising_output.reshape(latents.size(0), -1, *latents.shape[1:])
                    denoising_output = denoising_output.to(torch.float32)
                    policy = self.policy_class(
                        denoising_output, latents, sigma_t_src)
                else:
                    raise ValueError(f'Unknown policy type: {self.policy_type}.')

                raw_t_src = self.scheduler.unwarp_t(
                    sigma_t_src, seq_len=image_seq_len)
                raw_t_dst = self.scheduler.unwarp_t(
                    sigma_t_dst, seq_len=image_seq_len)
                latents_dst, _, _ = policy.integrate(
                    latents, sigma_t_src, raw_t_src, raw_t_dst, self.scheduler,
                    seq_len=image_seq_len, total_substeps=total_substeps)

                latents = self.scheduler.step(latents_dst, t_src, latents, return_dict=False)[0]

                # repack
                latents = self._pack_latents(
                    latents, latents.size(0), num_channels_latents,
                    2 * (int(height) // (self.vae_scale_factor * 2)),
                    2 * (int(width) // (self.vae_scale_factor * 2)),
                    patch_size=1)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t_src, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
