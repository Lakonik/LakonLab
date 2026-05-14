# Copyright (c) 2026 Hansheng Chen

from typing import Any, Callable

import PIL
import torch

from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline, Flux2PipelineOutput, Flux2ImageProcessor)
from .utils import LakonLabMixin
from lakonlab.models.diffusions.gaussian_flow import guidance_jit
from lakonlab.models.diffusions.schedulers import FlowEulerODEScheduler
from lakonlab.models.architectures import OklabColorEncoder


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class PixelFlux2KleinPipeline(Flux2KleinPipeline, LakonLabMixin):

    model_cpu_offload_seq = "text_encoder->transformer"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowEulerODEScheduler,
        vae: OklabColorEncoder,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        transformer: Flux2Transformer2DModel,
        is_distilled: bool = False,
    ):
        super(Flux2KleinPipeline, self).__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )

        self.register_to_config(is_distilled=is_distilled)

        self.vae_scale_factor = 1
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 16)
        self.tokenizer_max_length = 512
        self.default_sample_size = 1024

    def prepare_latents(
        self,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ):
        height = 16 * (int(height) // (self.vae_scale_factor * 16))
        width = 16 * (int(width) // (self.vae_scale_factor * 16))

        shape = (batch_size, 3, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    def prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size,
        device,
        dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            imagge_latent = self.vae.encode(image).to(self.transformer.dtype)
            image_latents.append(imagge_latent.repeat(batch_size, 1, 1, 1))  # (bs, 3, 1024, 1024)
        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        image: list[PIL.Image.Image] | PIL.Image.Image | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        orthogonal_guidance: float = 1.0,
        clamp_denoised: bool = True,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: str | list[str] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
    ):

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
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

        # 3. prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            if prompt is not None and isinstance(prompt, list) and not isinstance(negative_prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
            guidance_scale = torch.tensor(guidance_scale, device=device, dtype=torch.float32)

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 16
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        if condition_images is not None:
            image_latents = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        image_seq_len = latents.shape[2:].numel()
        self.scheduler.set_timesteps(
            num_inference_steps, seq_len=image_seq_len, device=self._execution_device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                _t = t / 1000
                timestep = _t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(self.transformer.dtype)

                with self.transformer.cache_context("cond"):
                    denoising_output = self.transformer(
                        x_t=latent_model_input,  # (B, 3, H, W)
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        condition_latents=image_latents,
                        txt_ids=text_ids,  # B, text_seq_len, 4
                        guidance=None,
                        joint_attention_kwargs=self.attention_kwargs,
                    ).float()

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_denoising_output = self.transformer(
                            x_t=latent_model_input,  # (B, 3, H, W)
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            condition_latents=image_latents,
                            txt_ids=negative_text_ids,
                            guidance=None,
                            joint_attention_kwargs=self._attention_kwargs,
                        ).float()
                    cfg_bias = guidance_jit(
                        denoising_output,
                        neg_denoising_output,
                        guidance_scale,
                        orthogonal_guidance,
                        latents - denoising_output * _t)
                    denoising_output = denoising_output + cfg_bias

                if clamp_denoised:
                    denoised = latents - denoising_output * _t
                    image = self.vae.decode(denoised.to(self.vae.dtype)).clamp(-1, 1)
                    denoised = self.vae.encode(image).to(latents.dtype)
                    denoising_output = (latents - denoised) / _t.clamp(min=1e-4)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(denoising_output, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents.to(self.vae.dtype))
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
