# Copyright (c) 2026 Hansheng Chen

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial

import PIL
import torch

from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from diffusers.utils import is_torch_xla_available
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, Flux2PipelineOutput
from lakonlab.models.diffusions.schedulers.flow_map_sde import FlowMapSDEScheduler
from lakonlab.models.diffusions.piflow_policies import POLICY_CLASSES
from .utils import LakonLabMixin


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class PiFlux2Pipeline(Flux2Pipeline, LakonLabMixin):
    r"""
    The policy-based Flux2 pipeline for text-to-image generation.

    Reference:
        https://arxiv.org/abs/2510.14974
        https://bfl.ai/blog/flux-2

    Args:
        transformer ([`Flux2Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMapSDEScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLFlux2`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Mistral3ForConditionalGeneration`]):
            [Mistral3ForConditionalGeneration](https://huggingface.co/docs/transformers/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration)
        tokenizer (`AutoProcessor`):
            Tokenizer of class
            [PixtralProcessor](https://huggingface.co/docs/transformers/en/model_doc/pixtral#transformers.PixtralProcessor).
        policy_type (`str`, *optional*, defaults to `"GMFlow"`):
            The type of flow policy to use. Currently supports `"GMFlow"` and `"DX"`.
        policy_kwargs (`Dict`, *optional*):
            Additional keyword arguments to pass to the policy class.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMapSDEScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        transformer: Flux2Transformer2DModel,
        policy_type: str = 'GMFlow',
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            transformer,
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

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        total_substeps: int = 128,
        temperature: Union[float, str] = 'auto',
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            total_substeps (`int`, *optional*, defaults to 128):
                The total number of substeps for policy-based flow integration.
            temperature (`float` or `"auto"`, *optional*, defaults to `"auto"`):
                The tmperature parameter for the flow policy.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
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
            text_encoder_out_layers (`Tuple[int]`):
                Layer indices to use in the `text_encoder` to derive the final prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.flux2.Flux2PipelineOutput`] or `tuple`: [`~pipelines.flux2.Flux2PipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
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

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        image_seq_len = latents.shape[1]
        self.scheduler.set_timesteps(num_inference_steps, seq_len=image_seq_len, device=self._execution_device)
        timesteps = self.scheduler.timesteps
        timesteps_dst = self.scheduler.timesteps_dst
        self._num_timesteps = len(timesteps)

        # handle guidance
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t_src, t_dst) in enumerate(zip(timesteps, timesteps_dst)):
                if self.interrupt:
                    continue

                self._current_timestep = t_src
                time_scaling = self.scheduler.config.num_train_timesteps
                sigma_t_src = t_src / time_scaling
                sigma_t_dst = t_dst / time_scaling

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                denoising_output = self.transformer(
                    hidden_states=latent_model_input,  # (B, image_seq_len, C)
                    timestep=t_src.expand(latents.shape[0]) / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=latent_image_ids,  # B, image_seq_len, 4
                    joint_attention_kwargs=self._attention_kwargs,
                )

                denoising_output = {
                    k: v[:, :latents.size(1)].to(torch.float32) for k, v in denoising_output.items()}

                # unpack and create policy
                latents = self._unpack_latents_with_ids(latents, latent_ids)
                latents = self._unpatchify_latents(latents)
                if self.policy_type == 'GMFlow':
                    denoising_output = self._unpack_gm(
                        denoising_output, height, width, num_channels_latents, gm_patch_size=1)
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
                    denoising_output = self._unpack_latents_with_ids(denoising_output, latent_ids)
                    denoising_output = self._unpatchify_latents(denoising_output)
                    denoising_output = denoising_output.reshape(latents.size(0), -1, *latents.shape[1:])
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
                latents = self._patchify_latents(latents)
                latents = self._pack_latents(latents)

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
            torch.save({"pred": latents}, "pred_d.pt")
            latents = self._unpack_latents_with_ids(latents, latent_ids)

            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean
            latents = self._unpatchify_latents(latents)

            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
