# Copyright (c) 2026 Hansheng Chen

from typing import Dict, List, Optional, Tuple, Union

import torch

from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiTPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from lakonlab.models.architectures.gmflow.gmdit import _GMDiTTransformer2DModel as GMDiTTransformer2DModel
from lakonlab.models.architectures.gmflow.spectrum_mlp import _SpectrumMLP as SpectrumMLP
from lakonlab.models.diffusions.schedulers import FlowSDEScheduler, FlowEulerODEScheduler
from lakonlab.models.diffusions.gmflow import probabilistic_guidance_jit, GMFlowMixin
from lakonlab.ops.gmflow_ops.gmflow_ops import (
    gm_to_mean, iso_gaussian_mul_iso_gaussian, gm_mul_iso_gaussian, gm_to_iso_gaussian)


class GMDiTPipeline(DiTPipeline, GMFlowMixin):

    def __init__(
            self,
            transformer: GMDiTTransformer2DModel,
            spectrum_net: SpectrumMLP,
            vae: AutoencoderKL,
            scheduler: FlowSDEScheduler | FlowEulerODEScheduler,
            id2label: Optional[Dict[int, str]] = None):
        super(DiTPipeline, self).__init__()
        self.register_modules(transformer=transformer, spectrum_net=spectrum_net, vae=vae, scheduler=scheduler)

        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 0.45,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 32,
        num_inference_substeps: int = 4,
        output_mode: str = "mean",
        order=2,
        orthogonal_guidance: float = 1.0,
        gm2_coefs=[0.005, 1.0],
        gm2_correction_steps=0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        assert 0 <= guidance_scale < 1, "guidance_scale must be in [0, 1)"

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        x_t = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
        )
        use_guidance = guidance_scale > 0.0
        if use_guidance:
            guidance_scale = x_t.new_tensor(
                [guidance_scale]
            ).expand(batch_size).reshape([batch_size] + [1] * (x_t.dim() - 1))

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if use_guidance else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps * num_inference_substeps, device=self._execution_device)

        self.init_gm_cache()

        for timestep_id in self.progress_bar(range(num_inference_steps)):
            t = self.scheduler.timesteps[timestep_id * num_inference_substeps]

            x_t_input = x_t
            if use_guidance:
                x_t_input = torch.cat([x_t_input, x_t], dim=0)

            gm_output = self.transformer(
                x_t_input.to(dtype=self.transformer.dtype),
                timestep=t.expand(x_t_input.size(0)),
                class_labels=class_labels_input)
            gm_output = {k: v.to(torch.float32) for k, v in gm_output.items()}
            gm_output = self.u_to_x_0(gm_output, x_t_input, t)

            # ========== Probabilistic CFG ==========
            if use_guidance:
                gm_cond = {k: v[:batch_size] for k, v in gm_output.items()}
                gm_uncond = {k: v[batch_size:] for k, v in gm_output.items()}
                uncond_mean = gm_to_mean(gm_uncond)
                gaussian_cond = gm_to_iso_gaussian(gm_cond)[0]
                gaussian_cond['var'] = gaussian_cond['var'].mean(dim=(-1, -2), keepdim=True)
                gaussian_output, cfg_bias, avg_var = probabilistic_guidance_jit(
                    gaussian_cond['mean'], gaussian_cond['var'], uncond_mean, guidance_scale,
                    orthogonal=orthogonal_guidance)
                gm_output = gm_mul_iso_gaussian(
                    gm_cond, iso_gaussian_mul_iso_gaussian(gaussian_output, gaussian_cond, 1, -1),
                    1, 1)[0]
            else:
                gaussian_output = gm_to_iso_gaussian(gm_output)[0]
                gm_cond = gaussian_cond = avg_var = cfg_bias = None

            # ========== 2nd order GM ==========
            if order == 2:
                if timestep_id < num_inference_steps - 1:
                    h = t - self.scheduler.timesteps[(timestep_id + 1) * num_inference_substeps]
                else:
                    h = t
                gm_output, gaussian_output = self.gm_2nd_order(
                    gm_output, gaussian_output, x_t, t, h,
                    guidance_scale, gm_cond, gaussian_cond, avg_var, cfg_bias,
                    ca=gm2_coefs[0], cb=gm2_coefs[1], gm2_correction_steps=gm2_correction_steps)

            # ========== GM SDE step or GM ODE substeps ==========
            x_t_base = x_t
            t_base = t
            for substep_id in range(num_inference_substeps):
                if substep_id == 0:
                    if output_mode == 'sample':
                        power_spectrum = self.spectrum_net(gaussian_output)
                    else:
                        power_spectrum = None
                    model_output = self.gm_to_model_output(
                        gm_output, output_mode, power_spectrum=power_spectrum)
                else:
                    assert output_mode == 'mean'
                    t = self.scheduler.timesteps[timestep_id * num_inference_substeps + substep_id]
                    model_output = self.gmflow_posterior_mean(
                        gm_output, x_t, x_t_base, t, t_base, prediction_type='x0')
                x_t = self.scheduler.step(model_output, t, x_t, return_dict=False, prediction_type='x0')[0]

        x_t = x_t / self.vae.config.scaling_factor
        samples = self.vae.decode(x_t.to(self.vae.dtype)).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
