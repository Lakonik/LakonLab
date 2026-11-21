# Copyright (c) 2026 Hansheng Chen

from abc import abstractmethod
from contextlib import ExitStack
from copy import deepcopy

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map
from accelerate import init_empty_weights

from .builder import build_module
from .base import BaseModel
from lakonlab.utils import (
    clone_params, is_warmup, module_eval, module_requires_grad, rgetattr,
    tie_untrained_submodules, untie_all_parameters)
from lakonlab.parallel import maybe_ddp_no_sync


def train_fwd_bwd(
        model, args, kwargs, loss_scaler=None, discriminator=None, disable_g_loss=False):
    if (isinstance(discriminator, DistributedDataParallel)
            and discriminator.static_graph):
        raise RuntimeError(
            'The discriminator must not be DDP-wrapped with static_graph=True: '
            'the generator-side discriminator forward requires no_sync(), '
            'which is incompatible with DDP static-graph recording. Disable '
            'static_graph in ddp_kwargs.')

    is_multistep = rgetattr(model, 'is_multistep', False)

    if is_multistep:
        initialize_multistep = rgetattr(model, 'initialize_multistep')
        step_states, log_vars = initialize_multistep(*args, **kwargs)
        loss = 0

        while not step_states.finished:
            # DDP decides gradient sync at FORWARD time (`require_backward_grad_sync`
            # is read in DDP's pre/post-forward), so gradient-accumulation no_sync
            # must wrap the forward, not just the backward. It requires knowing the
            # last step up front (num_steps) and per-step backwards (detachable);
            # dynamic-termination models (num_steps=None) sync every step.
            # Incompatible setups: DDP static_graph (skipped with a warning by
            # maybe_ddp_no_sync) and cudagraphs (compile mode='reduce-overhead') --
            # unsynced backwards leave cudagraph pool tensors live across graph
            # replays (pytorch#169545, no upstream fix), so multistep models must
            # compile without cudagraphs.
            skip_sync = (
                step_states.detachable
                and step_states.num_steps is not None
                and step_states.step_id < step_states.num_steps - 1)
            with maybe_ddp_no_sync([model], enabled=skip_sync), ExitStack() as g_context:
                step_loss, step_log_vars, step_states = model(
                    *args, return_loss=True, step_states=step_states, **kwargs)

                use_gan = step_states.fakes is not None and step_states.reals is not None and discriminator is not None
                if use_gan:
                    fakes = step_states.fakes
                    reals = tree_map(
                        lambda value: value.detach() if isinstance(value, torch.Tensor) else value, step_states.reals)
                    if not disable_g_loss:
                        # The G-side traversal is frozen and unsynced. Whole-wrapper
                        # eval prevents network state updates; configured input
                        # augmentation remains active by discriminator contract.
                        # Keep the freeze through backward, including FSDP's recreation of parameter views.
                        g_context.enter_context(maybe_ddp_no_sync([discriminator]))
                        g_context.enter_context(module_requires_grad(discriminator, False))
                        g_context.enter_context(module_eval(discriminator))
                        # Optional real inputs provide reference features for feature matching loss.
                        loss_gen, g_log_vars = discriminator(return_g_loss=True, reals=reals, fakes=fakes)
                        step_loss = step_loss + loss_gen
                        step_log_vars.update(g_log_vars)

                if step_states.detachable:
                    step_loss.backward() if loss_scaler is None else loss_scaler.scale(step_loss).backward()
                    step_loss.detach_()

            loss = loss + step_loss

            if use_gan:
                fakes = tree_map(
                    lambda value: value.detach() if isinstance(value, torch.Tensor) else value, fakes)
                loss_disc, d_log_vars = discriminator(return_d_loss=True, reals=reals, fakes=fakes)
                loss_disc.backward() if loss_scaler is None else loss_scaler.scale(loss_disc).backward()
                log_vars.update(d_log_vars)

            # per-step GAN outputs are consumed; clear them for the next step
            step_states.fakes = step_states.reals = None

            for k, v in step_log_vars.items():
                if k in log_vars:
                    log_vars[k] += v
                else:
                    log_vars[k] = v

            step_states.step_id += 1

    else:
        loss, log_vars = model(*args, return_loss=True, **kwargs)

    if isinstance(loss, torch.Tensor) and loss.requires_grad:
        loss.backward() if loss_scaler is None else loss_scaler.scale(loss).backward()

    return log_vars


class BaseDiffusion(BaseModel):
    """Base class providing the common training interface for diffusion models. Optionally supports:
    - Discriminator for adversarial training
    - Teacher models for distillation training
    - EMA version of the diffusion model
    - Multi-step diffusion training
    - Image/video patching for patch-wise GMFlow
    """

    def __init__(self,
                 diffusion=dict(type='GaussianFlow'),
                 diffusion_use_ema=False,
                 tie_ema=True,
                 teacher=None,
                 tie_teacher=False,
                 teacher_2=None,
                 tie_teacher_2=False,
                 discriminator=None,
                 patch_size=1,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.patch_size = patch_size

        # order matters: teacher must be built before diffusion for FSDP tying
        if teacher is not None and not inference_only:
            teacher.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.teacher = build_module(teacher)
        else:
            self.teacher = None

        if teacher_2 is not None and not inference_only:
            teacher_2.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.teacher_2 = build_module(teacher_2)
            if self.teacher is not None:
                if tie_teacher_2:
                    tie_untrained_submodules(self.teacher_2, self.teacher, tie_tgt_lora_base_layer=True)
                untie_all_parameters(
                    self.teacher_2, self.teacher, untie_tgt_lora_base_layer=True, preserve_tied=tie_teacher_2)
            self.teacher_2.requires_grad_(False)
        else:
            self.teacher_2 = None

        diffusion = deepcopy(diffusion)
        diffusion.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.diffusion = build_module(diffusion)
        if self.teacher is not None:
            if tie_teacher:
                tie_untrained_submodules(self.diffusion, self.teacher, tie_tgt_lora_base_layer=True)
            untie_all_parameters(
                self.diffusion, self.teacher, untie_tgt_lora_base_layer=True, preserve_tied=tie_teacher)

        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            if inference_only:
                self.diffusion_ema = self.diffusion
            else:
                diffusion_ema = deepcopy(diffusion)
                if isinstance(diffusion_ema.get('denoising', None), dict):
                    pretrained_keys = [k for k in diffusion_ema['denoising'].keys() if k.startswith('pretrained')]
                    for k in pretrained_keys:
                        diffusion_ema['denoising'].pop(k)
                with init_empty_weights():
                    self.diffusion_ema = build_module(diffusion_ema)
                if tie_ema:
                    tie_untrained_submodules(self.diffusion_ema, self.diffusion)
                untie_all_parameters(self.diffusion_ema, self.diffusion, preserve_tied=tie_ema)
                clone_params(self.diffusion_ema, self.diffusion)

        if discriminator is not None and not inference_only:
            discriminator.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None

        self.train_cfg = dict() if train_cfg is None else deepcopy(train_cfg)
        self.test_cfg = dict() if test_cfg is None else deepcopy(test_cfg)

    def patchify(self, x):
        if isinstance(self.patch_size, int) and self.patch_size == 1:
            return x
        if x.dim() == 4:
            if isinstance(self.patch_size, int):
                ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 2
                ph, pw = self.patch_size
            bs, c, h, w = x.size()
            x = x.reshape(
                bs, c, h // ph, ph, w // pw, pw
            ).permute(
                0, 1, 3, 5, 2, 4
            ).reshape(
                bs, c * ph * pw, h // ph, w // pw)
        elif x.dim() == 5:
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 3
                pt, ph, pw = self.patch_size
            bs, c, t, h, w = x.size()
            x = x.reshape(
                bs, c, t // pt, pt, h // ph, ph, w // pw, pw
            ).permute(
                0, 1, 3, 5, 7, 2, 4, 6
            ).reshape(
                bs, c * pt * ph * pw, t // pt, h // ph, w // pw)
        else:
            raise ValueError(f'Unsupported input dimension {x.dim()}. Expected 4 or 5 dimensions.')
        return x

    def unpatchify(self, x):
        if isinstance(self.patch_size, int) and self.patch_size == 1:
            return x
        if x.dim() == 4:
            if isinstance(self.patch_size, int):
                ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 2
                ph, pw = self.patch_size
            bs, c, h, w = x.size()
            x = x.reshape(
                bs, c // (ph * pw), ph, pw, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c // (ph * pw), h * ph, w * pw)
        elif x.dim() == 5:
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 3
                pt, ph, pw = self.patch_size
            bs, c, t, h, w = x.size()
            x = x.reshape(
                bs, c // (pt * ph * pw), pt, ph, pw, t, h, w
            ).permute(
                0, 1, 5, 2, 6, 3, 7, 4
            ).reshape(
                bs, c // (pt * ph * pw), t * pt, h * ph, w * pw)
        else:
            raise ValueError(f'Unsupported input dimension {x.dim()}. Expected 4 or 5 dimensions.')
        return x

    @abstractmethod
    def _prepare_train_minibatch_args(self, data, running_status=None):
        """
        Prepare the arguments for the training minibatch.

        Args:
            data (dict): The input data for the training step.
            running_status (dict): The running status for the training step.

        Returns:
            tuple: A tuple containing the batch size, diffusion arguments, and diffusion keyword arguments.
        """

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        bs, diffusion_args, diffusion_kwargs = self._prepare_train_minibatch_args(data, running_status)
        discriminator = self.discriminator
        disable_g_loss = False
        if discriminator is not None and not is_warmup():
            iteration = 0 if running_status is None else running_status['iteration']
            if iteration < self.train_cfg.get('gan_start_iter', 0):
                discriminator = None
            # Optional discriminator-only warmup before the generator sees the GAN loss.
            disable_g_loss = iteration < self.train_cfg.get('g_loss_start_iter', 0)

        log_vars = train_fwd_bwd(
            self.diffusion, diffusion_args, diffusion_kwargs, loss_scaler,
            discriminator=discriminator, disable_g_loss=disable_g_loss)
        return log_vars, bs

    def _get_clamp_denoised_callback(self):
        vae = getattr(self, 'vae_2', None)
        if vae is None:
            vae = self.vae
        assert vae is not None, 'VAE must be provided for clamp_denoised sampling.'

        def clamp_denoised_callback(diffusion, callback_kwargs):
            x_t = callback_kwargs['x_t']
            u = callback_kwargs['u']
            t = callback_kwargs['t']

            denoised = diffusion.u_to_x_0(u, x_t, t=t)
            denoised = self.unpatchify(denoised)
            if hasattr(vae, 'dtype'):
                vae_dtype = vae.dtype
            else:
                vae_dtype = next(vae.parameters()).dtype
            image = vae.decode(denoised.to(vae_dtype)).clamp(-1, 1)
            denoised = vae.encode(image)
            denoised = self.patchify(denoised).to(device=x_t.device, dtype=x_t.dtype)

            u = diffusion.x_0_to_u(denoised, x_t, t=t)
            return dict(u=u)

        return clamp_denoised_callback

    @abstractmethod
    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        """Perform a validation step.

        Args:
            data (dict): The input data for the validation step.
            test_cfg_override (dict): Override configuration for the test.

        Returns:
            dict: A dictionary containing the number of samples and predicted outputs.
        """
