# Copyright (c) 2026 Hansheng Chen

from abc import abstractmethod

from .builder import build_module
from .base import chunk_data_dict, get_grad_accum_steps, guess_bs
from .base_diffusion import BaseDiffusion, train_fwd_bwd
from lakonlab.parallel import maybe_ddp_no_sync
from lakonlab.utils import tie_untrained_submodules, module_requires_grad, module_eval, untie_all_parameters, clone_params


class BaseDiffusionSD(BaseDiffusion):

    def __init__(self,
                 *args,
                 critic=None,
                 tie_critic=False,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(
            *args, inference_only=inference_only, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        if not inference_only:
            assert critic is not None
            critic.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.critic = build_module(critic)
            if tie_critic:
                tie_untrained_submodules(self.critic, self.teacher, tie_tgt_lora_base_layer=True)
            untie_all_parameters(
                self.critic, self.teacher, untie_tgt_lora_base_layer=True, preserve_tied=tie_critic)
        else:
            self.critic = None

    @abstractmethod
    def _prepare_train_minibatch_args(self, data, running_status=None):
        """
        Prepare the arguments for the training step.

        Args:
            data (dict): The input data for the training step.
            running_status (dict): The running status for the training step.

        Returns:
            tuple: A tuple containing the batch size, diffusion arguments, and diffusion keyword arguments.
        """

    @abstractmethod
    def _prepare_train_minibatch_args_for_critic(self, data, running_status=None):
        """
        Prepare the arguments for the critic training step.

        Args:
            data (dict): The input data for the critic training step.
            running_status (dict): The running status for the critic training step.

        Returns:
            tuple: A tuple containing the batch size, critic arguments, and critic keyword arguments.
        """

    @abstractmethod
    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        """Perform a validation step.

        Args:
            data (dict): The input data for the validation step.
            test_cfg_override (dict): Override configuration for the test.

        Returns:
            dict: A dictionary containing the number of samples and predicted outputs.
        """

    def train_minibatch_critic(self, data, loss_scaler=None, running_status=None):
        bs, critic_args, critic_kwargs = self._prepare_train_minibatch_args_for_critic(data, running_status)
        log_vars = train_fwd_bwd(self.critic, critic_args, critic_kwargs, loss_scaler)
        return log_vars, bs

    def train_step(self, data, optimizer, loss_scaler=None, running_status=None):
        for v in optimizer.values():
            v.zero_grad()

        critic_optimizer = {k: v for k, v in optimizer.items() if k == 'critic'}
        generator_optimizer = {k: v for k, v in optimizer.items() if k != 'critic'}

        freeze_critic_for_generator = self.train_cfg.get('freeze_critic_for_generator', True)
        critic_extra_steps = self.train_cfg.get('critic_extra_steps', 1)

        total_bs = guess_bs(data)
        assert total_bs % (critic_extra_steps + 1) == 0, \
            f'Data batch size {total_bs} is not divisible by `critic_extra_steps + 1` {critic_extra_steps + 1}.'
        _bs = total_bs // (critic_extra_steps + 1)
        _data = chunk_data_dict(data, critic_extra_steps + 1)

        grad_accum_steps = get_grad_accum_steps(self.train_cfg, _bs)

        critic_log_vars = dict()
        for critic_step_id in range(critic_extra_steps):
            data_splits = chunk_data_dict(_data[critic_step_id], grad_accum_steps)
            critic_log_vars_single, bs = self.train_grad_accum(
                self.train_minibatch_critic,
                data_splits,
                critic_optimizer,
                grad_accum_steps,
                loss_scaler=loss_scaler,
                running_status=running_status)
            assert bs == _bs
            for k, v in critic_log_vars_single.items():
                if k in critic_log_vars:
                    critic_log_vars[k] += v
                else:
                    critic_log_vars[k] = v
            for v in critic_optimizer.values():
                v.zero_grad()

        data_splits = chunk_data_dict(_data[-1], grad_accum_steps)

        if freeze_critic_for_generator:
            assert critic_extra_steps > 0
            with (
                maybe_ddp_no_sync([self.critic]),
                module_requires_grad(self.critic, False),
                module_eval(self.teacher),
                module_eval(self.critic)
            ):
                log_vars, bs = self.train_grad_accum(
                    self.train_minibatch,
                    data_splits,
                    generator_optimizer,
                    grad_accum_steps,
                    loss_scaler=loss_scaler,
                    running_status=running_status)
                assert bs == _bs

        else:  # critic is updated within the generator step
            with module_eval(self.teacher):
                log_vars, bs = self.train_grad_accum(
                    self.train_minibatch,
                    data_splits,
                    optimizer,
                    grad_accum_steps,
                    loss_scaler=loss_scaler,
                    running_status=running_status)
                assert bs == _bs

        for k, v in critic_log_vars.items():
            if k in log_vars:
                log_vars[k] = (log_vars[k] + v) / (1 + critic_extra_steps)
            else:
                log_vars[k] = v / critic_extra_steps

        outputs_dict = dict(log_vars=log_vars, num_samples=bs)

        return outputs_dict
