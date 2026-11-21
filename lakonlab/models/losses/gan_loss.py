# Modified from https://github.com/open-mmlab/mmgeneration

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODULES


@MODULES.register_module()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge',
            'wgan-logistic-ns', 'relativistic'. Sequence inputs are averaged
            across discriminator branches.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan-logistic-ns':
            self.loss = self._wgan_logistic_ns_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        elif self.gan_type == 'relativistic':
            self.loss = nn.Softplus()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_logistic_ns_loss(self, input, target):
        """WGAN loss in logistically non-saturating mode.

        This loss is widely used in StyleGANv2.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise, \
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan-logistic-ns']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, relativistic_input=None):
        """
        Args:
            input (Tensor | list[Tensor] | tuple[Tensor, ...]): Discriminator
                scores. For a nonempty list or tuple, losses are averaged
                across discriminator branches.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
            relativistic_input (Tensor | list[Tensor] | tuple[Tensor, ...] | None):
                Comparison scores required by the relativistic objective,
                matching the structure and tensor shapes of input. Ignored by
                the other objectives.

        Returns:
            Tensor: GAN loss value.
        """
        if isinstance(input, (tuple, list)):
            assert len(input) > 0
            return sum(
                self.forward(
                    value, target_is_real, is_disc,
                    None if relativistic_input is None else relativistic_input[index])
                for index, value in enumerate(input)) / len(input)

        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        elif self.gan_type == 'relativistic':
            assert relativistic_input is not None
            difference = input.float() - relativistic_input.float()
            loss = self.loss(-difference if target_is_real else difference).mean()
        else:  # other gan types
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
