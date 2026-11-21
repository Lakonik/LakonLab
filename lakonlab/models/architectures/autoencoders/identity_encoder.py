# Copyright (c) 2026 Hansheng Chen

import torch
import torch.nn as nn

from ...builder import MODULES


@MODULES.register_module()
class IdentityEncoder(nn.Module):

    @property
    def dtype(self):
        return torch.float32

    @staticmethod
    def encode(value):
        return value

    @staticmethod
    def decode(value):
        return value
