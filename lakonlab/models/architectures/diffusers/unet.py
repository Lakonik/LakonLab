from typing import Dict, Any, Optional, Union, Tuple
from collections import OrderedDict

import torch
import torch.nn.functional as F

from diffusers.models import UNet2DConditionModel as _UNet2DConditionModel
from mmcv.runner import _load_checkpoint, load_state_dict

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger


def ceildiv(a, b):
    return -(a // -b)


def unet_enc(
        unet,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs=None):
    # 0. center input if necessary
    if unet.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    emb = unet.time_embedding(t_emb)
    aug_emb = unet.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    emb = emb + aug_emb if aug_emb is not None else emb

    if unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)

    encoder_hidden_states = unet.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

    # 2. pre-process
    sample = unet.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    return emb, down_block_res_samples, sample


def unet_dec(
        unet,
        emb,
        down_block_res_samples,
        sample,
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None):
    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if unet.mid_block is not None:
        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            sample = unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample = unet.mid_block(sample, emb)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(unet.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
            )

    # 6. post-process
    if unet.conv_norm_out:
        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    return sample


@MODULES.register_module()
class UNet2DConditionModel(_UNet2DConditionModel):
    def __init__(self,
                 *args,
                 freeze=True,
                 freeze_exclude=[],
                 pretrained=None,
                 torch_dtype='float32',
                 freeze_exclude_fp32=True,
                 freeze_exclude_autocast_dtype='float32',
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.init_weights(pretrained)
        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

    def init_weights(self, pretrained):
        if pretrained is not None:
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            metadata = getattr(state_dict, '_metadata', OrderedDict())
            state_dict._metadata = metadata
            assert self.conv_in.weight.shape[1] == self.conv_out.weight.shape[0]
            if state_dict['conv_in.weight'].size() != self.conv_in.weight.size():
                assert state_dict['conv_in.weight'].shape[1] == state_dict['conv_out.weight'].shape[0]
                src_chn = state_dict['conv_in.weight'].shape[1]
                tgt_chn = self.conv_in.weight.shape[1]
                assert src_chn < tgt_chn
                convert_mat_out = torch.tile(torch.eye(src_chn), (ceildiv(tgt_chn, src_chn), 1))
                convert_mat_out = convert_mat_out[:tgt_chn]
                convert_mat_in = F.normalize(convert_mat_out.pinverse(), dim=-1)
                state_dict['conv_out.weight'] = torch.einsum(
                    'ts,scxy->tcxy', convert_mat_out, state_dict['conv_out.weight'])
                state_dict['conv_out.bias'] = torch.einsum(
                    'ts,s->t', convert_mat_out, state_dict['conv_out.bias'])
                state_dict['conv_in.weight'] = torch.einsum(
                    'st,csxy->ctxy', convert_mat_in, state_dict['conv_in.weight'])
            load_state_dict(self, state_dict, logger=logger)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        dtype = sample.dtype
        return super().forward(
            sample, timestep, encoder_hidden_states, return_dict=False, **kwargs)[0].to(dtype)

    def forward_enc(self, sample, timestep, encoder_hidden_states, **kwargs):
        return unet_enc(self, sample, timestep, encoder_hidden_states, **kwargs)

    def forward_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs):
        return unet_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs)
