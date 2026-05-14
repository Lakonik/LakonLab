# Modified from https://github.com/MizzenAI/HPSv3

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import mmcv

from typing import List, Optional, Union
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from accelerate import init_empty_weights
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from transformers.utils import TensorType
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock, Qwen2VLDecoderLayer
from mmcv.runner import get_dist_info
from lakonlab.runner.checkpoint import _load_checkpoint
from .builder import METRICS
from .metrics import Metric


INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Text Alignment:**  
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image: 
"""


def smart_resize(
        height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
            self,
            do_resize: bool = True,
            do_normalize: bool = True,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            min_pixels: int = 256 * 28 * 28,
            max_pixels: int = 256 * 28 * 28,
            patch_size: int = 14,
            temporal_patch_size: int = 2,
            merge_size: int = 2,
            **kwargs):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size

    def _preprocess(
            self,
            images: torch.Tensor,
            do_resize: bool = None,
            do_normalize: bool = None,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None):
        batch_size, channel, height, width = images.size()
        if do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            images = F.interpolate(
                images,
                size=(resized_height, resized_width),
                mode='bicubic',
                align_corners=False,
                antialias=True,
            ).clamp(min=0, max=1)
        else:
            resized_height, resized_width = height, width

        if do_normalize:
            mean = torch.tensor(image_mean, device=images.device, dtype=images.dtype).view(-1, 1, 1)
            std = torch.tensor(image_std, device=images.device, dtype=images.dtype).view(-1, 1, 1)
            images = (images - mean) / std

        patches = images.unsqueeze(1).expand(-1, self.temporal_patch_size, -1, -1, -1)

        grid_t = 1
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

        patches = patches.reshape(
            batch_size * grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            batch_size * grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return flatten_patches, np.array((grid_t, grid_h, grid_w)).reshape(1, 3).repeat(batch_size, axis=0)

    def preprocess(
            self,
            images: torch.Tensor,
            return_tensors: Optional[Union[str, TensorType]] = None):
        pixel_values, vision_grid_thws = self._preprocess(
            images,
            do_resize=self.do_resize,
            do_normalize=self.do_normalize,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}
        return BatchFeature(data=data, tensor_type=return_tensors)


class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):

    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
    ):
        super().__init__(config)
        self.output_dim = output_dim
        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.text_config.hidden_size, output_dim, bias=False)
        elif rm_head_type == "ranknet":
            if rm_head_kwargs is not None:
                for layer in range(rm_head_kwargs.get("num_layers", 3)):
                    if layer == 0:
                        self.rm_head = nn.Sequential(
                            nn.Linear(config.text_config.hidden_size, rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        )
                    elif layer < rm_head_kwargs.get("num_layers", 3) - 1:
                        self.rm_head.add_module(
                            f"layer_{layer}",
                            nn.Sequential(
                                nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ),
                        )
                    else:
                        self.rm_head.add_module(
                            f"output_layer",
                            nn.Linear(rm_head_kwargs["hidden_size"], output_dim, bias=rm_head_kwargs.get("bias", False)),
                        )

            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(config.text_config.hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )

        self.rm_head.to(torch.float32)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        # modified from the origin class Qwen2VLForConditionalGeneration
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.get_dtype())
                image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.model.visual.get_dtype())
                video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [B, L, D]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            logits = self.rm_head(hidden_states)  # [B, L, N]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # get sequence length
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        # get the last token's logits
        if self.reward_token == "last":
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]
        elif self.reward_token == "mean":
            # get the mean of all valid tokens' logits
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack(
                [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
            )
        elif self.reward_token == "special":
            # special_token_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens)
            # create a mask for special tokens
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (
                    input_ids == special_token_id
                )
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(
                batch_size, 1, -1
            )  # [B, 3, N] assert 3 attributes
            pooled_logits = pooled_logits.view(batch_size, -1)

            # pdb.set_trace()
        else:
            raise ValueError("Invalid reward_token")

        return {"logits": pooled_logits}


_hpsv3_cache = {}


def load_hpsv3(device, dtype, use_fsdp=True):
    # Create cache key from arguments
    cache_key = f"{device}_{dtype}_{use_fsdp}"

    # Check if model is already cached
    if cache_key in _hpsv3_cache:
        return _hpsv3_cache[cache_key]

    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-7B-Instruct', padding_side='right',
    )
    processor.image_processor = Qwen2VLImageProcessor()
    special_tokens = ['<|Reward|>']
    processor.tokenizer.add_special_tokens(
        {'additional_special_tokens': special_tokens}
    )
    special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    with init_empty_weights():
        config = Qwen2VLRewardModelBT.config_class.from_pretrained(
            'Qwen/Qwen2-VL-7B-Instruct',
        )
        model = Qwen2VLRewardModelBT(
            config,
            output_dim=2,
            reward_token='special',
            special_token_ids=special_token_ids,
            rm_head_type='ranknet',
        )
        model.requires_grad_(False)

    model.resize_token_embeddings(len(processor.tokenizer))

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    state_dict = _load_checkpoint(
        'huggingface://MizzenAI/HPSv3/HPSv3.safetensors', map_location='cpu'
    )
    new_state_dict = dict()
    for k, v in state_dict.items():  # fix transformers version mismatch
        if k.startswith('model.'):
            new_k = 'model.language_model.' + k[len('model.'):]
        elif k.startswith('visual.'):
            new_k = 'model.visual.' + k[len('visual.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=True, assign=True)
    model.rm_head.to(torch.float32)

    if use_fsdp:
        mmcv.print_log('Wrapping HPSv3 model with FSDP.')
        ignored_states = []
        for p in model.rm_head.parameters():
            p.data = p.data.cuda()
            ignored_states.append(p)
        model = FullyShardedDataParallel(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=False,
            mixed_precision=MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
                cast_root_forward_inputs=False),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            auto_wrap_policy=ModuleWrapPolicy([Qwen2VLVisionBlock, Qwen2VLDecoderLayer]),
            ignored_states=ignored_states
        )
    else:
        model.to(device)

    result = model, processor
    _hpsv3_cache[cache_key] = result
    return result


@METRICS.register_module()
class HPSv3(Metric):
    name = 'HPSv3'
    requires_prompt = True

    def __init__(self,
                 num_images=None,
                 use_fsdp=True):
        super().__init__(num_images)
        use_fsdp = use_fsdp and torch.cuda.is_available() and dist.is_initialized() and dist.get_world_size() > 0

        self.use_fsdp = use_fsdp
        self.dtype = torch.bfloat16
        self.device = 'cuda' if use_fsdp else 'cpu'

        self.model, self.processor = load_hpsv3(device=self.device, dtype=self.dtype, use_fsdp=use_fsdp)
        self.model.eval()

    def prepare(self):
        self.scores = []

    @torch.no_grad()
    def feed_op(self, batch, mode):
        imgs = batch['imgs']
        prompts = batch['prompts']

        imgs = (imgs.to(device=self.device, dtype=torch.float32) / 2 + 0.5).clamp(0, 1)

        message_list = []
        for text in prompts:
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "min_pixels": self.processor.image_processor.min_pixels,
                            "max_pixels": self.processor.image_processor.max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                            ),
                        },
                    ],
                }
            ]
            message_list.append(out_message)

        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=imgs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True})
        batch = {k: v.to(self.device) for k, v in batch.items()}
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"][:, 0]

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.empty_like(rewards) for _ in range(ws)]
            dist.all_gather(placeholder, rewards)
            rewards = torch.stack(placeholder, dim=1).reshape(rewards.size(0) * ws)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.scores.append(rewards.float().cpu())

    def feed(self, batch, mode):
        if mode == 'reals':
            return 0

        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()

            if self.num_fake_feeded == self.num_fake_need:
                return 0

            if isinstance(batch, dict):
                batch_size = len(list(batch.values())[0])
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = {k: v[:end] for k, v in batch.items()}
            else:
                batch_size = batch.shape[0]
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = batch[:end]

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    @torch.no_grad()
    def summary(self):
        scores = torch.cat(self.scores, dim=0)
        if self.num_images is not None:
            assert scores.shape[0] >= self.num_images
            scores = scores[:self.num_images]
        mean_score = scores.mean().item()
        self._result_dict = dict(hpsv3=mean_score)
        self._result_str = f'HPSv3: {mean_score:.4f}'
        return mean_score

    def clear_fake_data(self):
        self.scores = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()

    def load_to_gpu(self):
        if torch.cuda.is_available() and not isinstance(self.model, FullyShardedDataParallel):
            self.model.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        if not isinstance(self.model, FullyShardedDataParallel):
            self.model.cpu()
            self.device = 'cpu'
