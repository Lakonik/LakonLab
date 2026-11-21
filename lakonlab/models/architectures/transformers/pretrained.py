import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Dinov2Model, DINOv3ViTModel, T5EncoderModel, ViTModel
from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoderModel
from transformers.utils.output_capturing import maybe_install_capturing_hooks

from ...builder import MODULES
from lakonlab.runner.checkpoint import _load_cached_checkpoint
from lakonlab.utils import get_root_logger, is_warmup
from lakonlab.utils.io_utils import hf_model_loader


@MODULES.register_module()
class PretrainedDinoV2(nn.Module):
    def __init__(self,
                 model_name_or_path='facebook/dinov2-base',
                 image_size=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='reduce-overhead',
                     fullgraph=True,
                     dynamic=False),
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.model = hf_model_loader(Dinov2Model, model_name_or_path, **kwargs)
        self.image_size = image_size
        self.freeze = freeze
        self.eval_mode = eval_mode
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.resize_position_embeddings()
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def embed_dim(self):
        return self.model.config.hidden_size

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def resize_position_embeddings(self):
        position_embeddings = self.model.embeddings.position_embeddings
        num_positions = position_embeddings.shape[1] - 1
        new_size = self.image_size // self.model.config.patch_size
        if num_positions == new_size * new_size:
            return

        class_pos_embed = position_embeddings[:, :1]
        patch_pos_embed = position_embeddings[:, 1:]
        old_size = int(num_positions ** 0.5)
        dim = patch_pos_embed.shape[-1]
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = patch_pos_embed.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.float(),
            size=(new_size, new_size),
            mode='bicubic',
            antialias=True  # REPA uses antialiasing in timm.layers.pos_embed.resample_abs_pos_embed
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        self.model.embeddings.position_embeddings = nn.Parameter(
            torch.cat((class_pos_embed, patch_pos_embed), dim=1))

    def preprocess(self, images):
        images = images.float() / 2 + 0.5
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            ).clamp(min=0, max=1)
        images = (images - self.mean) / self.std
        return images

    def _forward_impl(self, images):
        pixel_values = self.preprocess(images).to(self.dtype)
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        hidden_states = outputs.last_hidden_state
        num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)
        return hidden_states[:, 1 + num_register_tokens:]

    def forward(self, images):
        if self._compiled_forward is not None:
            return self._compiled_forward(images).clone()
        return self._forward_impl(images)


@MODULES.register_module()
class PretrainedViT(nn.Module):
    """Frozen transformers ViT exposing tapped block features (e.g. DINO ViT-S/8).

    Expects (B, 3, H, W) images in [-1, 1] (resized and normalized
    internally). Returns one (B, C, L) feature map per ``hidden_state_ids``
    entry, with the class token dropped. Index 0 is the pre-transformer patch
    embedding output and index i + 1 is the output of transformer block i. The
    default taps blocks 2, 5, 8 and 11 as in the StyleGAN-T DINO discriminator,
    but omits its pre-transformer ``hook_patch`` tap (index 0), following RAE.

    ``checkpoint_chunk_size`` bounds activation memory at that many images,
    which is otherwise the dominant cost of scoring video (every block's
    activations are kept for the whole batch of frames and crops). The taps are selected inside the
    checkpointed region, so the untapped block outputs are never retained.
    Incompatible with cudagraphs (``mode='reduce-overhead'``): the chunks
    replay the same graph, so earlier chunks' outputs would be clobbered from
    the graph pool before they are concatenated.
    """

    def __init__(self,
                 model_name_or_path='facebook/dino-vits8',
                 image_size=224,
                 hidden_state_ids=(3, 6, 9, 12),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 checkpoint_chunk_size=None,
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.model = hf_model_loader(ViTModel, model_name_or_path, add_pooling_layer=False, **kwargs)
        self.image_size = image_size
        self.hidden_state_ids = tuple(hidden_state_ids)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self._compiled_forward = None
        if compile_forward:
            # Transformers' lazy hidden-state hook installation cannot run inside fullgraph compile.
            maybe_install_capturing_hooks(self.model)
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def num_taps(self):
        return len(self.hidden_state_ids)

    @property
    def config(self):
        return self.model.config

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def preprocess(self, images):
        images = images.float() / 2 + 0.5
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            ).clamp(min=0, max=1)
        images = (images - self.mean) / self.std
        return images

    def _forward_impl(self, images):
        pixel_values = self.preprocess(images).to(self.dtype)
        hidden_states = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            interpolate_pos_encoding=True,
            return_dict=True).hidden_states
        taps = [hidden_states[i] for i in self.hidden_state_ids]
        return [tap[:, 1:].transpose(1, 2) for tap in taps]

    def forward(self, images):
        forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
        cs = self.checkpoint_chunk_size
        if cs is None or not torch.is_grad_enabled():
            return forward_fn(images)
        else:
            return [torch.cat(tap_chunks) for tap_chunks in zip(*[
                torch.utils.checkpoint.checkpoint(
                    forward_fn, images[i:i + cs], use_reentrant=False)
                for i in range(0, images.size(0), cs)])]


@MODULES.register_module()
class PretrainedRAEv2Encoder(nn.Module):
    """Frozen DINOv3 multi-layer-sum encoder (RAEv2 'dinov3mls').

    Averages the final-norm outputs of the blocks in ``layer_ids`` and adds
    the last selected layer's global mean token. Expects (B, 3, H, W) images
    in [-1, 1]; returns a (B, C, h, w) latent map, normalized with the RAEv2
    latent statistics when ``pretrained_latent_stat`` is given.
    """

    def __init__(self,
                 model_name_or_path='facebook/dinov3-vitl16-pretrain-lvd1689m',
                 layer_ids=(11, 13, 15, 17, 19, 21, 23),
                 norm_affine=False,
                 resolution=256,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 pretrained_latent_stat=None,
                 eps=1e-5,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='reduce-overhead',
                     fullgraph=True,
                     dynamic=False),
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.model = hf_model_loader(DINOv3ViTModel, model_name_or_path, **kwargs)
        if not norm_affine:
            self.model.norm = nn.LayerNorm(self.model.config.hidden_size, elementwise_affine=False)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self.layer_ids = tuple(layer_ids)
        self.patch_size = self.model.config.patch_size
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.eps = eps
        self.do_normalization = pretrained_latent_stat is not None
        if pretrained_latent_stat is not None:
            stats = _load_cached_checkpoint(
                pretrained_latent_stat, map_location='cpu', logger=get_root_logger())
            self.register_buffer('latent_mean', stats['mean'].float(), persistent=False)
            self.register_buffer('latent_var', stats['var'].float(), persistent=False)
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self._compiled_forward = None
        if compile_forward:
            # Transformers' lazy hidden-state hook installation cannot run inside fullgraph compile.
            maybe_install_capturing_hooks(self.model)
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def _forward_impl(self, img):
        img = img.float() / 2 + 0.5
        if img.shape[-2:] != self.resolution:
            img = F.interpolate(
                img, size=self.resolution, mode='bicubic',
                align_corners=False, antialias=True)
        latent_size = (
            img.shape[-2] // self.patch_size,
            img.shape[-1] // self.patch_size)
        img = ((img.clamp(0, 1) - self.mean) / self.std).to(self.dtype)
        output = self.model(img, output_hidden_states=True, return_dict=True)
        num_register_tokens = self.model.config.num_register_tokens
        outputs = [
            self.model.norm(output.hidden_states[i + 1])[:, 1 + num_register_tokens:]
            for i in self.layer_ids
        ]
        z = torch.stack(outputs, dim=0).mean(dim=0)
        z = z + outputs[-1].mean(dim=1, keepdim=True)
        b, n, c = z.shape
        h, w = latent_size
        if n != h * w:
            raise ValueError(f'Expected {h * w} encoder patches, got {n}.')
        z = z.transpose(1, 2).reshape(b, c, h, w)
        if self.do_normalization:
            z = (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return z

    def encode(self, img):
        if self._compiled_forward is not None:
            return self._compiled_forward(img).clone()
        return self._forward_impl(img)

    def forward(self, img):
        return self.encode(img)


@MODULES.register_module()
class PretrainedT5TextEncoder(nn.Module):
    """Frozen T5 encoder (e.g. FLAN-T5-Large for MMJiT), padded to a fixed length.

    The returned mask marks the real tokens, which `MMJiT` needs in order to substitute its
    learned `mask_token` at padded positions. `fill_zeros` zeroes those positions, whose
    activations are otherwise as large as a real token's.
    """

    def __init__(self,
                 model_name_or_path='google/flan-t5-large',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=256,
                 fill_zeros=True,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.fill_zeros = fill_zeros
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.text_encoder = hf_model_loader(
            T5EncoderModel,
            model_name_or_path,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt):
        tokens = self.tokenizer(
            [prompt] if isinstance(prompt, str) else list(prompt),
            padding=True if self.fill_zeros else 'max_length',
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors='pt')
        device = next(self.text_encoder.parameters()).device
        attention_mask = tokens.attention_mask.to(device)
        prompt_embeds = self.text_encoder(
            input_ids=tokens.input_ids.to(device), attention_mask=attention_mask).last_hidden_state
        if self.fill_zeros:
            prompt_embeds = prompt_embeds * attention_mask[..., None].to(prompt_embeds.dtype)
            pad = self.max_sequence_length - prompt_embeds.shape[1]
            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad))
            attention_mask = F.pad(attention_mask, (0, pad))
        return dict(
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=attention_mask)


class _T5GemmaEncoderModel(T5GemmaEncoderModel):
    # The encoder-only model intentionally skips the released checkpoint's decoder.
    _keys_to_ignore_on_load_unexpected = [r'^model\.decoder\.']


@MODULES.register_module()
class PretrainedT5GemmaTextEncoder(nn.Module):
    """Frozen T5Gemma encoder (the i1 text encoder), padded to a fixed length.

    T5Gemma is an encoder-decoder adaptation of Gemma 2; only the encoder is loaded. The
    returned mask marks the real tokens, which `MMJiT` needs in order to substitute its learned
    `mask_token` at padded positions. Those positions are always zeroed: this tokenizer adds no
    BOS/EOS, so nothing in the sequence itself says where the prompt ends, and a pad position's
    activations are otherwise as large as a real token's.

    Prompts are padded to the longest in the batch, so `pad_to_multiple_of` is what makes
    `compile_forward` worth using: it rounds that length up to one of `max_sequence_length /
    pad_to_multiple_of` values, each of which is its own graph, so the default of 1 leaves every
    length its own. A warmup pass compiles all of them for its batch size, once, since a batch that
    first happens to be long enough would otherwise stall every rank at a different step.
    """

    def __init__(self,
                 model_name_or_path='google/t5gemma-2b-2b-ul2-it',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=256,
                 pad_to_multiple_of=1,
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # the released checkpoints are encoder-decoder; T5GemmaEncoderModel takes an encoder-only
        # config, so the decoder is never built and its weights are skipped
        config = T5GemmaConfig.from_pretrained(model_name_or_path)
        config.is_encoder_decoder = False
        self.text_encoder = hf_model_loader(
            _T5GemmaEncoderModel,
            model_name_or_path,
            config=config,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self._compiled_forward = None
        self._warmed_up = set()
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    @property
    def _grad_enabled(self):
        """A frozen encoder never needs grad, and pinning the mode keeps compile to one graph
        per shape instead of one for the training step's grad-enabled state and one for eval's.
        """
        return torch.is_grad_enabled() and not self.freeze

    def _forward_impl(self, input_ids, encoder_mask):
        return self.text_encoder(
            input_ids=input_ids, attention_mask=encoder_mask).last_hidden_state

    def _warmup_shapes(self, input_ids):
        """Compile every padded length this batch size can produce, given one real input.

        Called from `forward`, so it runs wherever the real call runs: after any module wrapper's
        pre-forward hook has gathered the parameters, and at the batch size the caller actually
        passed, which under gradient accumulation is the minibatch rather than the batch. Only the
        shape selects a graph, so one call per length is enough and mask contents are free; buckets
        sharing a batch size warm up once between them.
        """
        self._warmed_up.add(input_ids.shape[0])
        for length in range(self.pad_to_multiple_of, self.max_sequence_length + 1, self.pad_to_multiple_of):
            padded = input_ids.new_zeros(input_ids.shape[0], length)
            self._compiled_forward(padded, torch.ones_like(padded))

    def forward(self, prompt):
        tokens = self.tokenizer(
            [prompt] if isinstance(prompt, str) else list(prompt),
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors='pt')
        device = next(self.text_encoder.parameters()).device
        input_ids, attention_mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
        if input_ids.shape[1] == 0:  # this tokenizer adds no BOS/EOS, so '' is no tokens at all
            width = self.pad_to_multiple_of  # a width `warmup` has already compiled
            input_ids = input_ids.new_zeros(input_ids.shape[0], width)  # the pad id
            attention_mask = attention_mask.new_zeros(attention_mask.shape[0], width)
        # a fully masked row makes attention return NaNs, so let every row attend to its first
        # token, which is a real one wherever the prompt is not empty as this tokenizer pads on the
        # right; the mask multiply below discards whatever an empty row picked up
        encoder_mask = attention_mask.clone()
        encoder_mask[:, 0] = 1
        with torch.set_grad_enabled(self._grad_enabled):
            if self._compiled_forward is not None:
                if is_warmup() and input_ids.shape[0] not in self._warmed_up:
                    self._warmup_shapes(input_ids)
                prompt_embeds = self._compiled_forward(input_ids, encoder_mask)
            else:
                prompt_embeds = self._forward_impl(input_ids, encoder_mask)
        prompt_embeds = prompt_embeds * attention_mask[..., None].to(prompt_embeds.dtype)
        pad = self.max_sequence_length - prompt_embeds.shape[1]
        prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad))
        attention_mask = F.pad(attention_mask, (0, pad))
        return dict(
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=attention_mask)
