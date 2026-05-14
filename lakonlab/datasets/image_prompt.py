# Copyright (c) 2026 Hansheng Chen

import warnings
import logging
import os
import math
import pickle
import gzip
import orjson
import zstandard as zstd
from io import BytesIO
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.storage
torch.storage.UntypedStorage.dtype = torch.uint8  # hot patch for torch 2.6 deserialization

from datasets import load_dataset, DatasetDict, Dataset as HFDataset
import mmcv
from mmcv.fileio import FileClient
from mmcv.parallel import DataContainer as DC

from .builder import DATASETS
from lakonlab.utils import get_root_logger
from lakonlab.utils.io_utils import load_image


@DATASETS.register_module()
class ImagePrompt(Dataset):
    """Initialize an image/prompt dataset that reads either cached pickled records
    (zstd-compressed) or a HuggingFace prompt dataset (optionally paired with images).

    Args:
        data_root (str): Root path for IO, resolved via `mmcv.FileClient`.
        cache_dir (Optional[str]): Subdirectory of `data_root` containing `.zst`
            cache shards. Enables cache mode when provided and exists. Caches must
            contain pickled dicts with keys `"prompt"` and `"prompt_embed_kwargs"`,
            and optionally `"latents"` or `"latent_size"`.
        cache_datalist_path (Optional[str]): Optional datalist path for `cache_dir`.
            Supports `.jsonl`, `.jsonl.gz`, or `.json`. If not exists, files are
            discovered by listing the directory.
        ignore_cached_latents (bool): If True, ignores any cached latents and
            prioritizes loading images from `image_dir`. Defaults to False.
        prompt_dataset_kwargs (Optional[dict]): Keyword arguments forwarded to
            `datasets.load_dataset(...)`. Enables prompt-dataset mode when provided.
            If a `DatasetDict` is returned, a split (e.g., "train") is selected
            internally.
        image_dir (Optional[str]): Subdirectory of `data_root` with images to pair
            with prompts (used only in prompt-dataset mode).
        image_datalist_path (Optional[str]): Optional datalist for `image_dir`
            (same formats as above). When `bucketize=True`, JSONL entries must
            include `"size_idx"`.
        image_extension (Optional[str]): Image file extension used to compose
            paths when `image_dir` is set. Defaults to ".png".
        image_scale_factor (float): Scale factor applied to image spatial dimensions
            after loading. Defaults to 1.0 (no scaling).
        condition_image_dir (Optional[str]): Subdirectory of `data_root` with
            condition images to pair with prompts (used only in prompt-dataset mode).
        condition_image_datalist_path (Optional[str]): Optional datalist for
            `condition_image_dir` (same formats as above).
        condition_image_extension (Optional[str]): Image file extension used to
            compose paths when `condition_image_dir` is set. Defaults to ".png".
        condition_image_scale_factor (float): Scale factor applied to condition
            image spatial dimensions after loading. Defaults to 1.0 (no scaling).
        negative_prompt_embeds_path (Optional[str]): Path to a `torch.load`-able
            file containing keyward arguments forwarded to the diffusion model
            for negative prompt embeddings. Added to each sample when provided.
        negative_prompt_kwargs (Optional[dict]): Keyword arguments forwarded to
            the text encoder for negative prompts. Added to each sample when provided.
        pad_seq_len (int): If set, pads/truncates `"encoder_hidden_states"` and
            `"encoder_hidden_states_mask"` along the sequence dimension to this
            length. Defaults to None.
        latent_size (Optional[Tuple[int]]): Default latent shape `(C, H, W)` used
            when no cached latents exist and no image size is provided. Defaults to
            `(16, 128, 128)`.
        vae_scale_factor (Optional[Union[int, Tuple[int]]]): Downscale factor(s)
            applied to image dimensions when deriving latent sizes from image size.
            If an `int`, applies to each spatial dim; if a `tuple`, its length must
            match the provided image spatial size (e.g., `(H, W)` or `(T, H, W)` for
            video VAEs).
        repeat (int): Virtual repetition factor for each underlying sample.
            Affects `__len__` and index mapping.
        start_ind (Optional[int]): Start index (inclusive) into the underlying
            dataset. Defaults to 0.
        end_ind (int): End index (exclusive) into the underlying dataset. Defaults
            to dataset length.
        bucketize (bool): If True, enables bucketing in `DistributedSampler` so that
            each rank receives samples of the same size. Expects `"size_idx"` in JSONL
            datalists and collects bucket ids. Defaults to False.
        prompt_key (str): Prompt column name in prompt-dataset mode.
            Defaults to "prompt".
        test_mode (bool): If True, return deterministic noise per sample instead of
            reading/allocating real latents or images.
    """

    PROMPT_KEY_MAPS = {
        'prompt_embeds': 'encoder_hidden_states',
        'prompt_embeds_scale': 'encoder_hidden_states_scale',
        'pooled_prompt_embeds': 'pooled_projections',
        'prompt_embeds_mask': 'encoder_hidden_states_mask'
    }

    def __init__(self,
                 data_root: str,
                 cache_dir: Optional[str] = None,
                 cache_datalist_path: Optional[str] = None,
                 ignore_cached_latents: bool = False,
                 prompt_dataset_kwargs: Optional[dict] = None,
                 image_dir: Optional[str] = None,
                 image_datalist_path: Optional[str] = None,
                 image_extension: Optional[str] = '.png',
                 image_scale_factor: float = 1.0,
                 image_scale_method: str = 'bicubic',
                 condition_image_dir: Optional[str] = None,
                 condition_image_datalist_path: Optional[str] = None,
                 condition_image_extension: Optional[str] = '.png',
                 condition_image_scale_factor: float = 1.0,
                 negative_prompt_embeds_path: Optional[str] = None,
                 negative_prompt_kwargs: Optional[dict] = None,
                 pad_seq_len: int = None,
                 latent_size: Optional[Tuple[int]] = (16, 128, 128),
                 latent_patch_size: Optional[Union[int, Tuple[int]]] = 2,
                 vae_scale_factor: Optional[Union[int, Tuple[int]]] = 8,
                 repeat: int = 1,
                 start_ind: Optional[int] = None,
                 end_ind: int = None,
                 bucketize: bool = False,
                 prompt_key: str = 'prompt',
                 test_mode: bool = False):
        super().__init__()
        self.data_root = data_root
        self._file_client = None

        self.pad_seq_len = pad_seq_len
        self.prompt_key = prompt_key

        self.cache_dir_path = self.cache_datalist_path = None
        self.prompt_dataset = self.image_dir_path = self.condition_image_dir_path = None
        self.image_extension = image_extension
        self.image_scale_factor = image_scale_factor
        self.condition_image_extension = condition_image_extension
        self.condition_image_scale_factor = condition_image_scale_factor
        self.ignore_cached_latents = ignore_cached_latents
        self.bucketize = bucketize
        bucket_ids = None

        _file_client = FileClient.infer_client(uri=data_root)

        if (cache_dir is not None
                and _file_client.isdir(_file_client.join_path(data_root, cache_dir))
                and cache_datalist_path is not None
                and FileClient.infer_client(uri=cache_datalist_path).isfile(cache_datalist_path)):
            self.cache_dir_path = _file_client.join_path(data_root, cache_dir)
            self.cache_datalist, bucket_ids, _ = self.parse_datalist(
                self.cache_dir_path, cache_datalist_path)
            dataset_len = len(self.cache_datalist)

        elif prompt_dataset_kwargs is not None:
            if 'data_files' in prompt_dataset_kwargs and isinstance(prompt_dataset_kwargs['data_files'], str):
                prompt_data_path = prompt_dataset_kwargs['data_files']
                with FileClient.infer_client(uri=prompt_data_path).get_local_path(
                        prompt_data_path) as local_prompt_data_path:
                    _prompt_dataset_kwargs = prompt_dataset_kwargs.copy()
                    _prompt_dataset_kwargs['data_files'] = local_prompt_data_path
                    self.prompt_dataset = load_dataset(**_prompt_dataset_kwargs)
            else:
                self.prompt_dataset = load_dataset(**prompt_dataset_kwargs)
            if prompt_dataset_kwargs.get('path', None) == 'text':
                self.prompt_dataset = self.prompt_dataset.rename_column(
                    'text', 'prompt')
            if isinstance(self.prompt_dataset, DatasetDict):
                split = 'train' if 'train' in self.prompt_dataset else list(self.prompt_dataset.keys())[0]
                self.prompt_dataset = self.prompt_dataset[split]
            assert isinstance(self.prompt_dataset, HFDataset), \
                f"Expected HF Dataset/DatasetDict, got {type(self.prompt_dataset)}."
            dataset_len = len(self.prompt_dataset)

        else:
            raise ValueError('Either `cache_dir` or `prompt_dataset_kwargs` must be provided.')

        if image_dir is not None and _file_client.isdir(
                _file_client.join_path(data_root, image_dir)):
            self.image_dir_path = _file_client.join_path(data_root, image_dir)
            self.image_datalist, bucket_ids, self.image_sizes = self.parse_datalist(
                self.image_dir_path, image_datalist_path, datalist_must_exist=True)
            assert dataset_len == len(self.image_datalist)

        if condition_image_dir is not None and _file_client.isdir(
                _file_client.join_path(data_root, condition_image_dir)):
            self.condition_image_dir_path = _file_client.join_path(data_root, condition_image_dir)
            # No bucket ids for condition images, we assume that they either have the same shape,
            # or share the same bucket_ids as the main images
            self.condition_image_datalist, _, self.condition_image_sizes = self.parse_datalist(
                self.condition_image_dir_path, condition_image_datalist_path,
                datalist_must_exist=True, ignore_bucket_ids=True)
            assert dataset_len == len(self.condition_image_datalist)

        if bucket_ids is None and self.bucketize:
            assert self.prompt_dataset is not None
            bucket_ids = self.get_bucket_ids_from_prompt_dataset()

        self.negative_prompt_embed_kwargs = None
        if negative_prompt_embeds_path is not None:
            negative_prompt_embeds_bytesio = BytesIO(
                FileClient.infer_client(uri=negative_prompt_embeds_path).get(negative_prompt_embeds_path))
            self.negative_prompt_embed_kwargs = self.parse_prompt_embeds(
                torch.load(negative_prompt_embeds_bytesio, map_location='cpu'))
        self.negative_prompt_kwargs = negative_prompt_kwargs

        self.latent_size = latent_size
        self.latent_patch_size = latent_patch_size
        self.vae_scale_factor = vae_scale_factor
        self.image_scale_mode = image_scale_method

        self.repeat = repeat
        if start_ind is not None:
            start_ind = max(min(start_ind, dataset_len - 1), -dataset_len) % dataset_len
        else:
            start_ind = 0
        if end_ind is not None:
            end_ind = max(min(end_ind - 1, dataset_len - 1), -dataset_len) % dataset_len + 1
        else:
            end_ind = dataset_len
        assert start_ind < end_ind, f'Invalid start_ind and end_ind.'
        self.start_ind = start_ind
        self.end_ind = end_ind

        if self.bucketize:
            assert bucket_ids is not None and len(bucket_ids) == dataset_len
            self.bucket_ids = [bucket_ids[self._map_idx(i)] for i in range(len(self))]

        self.test_mode = test_mode

    @property
    def file_client(self):
        if self._file_client is None:
            self._file_client = FileClient.infer_client(uri=self.data_root)
        return self._file_client

    def get_bucket_ids_from_prompt_dataset(self):
        ds = self.prompt_dataset
        assert 'height' in ds.column_names and 'width' in ds.column_names, \
            'When bucketize=True and no datalist is provided, the prompt dataset ' \
            'must contain `height` and `width` columns.'
        cols = ['height', 'width']
        if 'frames' in ds.column_names:
            cols = ['frames'] + cols
        ds_arrow = ds.with_format('arrow', columns=cols)
        batch = ds_arrow[:]

        arrs = [batch[c].combine_chunks().to_numpy(zero_copy_only=False) for c in cols]
        arrs = np.stack(arrs, axis=1)

        _, inv = np.unique(arrs, axis=0, return_inverse=True)
        return inv.tolist()

    def parse_datalist(self, dir_path, datalist_path=None, datalist_must_exist=False, ignore_bucket_ids=False):
        logger = get_root_logger()

        if datalist_path is not None and FileClient.infer_client(uri=datalist_path).isfile(datalist_path):
            filenames = []
            bucket_ids = []
            image_sizes = []

            datalist_bytesio = BytesIO(FileClient.infer_client(uri=datalist_path).get(datalist_path))
            if datalist_path.endswith('.jsonl.gz') or datalist_path.endswith('.jsonl'):
                if datalist_path.endswith('.jsonl.gz'):
                    with gzip.open(datalist_bytesio, 'rt', encoding='utf-8') as f:
                        datalist = f.readlines()
                else:
                    datalist = datalist_bytesio.read().decode('utf-8').splitlines()
                for line in datalist:
                    data_item = orjson.loads(line)
                    if 'filename' in data_item:
                        filenames.append(data_item['filename'])
                    elif 'image_hash' in data_item:
                        filenames.append(data_item['image_hash'])
                    else:
                        raise ValueError('No valid key to identify data item.')
                    if self.bucketize and not ignore_bucket_ids:
                        if 'size_idx' in data_item:
                            bucket_ids.append(data_item['size_idx'])
                        elif 'bucket_id' in data_item:
                            bucket_ids.append(data_item['bucket_id'])
                        else:
                            raise ValueError(
                                'Either `size_idx` or `bucket_id` must be present in datalist for bucketize.')
                    if 'image_size' in data_item:
                        image_sizes.append(data_item['image_size'])
                    else:
                        image_sizes.append(None)
            elif datalist_path.endswith('.json'):
                assert not self.bucketize, 'Bucketize not supported for json datalist.'
                datalist = orjson.loads(datalist_bytesio.read())
                for data_item in datalist:
                    filenames.append(os.path.splitext(os.path.basename(data_item))[0])
            else:
                raise ValueError('Datalist file must be .jsonl, .jsonl.gz or .json')

        else:
            assert not datalist_must_exist, f'Datalist file {datalist_path} does not exist.'
            assert not self.bucketize, 'Bucketize not supported when datalist is not provided.'
            mmcv.print_log(
                f'Datalist file {datalist_path} does not exist, directly list all files in the directory.',
                logger=logger,
                level=logging.WARNING)
            # list all files in the directory
            _file_client = FileClient.infer_client(uri=dir_path)
            filenames = [os.path.splitext(p)[0] for p in _file_client.list_dir_or_file(dir_path)]
            filenames.sort()
            bucket_ids = None
            image_sizes = None
            # save the datalist if datalist_path is provided
            if datalist_path is not None:
                if datalist_path.endswith('.jsonl.gz') or datalist_path.endswith('.jsonl'):
                    datalist = []
                    for filename in filenames:
                        datalist.append(orjson.dumps({'filename': filename}).decode('utf-8'))
                    datalist_str = '\n'.join(datalist)
                    if datalist_path.endswith('.jsonl.gz'):
                        datalist_bytesio = BytesIO()
                        with gzip.open(datalist_bytesio, 'wt', encoding='utf-8') as f:
                            f.write(datalist_str)
                        FileClient.infer_client(uri=datalist_path).put(datalist_bytesio.getvalue(), datalist_path)
                    else:
                        FileClient.infer_client(uri=datalist_path).put_text(datalist_str, datalist_path)
                elif datalist_path.endswith('.json'):
                    datalist = filenames
                    FileClient.infer_client(uri=datalist_path).put_text(
                        orjson.dumps(datalist).decode('utf-8'), datalist_path)

        mmcv.print_log(f'Loaded {len(filenames)} samples.', logger=logger)

        return filenames, bucket_ids, image_sizes

    def pad_prompt_embeds(self, prompt_embeds):
        if self.pad_seq_len is not None:
            if prompt_embeds.size(0) > self.pad_seq_len:
                prompt_embeds = prompt_embeds[:self.pad_seq_len]
            else:
                zeros_size = (self.pad_seq_len - prompt_embeds.size(0),) + prompt_embeds.shape[1:]
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds.new_zeros(zeros_size)], dim=0)
        return prompt_embeds

    def parse_prompt_embeds(self, data):
        prompt_embed_kwargs = data.get('prompt_embed_kwargs', {}).copy()

        # Map legacy keys to new ones if not already present
        for legacy_key, new_key in self.PROMPT_KEY_MAPS.items():
            if legacy_key in data and new_key not in prompt_embed_kwargs:
                prompt_embed_kwargs[new_key] = data[legacy_key]

        # Common post-processing
        encoder_hidden_states_scale = prompt_embed_kwargs.pop('encoder_hidden_states_scale', None)
        if 'encoder_hidden_states' in prompt_embed_kwargs:
            encoder_hidden_states = prompt_embed_kwargs['encoder_hidden_states'].float()
            if encoder_hidden_states_scale is not None:
                encoder_hidden_states = encoder_hidden_states * encoder_hidden_states_scale
            prompt_embed_kwargs['encoder_hidden_states'] = self.pad_prompt_embeds(encoder_hidden_states)

        cap_feats_scale = prompt_embed_kwargs.pop('cap_feats_scale', None)
        if 'cap_feats' in prompt_embed_kwargs:
            cap_feats = prompt_embed_kwargs['cap_feats'].float()
            if cap_feats_scale is not None:
                cap_feats = cap_feats * cap_feats_scale
            prompt_embed_kwargs['cap_feats'] = DC(cap_feats)  # no stacking/padding for cap_feats

        if 'pooled_projections' in prompt_embed_kwargs:
            prompt_embed_kwargs['pooled_projections'] = prompt_embed_kwargs['pooled_projections'].float()

        if 'encoder_hidden_states_mask' in prompt_embed_kwargs:
            prompt_embed_kwargs['encoder_hidden_states_mask'] = self.pad_prompt_embeds(
                prompt_embed_kwargs['encoder_hidden_states_mask'])

        return prompt_embed_kwargs

    def calculate_latent_size(self, image_spatial_size):
        if isinstance(self.vae_scale_factor, int):
            latent_spatial_size = tuple(s // self.vae_scale_factor for s in image_spatial_size)
        else:
            assert len(self.vae_scale_factor) == len(image_spatial_size)
            latent_spatial_size = tuple(
                s // f for s, f in zip(image_spatial_size, self.vae_scale_factor))
        latent_size = (self.latent_size[0],) + latent_spatial_size
        return latent_size

    def calculate_scaled_image_size(self, image_spatial_size, scale_factor):
        vae_scale_factor = self.vae_scale_factor
        if isinstance(vae_scale_factor, int):
            vae_scale_factor = [vae_scale_factor] * len(image_spatial_size)
        latent_patch_size = self.latent_patch_size
        if isinstance(latent_patch_size, int):
            latent_patch_size = [latent_patch_size] * len(image_spatial_size)
        image_patch_size = [vsf * lps for vsf, lps in zip(vae_scale_factor, latent_patch_size)]
        if len(image_spatial_size) == 2:
            new_spatial_size = (
                max(int(round(image_spatial_size[0] * scale_factor / image_patch_size[0])), 1) * image_patch_size[0],
                max(int(round(image_spatial_size[1] * scale_factor / image_patch_size[1])), 1) * image_patch_size[1])
        elif len(image_spatial_size) == 3:
            new_spatial_size = (
                image_spatial_size[0],
                max(int(round(image_spatial_size[1] * scale_factor / image_patch_size[1])), 1) * image_patch_size[1],
                max(int(round(image_spatial_size[2] * scale_factor / image_patch_size[2])), 1) * image_patch_size[2])
        else:
            raise ValueError(f'Unsupported image spatial size {image_spatial_size}.')
        return new_spatial_size

    def scale_image(self, image, scale_factor):
        new_spatial_size = self.calculate_scaled_image_size(image.shape[1:], scale_factor)

        out_h, out_w = new_spatial_size[-2:]
        in_h, in_w = image.shape[-2:]
        scale = max(out_h / in_h, out_w / in_w)
        scaled_h = int(math.ceil(in_h * scale))
        scaled_w = int(math.ceil(in_w * scale))

        if scaled_h != in_h or scaled_w != in_w:
            if len(new_spatial_size) == 2:
                if self.image_scale_mode == 'lanczos':
                    image = (image.permute(1, 2, 0).clamp(min=0, max=1) * 255.0).round().to(torch.uint8).numpy()
                    image = Image.fromarray(image).resize(
                        (scaled_w, scaled_h), resample=Image.LANCZOS)
                    image = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
                else:
                    image = F.interpolate(
                        image[None], size=(scaled_h, scaled_w),
                        mode=self.image_scale_mode, align_corners=False, antialias=True
                    )[0].clamp(min=0, max=1)
            elif len(new_spatial_size) == 3:
                if self.image_scale_mode == 'lanczos':
                    image_list = []
                    for image_single in (
                            image.permute(1, 2, 3, 0).clamp(min=0, max=1) * 255.0).round().to(torch.uint8).numpy():
                        image_single = Image.fromarray(image_single).resize(
                            (scaled_w, scaled_h), resample=Image.LANCZOS)
                        image_single = torch.from_numpy(np.asarray(image_single))
                        image_list.append(image_single)
                    image = torch.stack(image_list, dim=0).permute(3, 0, 1, 2).float() / 255.0
                else:
                    image = F.interpolate(
                        image, size=(scaled_h, scaled_w),
                        mode=self.image_scale_mode, align_corners=False, antialias=True
                    ).clamp(min=0, max=1)
            else:
                raise ValueError(f'Unsupported image spatial size {image.shape[1:]}.')

        top = (scaled_h - out_h) // 2
        left = (scaled_w - out_w) // 2
        image = image[..., top:top + out_h, left:left + out_w]

        return image

    def calculate_scaled_latent_size(self, latent_spatial_size, scale_factor):
        latent_patch_size = self.latent_patch_size
        if isinstance(latent_patch_size, int):
            latent_patch_size = [latent_patch_size] * len(latent_spatial_size)
        if len(latent_spatial_size) == 2:
            new_spatial_size = (
                max(int(round(latent_spatial_size[0] * scale_factor / latent_patch_size[0])), 1) * latent_patch_size[0],
                max(int(round(latent_spatial_size[1] * scale_factor / latent_patch_size[1])), 1) * latent_patch_size[1])
        elif len(latent_spatial_size) == 3:
            new_spatial_size = (
                latent_spatial_size[0],
                max(int(round(latent_spatial_size[1] * scale_factor / latent_patch_size[1])), 1) * latent_patch_size[1],
                max(int(round(latent_spatial_size[2] * scale_factor / latent_patch_size[2])), 1) * latent_patch_size[2])
        else:
            raise ValueError(f'Unsupported image spatial size {latent_spatial_size}.')
        return new_spatial_size

    def scale_latent(self, latent, scale_factor):
        new_spatial_size = self.calculate_scaled_latent_size(latent.shape[1:], scale_factor)

        out_h, out_w = new_spatial_size[-2:]
        in_h, in_w = latent.shape[-2:]
        scale = max(out_h / in_h, out_w / in_w)
        scaled_h = int(math.ceil(in_h * scale))
        scaled_w = int(math.ceil(in_w * scale))

        if scaled_h != in_h or scaled_w != in_w:
            if len(new_spatial_size) == 2:
                latent = F.interpolate(
                    latent[None], size=(scaled_h, scaled_w), mode='bilinear', align_corners=False, antialias=False
                )[0]
            elif len(new_spatial_size) == 3:
                latent = F.interpolate(
                    latent, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False, antialias=False
                )
            else:
                raise ValueError(f'Unsupported image spatial size {latent.shape[1:]}.')

        top = (scaled_h - out_h) // 2
        left = (scaled_w - out_w) // 2
        latent = latent[..., top:top + out_h, left:left + out_w]

        return latent

    def _map_idx(self, idx):
        return self.start_ind + (idx // self.repeat)

    def __len__(self):
        return self.repeat * (self.end_ind - self.start_ind)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*",
                category=UserWarning,
            )

            mapped_idx = self._map_idx(idx)

            prompt_data = None

            if self.cache_dir_path is not None:
                data_path = self.file_client.join_path(
                    self.cache_dir_path, f'{self.cache_datalist[mapped_idx]}.zst')
                data_bytesio = BytesIO(self.file_client.get(data_path))
                with zstd.ZstdDecompressor().stream_reader(data_bytesio) as f:
                    raw_data = pickle.load(f)

                data = dict(
                    ids=DC(idx, cpu_only=True),
                    name=DC(raw_data['prompt'], cpu_only=True),
                    prompt_embed_kwargs=self.parse_prompt_embeds(raw_data))

                if not self.ignore_cached_latents:  # load latents
                    if 'latents' in raw_data:
                        latents = raw_data['latents']
                        if self.test_mode:
                            latent_size = (latents.size(0), ) + self.calculate_scaled_latent_size(
                                latents.shape[1:], self.image_scale_factor)
                            data['noise'] = torch.randn(
                                latent_size, dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
                        else:
                            data['latents'] = latents.float()
                            latents_scale = raw_data.get('latents_scale', None)
                            if latents_scale is not None:
                                data['latents'] = data['latents'] * latents_scale
                            data['latents'] = self.scale_latent(data['latents'], self.image_scale_factor)
                    else:
                        if 'latent_size' in raw_data:
                            latent_size = raw_data['latent_size']
                            latent_size = (latent_size[0],) + self.calculate_scaled_latent_size(
                                latent_size[1:], self.image_scale_factor)
                        elif 'condition_latents' in raw_data:
                            latent_size = raw_data['condition_latents'].shape
                        else:
                            latent_size = self.latent_size
                        if self.test_mode:
                            data['noise'] = torch.randn(
                                latent_size, dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
                        else:
                            data['latents'] = torch.empty(latent_size, dtype=torch.float32)

                if 'condition_latents' in raw_data:
                    condition_latents = raw_data['condition_latents']
                    data['condition_latents'] = condition_latents.float()
                    condition_latents_scale = raw_data.get('condition_latents_scale', None)
                    if condition_latents_scale is not None:
                        data['condition_latents'] = data['condition_latents'] * condition_latents_scale
                    data['condition_latents'] = self.scale_latent(
                        data['condition_latents'], self.condition_image_scale_factor)

            else:
                prompt_data = self.prompt_dataset[mapped_idx]
                prompt = prompt_data[self.prompt_key]
                if 'prompt_kwargs' in prompt_data:
                    prompt_kwargs = {k: DC(v, cpu_only=True) for k, v in prompt_data['prompt_kwargs'].items()}
                else:
                    prompt_kwargs = dict(prompt=DC(prompt, cpu_only=True))
                data = dict(
                    ids=DC(idx, cpu_only=True),
                    name=DC(prompt, cpu_only=True),
                    prompt_kwargs=prompt_kwargs)

            if self.image_dir_path is not None:
                image_path = self.file_client.join_path(
                    self.image_dir_path, self.image_datalist[mapped_idx] + self.image_extension)
                image_size = self.image_sizes[mapped_idx] if self.image_sizes is not None else None
                image = load_image(image_path, self.file_client, target_size=image_size)
                image = np.moveaxis(image, -1, 0)  # channel first
                if self.test_mode:
                    data['noise'] = torch.randn(
                        self.calculate_latent_size(
                            self.calculate_scaled_image_size(image.shape[1:], self.image_scale_factor)),
                        dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
                else:
                    images = torch.from_numpy(image)
                    if images.dtype == torch.uint8:
                        images = images.float() / 255.0
                    assert torch.is_floating_point(images), f'Image dtype {images.dtype} not supported.'
                    data['images'] = self.scale_image(images.float(), self.image_scale_factor)
            elif 'latents' not in data and 'noise' not in data:  # allocate latents if not already loaded
                if prompt_data is not None and 'height' in prompt_data and 'width' in prompt_data:
                    image_spatial_size = (prompt_data['height'], prompt_data['width'])
                    if 'frames' in prompt_data:
                        image_spatial_size = (prompt_data['frames'],) + image_spatial_size
                    latent_size = self.calculate_latent_size(
                        self.calculate_scaled_image_size(image_spatial_size, self.image_scale_factor))
                elif 'condition_latents' in data:
                    latent_size = data['condition_latents'].shape
                else:
                    latent_size = self.latent_size
                if self.test_mode:
                    data['noise'] = torch.randn(
                        latent_size, dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
                else:
                    data['latents'] = torch.empty(latent_size, dtype=torch.float32)

            if self.condition_image_dir_path is not None:
                condition_image_path = self.file_client.join_path(
                    self.condition_image_dir_path,
                    self.condition_image_datalist[mapped_idx] + self.condition_image_extension)
                condition_image_size = self.condition_image_sizes[mapped_idx] \
                    if self.condition_image_sizes is not None else None
                condition_image = load_image(condition_image_path, self.file_client, target_size=condition_image_size)
                condition_image = np.moveaxis(condition_image, -1, 0)  # channel first
                condition_images = torch.from_numpy(condition_image)
                if condition_images.dtype == torch.uint8:
                    condition_images = condition_images.float() / 255.0
                assert torch.is_floating_point(condition_images), \
                    f'Condition image dtype {condition_images.dtype} not supported.'
                data['condition_images'] = self.scale_image(
                    condition_images.float(), self.condition_image_scale_factor)

            if self.negative_prompt_embed_kwargs is not None:
                data.update(negative_prompt_embed_kwargs=self.negative_prompt_embed_kwargs)
            if self.negative_prompt_kwargs is not None:
                data.update(negative_prompt_kwargs={
                    k: DC(v, cpu_only=True) for k, v in self.negative_prompt_kwargs.items()
                })

            return data
