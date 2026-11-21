# Copyright (c) 2026 Hansheng Chen

import gzip
import hashlib
import pickle
from contextlib import contextmanager
from io import BytesIO
from typing import List, Optional, Union

import numpy as np
import orjson
import zstandard as zstd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

import mmcv
from mmcv.parallel import DataContainer as DC
from lakonlab.utils import get_root_logger
from lakonlab.utils.io_utils import FileClient


# bucket keys that describe how a config selects and samples, not the geometry the cache was
# written at, so a cache stays valid when they change or when an older manifest lacks them
CACHE_AGNOSTIC_BUCKET_KEYS = ('tier', 'latent_size', 'audio_latent_size')

# parsed jsonl files keyed by path, set only inside `shared_parse_cache`
_PARSE_CACHE = None


@contextmanager
def shared_parse_cache():
    """Parse each jsonl file once while building datasets that share metadata or datalists.

    The parsed records are shared between datasets, which must treat them as read-only. The cache
    is dropped on exit, so build every dataset inside it and fork dataloader workers after it.
    Nested uses share the outermost cache, which `build_dataset` opens around every build.
    """
    global _PARSE_CACHE
    if _PARSE_CACHE is not None:
        yield
        return
    _PARSE_CACHE = dict()
    try:
        yield
    finally:
        _PARSE_CACHE = None


def read_jsonl(path):
    """The parsed lines of a `.jsonl` or `.jsonl.gz` file, read once inside `shared_parse_cache`."""
    if _PARSE_CACHE is not None and path in _PARSE_CACHE:
        return _PARSE_CACHE[path]
    if not path.endswith(('.jsonl', '.jsonl.gz')):
        raise ValueError(f'{path} must be a .jsonl or .jsonl.gz file.')
    data = FileClient.infer_client(uri=path).get(path)
    if path.endswith('.gz'):
        data = gzip.decompress(data)
    records = [orjson.loads(line) for line in data.splitlines()]
    if _PARSE_CACHE is not None:
        _PARSE_CACHE[path] = records
    return records


class BucketedPromptDataset(Dataset):
    """Base for the bucketed prompt datasets: how records become samples.

    Owns the index space -- bucket assignment, slicing, fractional `repeat`, the cached
    latent/embedding overlay and the deterministic noise of `test_mode` -- and leaves the record
    format to a subclass, which supplies `_parse_metadata`, `_get_record_key` and `_load_record`.
    A prompt-only source bypasses the media records and uses one configured bucket.

    Args:
        data_root (str): Root path for IO, resolved via `FileClient`.
        metadata_path (Optional[str]): Metadata path, in whatever format the subclass reads.
        buckets (List[dict]): Ordered buckets, each with `height` and `width`, plus a
            `latent_size` when `test_mode` is set. The first bucket that the source can cover is
            selected.
        cache_dir (Optional[str]): Subdirectory of `data_root` containing `.zst` cache shards
            (latents and prompt embeddings). Enables the cache overlay when both the directory
            and `cache_datalist_path` exist.
        cache_datalist_path (Optional[str]): Datalist path written by the caching tool; cached
            samples are matched by record key, so filters and buckets may select a subset.
        prompt_cache_dir (Optional[str]): Subdirectory of `data_root` containing bucket-independent
            prompt embeddings, one `.zst` file per distinct prompt, which several datasets may
            share. Enables the prompt cache overlay, which supersedes any prompt embeddings in
            `cache_dir`, when both the directory and `prompt_cache_datalist_path` exist.
        prompt_cache_datalist_path (Optional[str]): Datalist path written by the prompt caching
            tool, listing the file of each of a record's prompts in record order; cached prompts
            are matched by record key, so filters and buckets may select a subset.
        bucket_samples_per_gpu (Optional[Union[int, dict]]): Per-bucket batch size, either
            shared by all populated buckets or keyed by bucket id. A non-positive dict entry
            drops that bucket without renumbering the remaining buckets.
        repeat (Union[float, dict]): Virtual repetition factor for each underlying sample,
            weighting this dataset inside a `ConcatDataset`. A dict applies a separate factor
            to each bucket id. The whole part repeats every record over consecutive indices, as
            `ImagePrompt` does; the fraction is a random subset drawn once, so `0.5` keeps a
            random half and `2.5` repeats everything twice plus a random half, without favouring
            records by position.
        repeat_seed (int): Seed for that draw, so every rank and every restart agrees.
        start_ind (Optional[int]): Start index (inclusive) into the filtered records. Defaults
            to 0.
        end_ind (Optional[int]): End index (exclusive) into the filtered records. Defaults to
            the number of filtered records.
        negative_prompt_embeds_path (Optional[str]): Path to a `torch.load`-able file of
            negative prompt embeddings, added to every sample. Training needs a negative prompt
            whenever `prob_prompt` drops a prompt, and sampling needs one for guidance.
        negative_prompt_kwargs (Optional[dict]): Keyword arguments forwarded to the text encoder
            for negative prompts, added to every sample. An alternative to the above, for a live
            text encoder.
        test_mode (bool): If True, return deterministic per-sample noise of the bucket's
            `latent_size` instead of reading the record's pixels.
        prompt_dataset_kwargs (Optional[dict]): Keyword arguments forwarded to
            `datasets.load_dataset(...)` for a prompt-only source instead of media metadata.
            Requires one bucket. Text datasets use their `text` column as `prompt`.
        prompt_key (str): Prompt column name in a prompt-only source. Defaults to `prompt`.
    """

    def __init__(self,
                 data_root: str,
                 metadata_path: Optional[str] = None,
                 buckets: Optional[List[dict]] = None,
                 cache_dir: Optional[str] = None,
                 cache_datalist_path: Optional[str] = None,
                 prompt_cache_dir: Optional[str] = None,
                 prompt_cache_datalist_path: Optional[str] = None,
                 bucket_samples_per_gpu: Optional[Union[int, dict]] = None,
                 repeat: Union[float, dict] = 1.0,
                 repeat_seed: int = 0,
                 start_ind: Optional[int] = None,
                 end_ind: Optional[int] = None,
                 negative_prompt_embeds_path: Optional[str] = None,
                 negative_prompt_kwargs: Optional[dict] = None,
                 test_mode: bool = False,
                 prompt_dataset_kwargs: Optional[dict] = None,
                 prompt_key: str = 'prompt'):
        super().__init__()
        self.data_root = data_root
        self._file_client = None

        self.metadata_path = metadata_path
        self.prompt_dataset_kwargs = prompt_dataset_kwargs
        self.prompt_key = prompt_key
        self.prompt_dataset = None
        self.buckets = buckets
        self.repeat = repeat
        self.repeat_seed = repeat_seed
        self.negative_prompt_embed_kwargs = None
        if negative_prompt_embeds_path is not None:
            self.negative_prompt_embed_kwargs = self.parse_prompt_embeds(
                torch.load(BytesIO(FileClient.infer_client(uri=negative_prompt_embeds_path).get(
                    negative_prompt_embeds_path)), map_location='cpu'), 0)
        self.negative_prompt_kwargs = negative_prompt_kwargs
        self.test_mode = test_mode
        if test_mode:
            assert all('latent_size' in bucket for bucket in buckets), \
                'test_mode needs a latent_size in every bucket.'

        # read before parsing metadata: `select_bucket` drops a record whose bucket is dropped
        self.dropped_bucket_ids = frozenset(
            int(k) for k, v in bucket_samples_per_gpu.items() if v <= 0
        ) if isinstance(bucket_samples_per_gpu, dict) else frozenset()

        self.records, record_bucket_ids = self.parse_metadata()
        self.record_bucket_ids = np.asarray(record_bucket_ids, dtype=np.int16)

        dataset_len = self.num_records
        if dataset_len == 0:  # every record was filtered out, which a ConcatDataset tolerates
            start_ind = end_ind = 0
        else:
            if start_ind is not None:
                start_ind = max(min(start_ind, dataset_len - 1), -dataset_len) % dataset_len
            else:
                start_ind = 0
            if end_ind is not None:
                end_ind = max(min(end_ind - 1, dataset_len - 1), -dataset_len) % dataset_len + 1
            else:
                end_ind = dataset_len
            assert start_ind < end_ind, \
                f'Invalid start_ind {start_ind} and end_ind {end_ind} for {dataset_len} records.'
        self.start_ind = start_ind
        self.end_ind = end_ind

        # an int16 array rather than a list of Python ints, which costs GBs per rank once
        # `repeat` inflates the sample count
        sliced_bucket_ids = self.record_bucket_ids[self.start_ind:self.end_ind]
        if isinstance(self.repeat, dict):
            self.repeat = self._parse_repeat(self.repeat, sliced_bucket_ids)
            whole_repeat_lut = np.array([
                int(self.repeat.get(bucket_id, 0))
                for bucket_id in range(len(self.buckets))], dtype=np.int64)
            whole_repeats = whole_repeat_lut[sliced_bucket_ids]
            self.whole_repeat_ends = np.cumsum(whole_repeats, dtype=np.int64)
            self.num_whole = int(self.whole_repeat_ends[-1]) if len(whole_repeats) else 0
        else:
            whole_repeats = int(self.repeat)
            self.whole_repeat_ends = None
            self.num_whole = whole_repeats * len(sliced_bucket_ids)
        self.extra_map = self._build_extra_map(sliced_bucket_ids)
        self.bucket_ids = np.concatenate([
            np.repeat(sliced_bucket_ids, whole_repeats), sliced_bucket_ids[self.extra_map]])
        if bucket_samples_per_gpu is not None:
            self.bucket_samples_per_gpu = self._parse_bucket_samples_per_gpu(
                bucket_samples_per_gpu)

        _file_client = FileClient.infer_client(uri=data_root)

        self.cache_dir_path = None
        if (cache_dir is not None
                and _file_client.isdir(_file_client.join_path(data_root, cache_dir))
                and cache_datalist_path is not None
                and FileClient.infer_client(uri=cache_datalist_path).isfile(cache_datalist_path)):
            self.cache_dir_path = _file_client.join_path(data_root, cache_dir)
            self.check_cache_datalist(cache_datalist_path)

        self.prompt_cache_dir_path = None
        if (prompt_cache_dir is not None
                and _file_client.isdir(_file_client.join_path(data_root, prompt_cache_dir))
                and prompt_cache_datalist_path is not None
                and FileClient.infer_client(uri=prompt_cache_datalist_path).isfile(prompt_cache_datalist_path)):
            self.prompt_cache_dir_path = _file_client.join_path(data_root, prompt_cache_dir)
            self.check_prompt_cache_datalist(prompt_cache_datalist_path)

        logger = get_root_logger()
        mmcv.print_log(f'Loaded {len(self)} records.', logger=logger)
        mmcv.print_log(f'Metadata path: {self.metadata_path}', logger=logger)
        mmcv.print_log(f'Cache dir: {self.cache_dir_path}', logger=logger)
        mmcv.print_log(f'Prompt cache dir: {self.prompt_cache_dir_path}', logger=logger)
        bucket_counts = np.bincount(self.bucket_ids, minlength=len(self.buckets))
        for bucket_id, bucket in enumerate(self.buckets):
            mmcv.print_log(
                f'Bucket {bucket_id} {bucket}: {bucket_counts[bucket_id]} records.', logger=logger)

    def parse_metadata(self):
        """Return the kept source records and their aligned bucket ids."""
        if self.prompt_dataset_kwargs is None:
            return self._parse_metadata()

        assert len(self.buckets) == 1, 'A prompt-only source needs exactly one bucket.'
        prompt_dataset_kwargs = self.prompt_dataset_kwargs
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
            self.prompt_dataset = self.prompt_dataset.rename_column('text', 'prompt')
        if isinstance(self.prompt_dataset, DatasetDict):
            split = 'train' if 'train' in self.prompt_dataset else list(self.prompt_dataset.keys())[0]
            self.prompt_dataset = self.prompt_dataset[split]
        assert isinstance(self.prompt_dataset, HFDataset), \
            f"Expected HF Dataset/DatasetDict, got {type(self.prompt_dataset)}."
        if 0 in self.dropped_bucket_ids:
            self.prompt_dataset = self.prompt_dataset.select([])
        return self.prompt_dataset, np.zeros(len(self.prompt_dataset), dtype=np.int16)

    def _parse_metadata(self):
        raise NotImplementedError

    def _select_bucket(self):
        raise NotImplementedError

    def select_bucket(self, *args, **kwargs):
        bucket_id = self._select_bucket(*args, **kwargs)
        return -1 if bucket_id in self.dropped_bucket_ids else bucket_id

    @property
    def num_records(self):
        return len(self.record_bucket_ids)

    def get_record_key(self, mapped_idx):  # unique per record, written to the cache datalist by the caching tool
        if self.prompt_dataset is None:
            return self._get_record_key(mapped_idx)
        prompt_data = self.prompt_dataset[mapped_idx]
        prompt_kwargs = prompt_data.get('prompt_kwargs', dict(prompt=prompt_data[self.prompt_key]))
        key = (prompt_kwargs['prompt'] if prompt_kwargs.keys() == {'prompt'}
               else orjson.dumps(prompt_kwargs, option=orjson.OPT_SORT_KEYS).decode('utf-8'))
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def _get_record_key(self, mapped_idx):
        raise NotImplementedError

    def load_record(self, mapped_idx, bucket):
        """One record as a sample, without `ids`, noise or cached tensors, plus which of its
        prompts was used, so a multi-prompt cache can be indexed the same way."""
        if self.prompt_dataset is None:
            return self._load_record(mapped_idx, bucket)
        prompt_data = self.prompt_dataset[mapped_idx]
        prompt = prompt_data[self.prompt_key]
        if 'prompt_kwargs' in prompt_data:
            prompt_kwargs = {k: DC(v, cpu_only=True) for k, v in prompt_data['prompt_kwargs'].items()}
        else:
            prompt_kwargs = dict(prompt=DC(prompt, cpu_only=True))
        return dict(name=DC(prompt, cpu_only=True), prompt_kwargs=prompt_kwargs), 0

    def _load_record(self, mapped_idx, bucket):
        raise NotImplementedError

    @property
    def file_client(self):
        if self._file_client is None:
            self._file_client = FileClient.infer_client(uri=self.data_root)
        return self._file_client

    def check_cache_datalist(self, cache_datalist_path):
        # cache shards are indexed by the caching run's own dataset index, which depends on its
        # filtering, so map local indices to cached indices by record key
        cache_entries = dict()
        for cache_idx, entry in enumerate(read_jsonl(cache_datalist_path)):
            if 'record_key' not in entry or 'bucket' not in entry:
                raise ValueError(
                    f'Cache datalist {cache_datalist_path} has no record keys or bucket '
                    'definitions; recache the data with the current caching tool.')
            if entry['record_key'] in cache_entries:
                raise ValueError(
                    f'Duplicate record key {entry["record_key"]} in cache datalist '
                    f'{cache_datalist_path}; record keys must be unique.')
            cache_entries[entry['record_key']] = (cache_idx, entry['bucket'])

        # Round trip through JSON so tuples and config dicts compare equal to datalist entries.
        buckets = [self.cache_bucket(bucket) for bucket in self.buckets]

        # indexed by record, like `_map_idx` output, so only the served slice has to be cached
        self.cache_idx_map = [cache_entries.get(self.get_record_key(mapped_idx), (-1, None))[0]
                              for mapped_idx in range(self.num_records)]
        for mapped_idx in range(self.start_ind, self.end_ind):
            record_key = self.get_record_key(mapped_idx)
            if record_key not in cache_entries:
                raise ValueError(
                    f'Record {record_key} is not in the cache datalist {cache_datalist_path}; '
                    'the current config selects samples that have not been cached.')
            cache_bucket = self.cache_bucket(cache_entries[record_key][1])
            # Bucket ids may be renumbered, but cached latents only match their original bucket.
            bucket = buckets[self.record_bucket_ids[mapped_idx]]
            if cache_bucket != bucket:
                raise ValueError(
                    f'Record {record_key} was cached with bucket {cache_bucket} but now selects '
                    f'bucket {bucket}, whose samples have not been cached.')

    def check_prompt_cache_datalist(self, prompt_cache_datalist_path):
        # bucket-independent, so a record maps to its prompt's file in every bucket and dataset
        entries = read_jsonl(prompt_cache_datalist_path)
        if self.prompt_dataset is not None and entries and 'filename' in entries[0]:
            # Prompt-only caches written by ImagePrompt already use hashes of prompt_kwargs.
            cache_names = {entry['filename']: [entry['filename']] for entry in entries}
        else:
            cache_names = {entry['record_key']: entry['filenames'] for entry in entries}
        self.prompt_cache_names = [cache_names.get(self.get_record_key(mapped_idx))
                                   for mapped_idx in range(self.num_records)]
        for mapped_idx in range(self.start_ind, self.end_ind):
            if self.prompt_cache_names[mapped_idx] is None:
                raise ValueError(
                    f'Record {self.get_record_key(mapped_idx)} is not in the prompt cache datalist '
                    f'{prompt_cache_datalist_path}; the current config selects prompts that have not '
                    'been cached.')

    @staticmethod
    def cache_bucket(bucket):
        """A bucket as the cache defines it, for comparing a config against a cache manifest."""
        bucket = {k: v for k, v in dict(bucket).items() if k not in CACHE_AGNOSTIC_BUCKET_KEYS}
        # round trip through json so tuples and config dicts compare equal to datalist entries
        return orjson.loads(orjson.dumps(bucket))

    def _parse_bucket_samples_per_gpu(self, bucket_samples_per_gpu):
        bucket_ids = np.unique(self.bucket_ids).tolist()
        if isinstance(bucket_samples_per_gpu, int):
            mapping = {bucket_id: bucket_samples_per_gpu for bucket_id in bucket_ids}
        elif isinstance(bucket_samples_per_gpu, dict):
            _bucket_samples_per_gpu = {int(k): v for k, v in bucket_samples_per_gpu.items()}
            try:
                mapping = {bucket_id: _bucket_samples_per_gpu[bucket_id] for bucket_id in bucket_ids}
            except KeyError as e:
                raise ValueError(f'bucket_samples_per_gpu is missing bucket {e.args[0]}.') from e
        else:
            raise TypeError(
                'bucket_samples_per_gpu must be an int or a dict from bucket id to samples_per_gpu.')

        for bucket_id, samples_per_gpu in mapping.items():
            if not isinstance(samples_per_gpu, int) or samples_per_gpu <= 0:
                raise ValueError(
                    f'bucket_samples_per_gpu[{bucket_id}] must be a positive integer, '
                    f'but got {samples_per_gpu}.')
        return mapping

    def parse_prompt_embeds(self, raw_data, index):
        # a cache written with prompt_sampling='all' holds one entry per prompt, in record order
        prompt_embed_kwargs = {k: v[index] if isinstance(v, list) else v
                               for k, v in raw_data['prompt_embed_kwargs'].items()}
        encoder_hidden_states_scale = prompt_embed_kwargs.pop('encoder_hidden_states_scale', None)
        if 'encoder_hidden_states' in prompt_embed_kwargs:
            encoder_hidden_states = prompt_embed_kwargs['encoder_hidden_states'].float()
            if encoder_hidden_states_scale is not None:
                encoder_hidden_states = encoder_hidden_states * encoder_hidden_states_scale
            prompt_embed_kwargs['encoder_hidden_states'] = encoder_hidden_states
        cap_feats_scale = prompt_embed_kwargs.pop('cap_feats_scale', None)
        if 'cap_feats' in prompt_embed_kwargs:
            cap_feats = prompt_embed_kwargs['cap_feats'].float()
            if cap_feats_scale is not None:
                cap_feats = cap_feats * cap_feats_scale
            prompt_embed_kwargs['cap_feats'] = DC(cap_feats)
        return prompt_embed_kwargs

    @staticmethod
    def _parse_repeat(repeat, bucket_ids):
        repeat = {int(k): v for k, v in repeat.items()}
        for bucket_id in np.unique(bucket_ids):
            if int(bucket_id) not in repeat:
                raise ValueError(f'repeat is missing bucket {bucket_id}.')
        for bucket_id, factor in repeat.items():
            if (isinstance(factor, bool) or not isinstance(factor, (int, float))
                    or not np.isfinite(factor) or factor < 0):
                raise ValueError(
                    f'repeat[{bucket_id}] must be a non-negative finite number, but got {factor}.')
        return repeat

    def _build_extra_map(self, bucket_ids):
        """The records that the fraction of `repeat` adds, relative to the slice.

        Drawn at random once with `repeat_seed`, so the extra copies are not always the same
        records, and a `repeat` below 1 truncates at random rather than by position, which
        matters because a source's records are not in a random order to begin with. The whole
        part of `repeat` needs no map: it repeats each record over consecutive indices. With
        bucket-wise repeat, the fractional draw is made separately within each bucket.
        """
        rng = np.random.default_rng(self.repeat_seed)
        if not isinstance(self.repeat, dict):
            extra = round(self.repeat % 1 * len(bucket_ids))
            return rng.choice(len(bucket_ids), size=extra, replace=False).astype(np.int32)

        extra_maps = []
        for bucket_id in np.unique(bucket_ids):
            bucket_map = np.flatnonzero(bucket_ids == bucket_id)
            extra = round(self.repeat[int(bucket_id)] % 1 * len(bucket_map))
            extra_maps.append(rng.choice(bucket_map, size=extra, replace=False))
        return (np.concatenate(extra_maps).astype(np.int32)
                if extra_maps else np.empty(0, dtype=np.int32))

    def _map_idx(self, idx):
        if idx < self.num_whole:
            if self.whole_repeat_ends is not None:
                return self.start_ind + int(np.searchsorted(
                    self.whole_repeat_ends, idx, side='right'))
            return self.start_ind + idx // int(self.repeat)
        return self.start_ind + int(self.extra_map[idx - self.num_whole])

    def __len__(self):
        return self.num_whole + len(self.extra_map)

    def __getitem__(self, idx):
        mapped_idx = self._map_idx(idx)
        bucket = self.buckets[self.bucket_ids[idx]]
        data, index = self.load_record(mapped_idx, bucket)
        data['ids'] = DC(idx, cpu_only=True)
        if self.negative_prompt_embed_kwargs is not None:
            data['negative_prompt_embed_kwargs'] = self.negative_prompt_embed_kwargs
        if self.negative_prompt_kwargs is not None:
            data['negative_prompt_kwargs'] = {
                k: DC(v, cpu_only=True) for k, v in self.negative_prompt_kwargs.items()}

        if self.test_mode:  # a fixed noise per sample, so samples are reproducible
            generator = torch.Generator().manual_seed(idx)
            data['noise'] = torch.randn(
                bucket['latent_size'], dtype=torch.float32, generator=generator)
            if 'audio_latent_size' in bucket:
                data['audio_noise'] = torch.randn(
                    bucket['audio_latent_size'], dtype=torch.float32, generator=generator)

        if self.cache_dir_path is not None:
            cache_path = self.file_client.join_path(
                self.cache_dir_path, f'{self.cache_idx_map[mapped_idx]:012d}.zst')
            data_bytesio = BytesIO(self.file_client.get(cache_path))
            with zstd.ZstdDecompressor().stream_reader(data_bytesio) as f:
                raw_data = pickle.load(f)
            if 'latents' in raw_data:
                latents = raw_data['latents'].float()
                latents_scale = raw_data.get('latents_scale', None)
                if latents_scale is not None:
                    latents = latents * latents_scale
                data['latents'] = latents
            if 'audio_latents' in raw_data:
                audio_latents = raw_data['audio_latents'].float()
                audio_latents_scale = raw_data.get('audio_latents_scale', None)
                if audio_latents_scale is not None:
                    audio_latents = audio_latents * audio_latents_scale
                data['audio_latents'] = audio_latents
            if 'prompt_embed_kwargs' in raw_data:
                data['prompt_embed_kwargs'] = self.parse_prompt_embeds(raw_data, index)

        if self.prompt_cache_dir_path is not None:
            cache_path = self.file_client.join_path(
                self.prompt_cache_dir_path, self.prompt_cache_names[mapped_idx][index] + '.zst')
            data_bytesio = BytesIO(self.file_client.get(cache_path))
            with zstd.ZstdDecompressor().stream_reader(data_bytesio) as f:
                raw_data = pickle.load(f)
            data['prompt_embed_kwargs'] = self.parse_prompt_embeds(raw_data, index)

        return data
