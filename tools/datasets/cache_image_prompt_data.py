from lakonlab.utils.env import apply_default_env
apply_default_env()

import os
import inspect
import argparse
import multiprocessing as mp
import re

import pickle
import zstandard as zstd
import gzip
import orjson

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from io import BytesIO

import torch
import torch.nn as nn
import torch.distributed as dist

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist

from lakonlab.models import build_module
from lakonlab.utils.io_utils import FileClient
from lakonlab.utils import get_root_logger, setup_global_config, setup_multi_processes
from lakonlab.datasets import build_dataset, build_dataloader
from lakonlab.parallel import apply_module_wrapper
from lakonlab.runner.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cache the image latents and text embeddings for the ImagePrompt dataset. Usage: torchrun '
                    '--nnodes=1 --nproc_per_node=<NUM_GPUS> tools/datasets/cache_image_prompt_data.py <PATH_TO_CONFIG> '
                    '--text-encoder <PATH_TO_TEXT_ENCODER_CONFIG> --launcher pytorch --diff_seed')
    parser.add_argument('config', help='file path to the config containing the dataset and vae')
    parser.add_argument('--text-encoder', type=str, help='file path to the config of text encoder')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--data',
        type=str,
        nargs='+')
    parser.add_argument(
        '--max-size',
        type=int,
        help='max dataset size')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='shuffle the dataloader'
    )
    parser.add_argument(
        '--workers', type=int, default=4, help='number of data loading workers per GPU')
    parser.add_argument(
        '--batch-size', type=int, default=4, help='batch size per GPU')
    parser.add_argument(
        '--no-text',
        action='store_true',
        help='cache only the latents, leaving the prompts to tools/datasets/cache_prompt_embeds.py')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip computing samples whose cache .zst file already exists')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def to_scaled_fp8(t: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn, eps=1e-6):
    amax = t.abs().max()
    fp8_max = torch.finfo(dtype).max
    scale = (amax.float() / fp8_max).clamp(min=eps)
    t_scaled = (t.float() / scale).to(dtype)
    return t_scaled, scale


class Preprocessor(nn.Module):

    def __init__(self, text_encoder, vae=None, audio_vae=None):
        super().__init__()
        self.text_encoder = build_module(text_encoder) if text_encoder is not None else None
        self.vae = build_module(vae) if vae is not None else None
        self.audio_vae = build_module(audio_vae) if audio_vae is not None else None

    def val_step(self, data, **kwargs):
        prompt_embed_kwargs = None
        if self.text_encoder is not None:
            prompt_kwargs = dict(data['prompt_kwargs'])
            # a dataset in all-prompts mode hands over one list of prompts per sample; encode them in
            # one batch, then regroup, so each sample keeps a list of embeddings
            prompts = prompt_kwargs.get('prompt')
            counts = ([len(prompt) for prompt in prompts]
                      if prompts and isinstance(prompts[0], list) else None)
            if counts is not None:
                prompt_kwargs['prompt'] = [prompt for group in prompts for prompt in group]

            prompt_embed_kwargs = self.text_encoder(**prompt_kwargs)
            if counts is not None:
                offsets = [0]
                for count in counts:
                    offsets.append(offsets[-1] + count)
                prompt_embed_kwargs = {
                    k: [list(v[offsets[i]:offsets[i + 1]]) for i in range(len(counts))]
                    for k, v in prompt_embed_kwargs.items()}

        output = dict(
            data_ids=data['ids'],
            prompts=data['name'],
            prompt_embed_kwargs=prompt_embed_kwargs)
        if 'images' in data and self.vae is not None:
            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents = self.vae.encode((data['images'] * 2 - 1).to(vae_dtype)).float()
            output.update(latents=latents)
        if 'condition_images' in data and self.vae is not None:
            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            kwargs = dict()
            if 'sample_mode' in inspect.signature(self.vae.encode).parameters:
                kwargs.update(sample_mode='argmax')
            condition_latents = self.vae.encode((data['condition_images'] * 2 - 1).to(vae_dtype), **kwargs).float()
            output.update(condition_latents=condition_latents)
        if 'audio' in data and self.audio_vae is not None:
            if hasattr(self.audio_vae, 'dtype'):
                audio_vae_dtype = self.audio_vae.dtype
            else:
                audio_vae_dtype = next(self.audio_vae.parameters()).dtype
            audio_latents = self.audio_vae.encode(data['audio'].to(audio_vae_dtype)).float()
            output.update(audio_latents=audio_latents)
        return output


def save_cache(data, cache_path):
    file_client = FileClient.infer_client(uri=cache_path)
    bytesio = BytesIO()
    compressor = zstd.ZstdCompressor(level=3)
    with compressor.stream_writer(bytesio, closefd=False) as f:
        pickle.dump(data, f)
    file_client.put(bytesio.getvalue(), cache_path)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.text_encoder is not None:
        cfg.model.text_encoder = Config.fromfile(args.text_encoder).model.text_encoder
    assert args.no_text or hasattr(cfg.model, 'text_encoder')

    setup_multi_processes(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    setup_global_config(dict(cfg.get('global_config', dict())))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
        world_size = 1
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE', 1)
    num_cpus = mp.cpu_count() // int(local_world_size)
    max_save_cache_workers = max(num_cpus - 2, 1)

    logger = get_root_logger(log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)

    model = Preprocessor(
        None if args.no_text else cfg.model.text_encoder,
        getattr(cfg.model, 'vae', None),
        getattr(cfg.model, 'audio_vae', None)
    ).requires_grad_(False).eval()

    if distributed:
        model = apply_module_wrapper(model, 'ddp', cfg)
    else:
        model = MMDataParallel(model, device_ids=[0])

    if args.data is None:
        data_dict = {k: v for k, v in cfg.data.items() if re.fullmatch(r'(train|val|test)\d*', k)}
    else:
        data_dict = {k: cfg.data[k] for k in args.data}

    for dataset_name, dataset in data_dict.items():
        data_root = dataset['data_root']
        root_file_client = FileClient.infer_client(uri=data_root)
        cache_dir_path = root_file_client.join_path(data_root, dataset['cache_dir'])

        cache_datalist_path = dataset['cache_datalist_path']
        datalist_file_client = FileClient.infer_client(uri=cache_datalist_path)
        if datalist_file_client.isfile(cache_datalist_path):
            continue  # already cached

        # disable slicing, repeat and the prompt cache overlay, whose embeddings are not cached here
        for key in ['start_ind', 'end_ind', 'repeat', 'prompt_cache_dir', 'prompt_cache_datalist_path']:
            dataset.pop(key, None)
        if args.max_size is not None:
            dataset.start_ind = -args.max_size

        if args.skip_existing:
            dataset = build_dataset(dataset)
            ori_len = len(dataset)
            bucket_ids = getattr(dataset, 'bucket_ids', None)
            buckets = getattr(dataset, 'buckets', None)
            get_record_key = getattr(dataset, 'get_record_key', None)
            start_ind = getattr(dataset, 'start_ind', None)

            if rank == 0:
                print(f'Checking existing cached files in {cache_dir_path} ...')
                existing_files = root_file_client.list_dir_or_file(
                    cache_dir_path)
                existing_ids = set()
                for file_path in existing_files:
                    if file_path.endswith('.zst'):
                        existing_ids.add(int(os.path.splitext(file_path)[0]))
                subset_ids = [i for i in range(ori_len) if i not in existing_ids]
                print(f'Found {len(existing_ids)} existing cached files, '
                      f'Original dataset size: {ori_len}, '
                      f'Skipping {ori_len - len(subset_ids)} samples. '
                      f'{len(subset_ids)} samples to be cached.')
            else:
                subset_ids = None

            if distributed:
                _subset_ids = [subset_ids]
                dist.broadcast_object_list(_subset_ids, src=0)
                subset_ids = _subset_ids[0]

            dataset = dict(
                type='Subset',
                dataset=dataset,
                indices=subset_ids)
            dataset = build_dataset(dataset)

        else:
            # build the dataloader
            dataset = build_dataset(dataset)
            ori_len = len(dataset)
            bucket_ids = getattr(dataset, 'bucket_ids', None)
            buckets = getattr(dataset, 'buckets', None)
            get_record_key = getattr(dataset, 'get_record_key', None)
            start_ind = getattr(dataset, 'start_ind', None)

        if len(dataset) > 0:
            # The default loader config
            loader_cfg = dict(shuffle=args.shuffle)
            # The overall dataloader settings
            loader_cfg.update({
                k: v
                for k, v in cfg.data.items()
                if k not in [
                    'train_dataloader', 'val_dataloader', 'test_dataloader'
                ] and not re.fullmatch(r'(train|val|test)\d*', k)
            })

            # specific config for test loader
            batch_size = args.batch_size
            test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}
            test_loader_cfg.update(
                samples_per_gpu=batch_size,
                workers_per_gpu=args.workers,
                prefetch_factor=batch_size * 2)

            dataloader = build_dataloader(dataset, **test_loader_cfg)

            if args.seed is not None:
                logger.info(f'Set random seed to {args.seed}, '
                            f'deterministic: {args.deterministic}, '
                            f'use_rank_shift: {args.diff_seed}')
                set_random_seed(
                    args.seed,
                    deterministic=args.deterministic,
                    use_rank_shift=args.diff_seed)

            batch_size = dataloader.batch_size
            if batch_size is None:  # bucketized batch sampler
                batch_size = max(dataset.bucket_samples_per_gpu.values())

            mp_ctx = mp.get_context("spawn")
            proc_pool = ProcessPoolExecutor(
                max_workers=min(max_save_cache_workers, batch_size * 2),
                mp_context=mp_ctx)
            cap = max_save_cache_workers * 2
            pending = set()

            torch.set_grad_enabled(False)

            for data in tqdm(dataloader, desc=f'Caching {dataset_name} data', unit='batch', disable=rank != 0):
                outputs_dict = model.val_step(data)
                data_ids = outputs_dict['data_ids']
                prompts = outputs_dict['prompts']

                for batch_id, data_id in enumerate(data_ids):
                    cached_data = dict(prompt=prompts[batch_id])
                    if outputs_dict['prompt_embed_kwargs'] is not None:
                        cached_data['prompt_embed_kwargs'] = dict()
                        for k, v in outputs_dict['prompt_embed_kwargs'].items():
                            value = v[batch_id]  # a tensor, or a list of them in all-prompts mode
                            if k in ('encoder_hidden_states', 'cap_feats'):
                                if isinstance(value, list):
                                    scaled = [to_scaled_fp8(item) for item in value]
                                    cached_data['prompt_embed_kwargs'][k] = [
                                        item.cpu() for item, _ in scaled]
                                    cached_data['prompt_embed_kwargs'][k + '_scale'] = [
                                        scale.cpu() for _, scale in scaled]
                                else:
                                    value_fp8, value_scale = to_scaled_fp8(value)
                                    cached_data['prompt_embed_kwargs'][k] = value_fp8.cpu()
                                    cached_data['prompt_embed_kwargs'][k + '_scale'] = value_scale.cpu()
                            elif isinstance(value, list):
                                cached_data['prompt_embed_kwargs'][k] = [
                                    item.cpu().clone() for item in value]
                            else:
                                cached_data['prompt_embed_kwargs'][k] = value.cpu().clone()
                    if 'latents' in outputs_dict:
                        latents_fp8, latents_scale = to_scaled_fp8(outputs_dict['latents'][batch_id])
                        cached_data['latents'] = latents_fp8.cpu()
                        cached_data['latents_scale'] = latents_scale.cpu()
                    if 'condition_latents' in outputs_dict:
                        condition_latents_fp8, condition_latents_scale = to_scaled_fp8(outputs_dict['condition_latents'][batch_id])
                        cached_data['condition_latents'] = condition_latents_fp8.cpu()
                        cached_data['condition_latents_scale'] = condition_latents_scale.cpu()
                    if 'audio_latents' in outputs_dict:
                        cached_data['audio_latents'] = outputs_dict['audio_latents'][batch_id].bfloat16().cpu()
                    cached_data_name = f'{data_id:012d}'

                    if len(pending) >= cap:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in done:
                            f.result()

                    pending.add(proc_pool.submit(
                        save_cache,
                        cached_data,
                        root_file_client.join_path(cache_dir_path, cached_data_name + '.zst')))

            if pending:
                done, pending = wait(pending)
                for f in done:
                    f.result()
            proc_pool.shutdown(wait=True)

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            bytesio = BytesIO()
            if cache_datalist_path.endswith('.jsonl.gz') or cache_datalist_path.endswith('.jsonl'):
                datalist = []
                for data_id in range(ori_len):
                    entry = dict(filename=f'{data_id:012d}')
                    if buckets is not None:  # bucket definition, so the training config may renumber buckets
                        entry.update(bucket=dict(buckets[int(bucket_ids[data_id])]))
                    elif bucket_ids is not None:
                        entry.update(bucket_id=int(bucket_ids[data_id]))
                    if get_record_key is not None:  # stable keys, so the training filters may select a subset
                        entry.update(record_key=get_record_key(start_ind + data_id))  # repeat is disabled
                    datalist.append(orjson.dumps(entry).decode('utf-8') + '\n')
                datalist = ''.join(datalist).encode('utf-8')
                if cache_datalist_path.endswith('.jsonl.gz'):
                    with gzip.GzipFile(fileobj=bytesio, mode='wb') as f:
                        f.write(datalist)
                else:
                    bytesio.write(datalist)
            elif cache_datalist_path.endswith('.json'):
                datalist = [f'{data_id:012d}' for data_id in range(ori_len)]
                bytesio.write(orjson.dumps(datalist))
            else:
                raise ValueError('Datalist file must be .jsonl, .jsonl.gz or .json')

            datalist_file_client.put(bytesio.getvalue(), cache_datalist_path)
            logger.info(f'Wrote datalist to {cache_datalist_path}')

        if world_size > 1:
            dist.barrier()

    return


if __name__ == '__main__':
    main()
