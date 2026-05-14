# Copyright (c) 2026 Hansheng Chen

import os
import sys
import json
import shutil
import logging
import pickle
import warnings
import hashlib
from abc import ABC, abstractmethod
from copy import deepcopy
from contextlib import contextmanager, redirect_stdout, nullcontext
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy import linalg
from scipy.stats import entropy

import torch
import torch.distributed as dist
import torch.nn.functional as F
import mmcv
from datasets import DatasetDict, load_dataset
from mmcv.runner import get_dist_info, load_checkpoint
from open_clip import get_tokenizer, create_model
from lakonlab.utils import get_root_logger
from lakonlab.utils.io_utils import download_from_huggingface, download_from_url
from .builder import METRICS
from .precision_recall import compute_pr_score, compute_pr_score_distributed

TERO_INCEPTION_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'  # noqa

# Global caches for model loading
_inception_cache = {}
_hpsv2_cache = {}
_clip_cache = {}


def _argv_ctx(argv):
    class _Argv:

        def __enter__(self):
            self._old = sys.argv
            sys.argv = argv

        def __exit__(self, exc_type, exc, tb):
            sys.argv = self._old

    return _Argv()


def _redirect_stdout(to_buf):
    return redirect_stdout(to_buf) if to_buf is not None else nullcontext()


@contextmanager
def _quarantine_openclip_logging():
    """
    Guard against open_clip (and friends) mutating global logging.
    Snapshots root handlers/level, runs the block, then removes any
    NEW handlers and restores the level. Also disables propagation
    for the open_clip logger so logs don’t bubble to root.
    """
    root = logging.getLogger()
    before_handlers = tuple(root.handlers)   # snapshot by identity
    before_ids = {id(h) for h in before_handlers}
    before_level = root.level

    try:
        yield
    finally:
        # Remove only handlers that were added during the block
        for h in list(root.handlers):
            if id(h) not in before_ids:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        root.setLevel(before_level)

        # Clamp open_clip logger so it won’t re-emit to root
        oc = logging.getLogger("open_clip")
        oc.propagate = False
        oc.handlers.clear()


def _load_inception_from_path(inception_path, map_location=None):
    mmcv.print_log(
        'Try to load Tero\'s Inception Model from '
        f'\'{inception_path}\'.', 'lakonlab')
    model = torch.jit.load(inception_path, map_location=map_location)
    mmcv.print_log('Load Tero\'s Inception Model successfully.', 'lakonlab')
    return model


def _load_inception_from_url(inception_url, map_location=None):
    inception_url = inception_url if inception_url else TERO_INCEPTION_URL
    mmcv.print_log(f'Try to download Inception Model from {inception_url}...',
                   'lakonlab')
    path = download_from_url(inception_url)
    mmcv.print_log('Download Finished.')
    return _load_inception_from_path(path, map_location=map_location)


def load_inception(inception_args, metric, map_location=None):
    if not isinstance(inception_args, dict):
        raise TypeError('Receive invalid \'inception_args\': '
                        f'\'{inception_args}\'')

    # Create cache key from arguments
    cache_key = hashlib.md5(str(sorted(inception_args.items())).encode()).hexdigest()
    cache_key += f"_{metric}"
    
    # Check if model is already cached
    if cache_key in _inception_cache:
        return _inception_cache[cache_key]

    _inception_args = deepcopy(inception_args)
    inceptoin_type = _inception_args.pop('type', None)

    if inceptoin_type != 'StyleGAN':
        raise NotImplementedError

    # try to load Tero's version
    path = _inception_args.get('inception_path', TERO_INCEPTION_URL)

    # try to parse `path` as web url and download
    if 'http' not in path:
        model = _load_inception_from_path(path, map_location=map_location)
        if isinstance(model, torch.nn.Module):
            result = model, 'StyleGAN'
            _inception_cache[cache_key] = result
            return result

    # try to parse `path` as path on disk
    model = _load_inception_from_url(path, map_location=map_location)
    if isinstance(model, torch.nn.Module):
        result = model, 'StyleGAN'
        _inception_cache[cache_key] = result
        return result

    raise RuntimeError('Cannot Load Inception Model, please check the input '
                       f'`inception_args`: {inception_args}')


def load_hpsv2(hps_version, device='cpu', precision='fp16'):
    assert hps_version in ['v2', 'v2.1']
    
    # Create cache key from arguments
    cache_key = f"{hps_version}_{device}_{precision}"
    
    # Check if model is already cached
    if cache_key in _hpsv2_cache:
        return _hpsv2_cache[cache_key]

    with _quarantine_openclip_logging():
        model = create_model(
            'ViT-H-14-quickgelu',
            precision=precision,
            device=device,
            output_dict=True)
        model.requires_grad_(False)
        tokenizer = get_tokenizer('ViT-H-14')
    load_checkpoint(
        model,
        f'huggingface://xswu/HPSv2/HPS_{hps_version}_compressed.pt',
        map_location='cpu', strict=True)

    result = model, tokenizer
    _hpsv2_cache[cache_key] = result
    return result


def load_openclip(
        model_name='ViT-L-14-336-quickgelu',
        pretrained='openai',
        device='cpu',
        precision='fp16'):
    cache_key = f'{model_name}_{pretrained}_{device}_{precision}'
    if cache_key in _clip_cache:
        return _clip_cache[cache_key]

    with _quarantine_openclip_logging():
        model = create_model(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
            output_dict=True)
        model.requires_grad_(False)
        tokenizer = get_tokenizer(model_name)
    _clip_cache[cache_key] = (model, tokenizer)
    return _clip_cache[cache_key]


class Metric(ABC):
    """The abstract base class of metrics. Basically, we split calculation into
    three steps. First, we initialize the metric object and do some
    preparation. Second, we will feed the real and fake images into metric
    object batch by batch, and we calculate intermediate results of these
    batches. Finally, We use these intermediate results to summarize the final
    result. And the result as a string can be obtained by property
    'result_str'.

    Args:
        num_images (int): The number of real/fake images needed to calculate
            metric.
        image_shape (tuple): Shape of the real/fake images with order "CHW".
    """

    def __init__(self, num_images, image_shape=None):
        self.num_images = num_images
        self.image_shape = image_shape
        self.num_real_need = num_images
        self.num_fake_need = num_images
        self.num_real_feeded = 0  # record of the fed real images
        self.num_fake_feeded = 0  # record of the fed fake images
        self._result_str = None  # string of metric result

    @property
    def result_str(self):
        """Get results in string format.

        Returns:
            str: results in string format
        """
        if not self._result_str:
            self.summary()
            return self._result_str

        return self._result_str

    def feed(self, batch, mode):
        """Feed a image batch into metric calculator and perform intermediate
        operation in 'feed_op' function.

        Args:
            batch (Tensor | dict): Images or dict to be fed into
                metric object. If ``Tensor`` is passed, the order of ``Tensor``
                should be "NCHW". If ``dict`` is passed, each term in the
                ``dict`` are ``Tensor`` with order "NCHW".
            mode (str): Mark the batch as real or fake images. Value can be
                'reals' or 'fakes',
        """
        _, ws = get_dist_info()
        if mode == 'reals':
            if self.num_real_feeded == self.num_real_need:
                return 0

            if isinstance(batch, dict):
                batch_size = [v for v in batch.values()][0].shape[0]
                end = min(batch_size,
                          self.num_real_need - self.num_real_feeded)
                batch_to_feed = {k: v[:end, ...] for k, v in batch.items()}
            else:
                batch_size = batch.shape[0]
                end = min(batch_size,
                          self.num_real_need - self.num_real_feeded)
                batch_to_feed = batch[:end, ...]

            global_end = min(batch_size * ws,
                             self.num_real_need - self.num_real_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_real_feeded += global_end
            return end

        elif mode == 'fakes':
            if self.num_fake_feeded == self.num_fake_need:
                return 0

            batch_size = batch.shape[0]
            end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
            if isinstance(batch, dict):
                batch_to_feed = {k: v[:end, ...] for k, v in batch.items()}
            else:
                batch_to_feed = batch[:end, ...]

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end
        else:
            raise ValueError(
                'The expected mode should be set to \'reals\' or \'fakes\','
                f'but got \'{mode}\'')

    def check(self):
        """Check the numbers of image."""
        assert self.num_real_feeded == self.num_fake_feeded == self.num_images

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """please implement in subclass."""

    @abstractmethod
    def feed_op(self, batch, mode):
        """please implement in subclass."""

    @abstractmethod
    def summary(self):
        """please implement in subclass."""


@METRICS.register_module()
class InceptionMetrics(Metric):
    name = 'InceptionMetrics'

    def __init__(self,
                 num_images=None,
                 reference_pkl=None,
                 bgr2rgb=False,
                 center_crop=False,  # SDXL-Lightning patch FID
                 resize=True,
                 resize_mode='bicubic',
                 inception_args=dict(
                    type='StyleGAN',
                    inception_path=TERO_INCEPTION_URL),
                 use_kid=False,
                 use_pr=True,
                 use_is=True,
                 kid_num_subsets=100,
                 kid_max_subset_size=1000,
                 pr_k=3,
                 pr_row_batch_size=10000,
                 pr_col_batch_size=10000,
                 is_splits=10,
                 is_shuffle=False,
                 prefix=''):
        super().__init__(num_images)
        self.reference_pkl = reference_pkl
        self.real_feats = []
        self.fake_feats = []
        self.preds = []
        self.real_mean = None
        self.real_cov = None
        self.bgr2rgb = bgr2rgb
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_mode
        self.device = 'cpu'

        if self.center_crop and self.resize:
            warnings.warn('`center_crop` is set to True, `resize` will be ignored.')
        if self.resize_mode not in ('bicubic', 'bilinear'):
            raise ValueError(f'Unsupported resize_mode: {self.resize_mode}')

        logger = get_root_logger()
        ori_level = logger.level
        logger.setLevel('ERROR')
        self.inception_net, self.inception_style = load_inception(
            inception_args, 'FID', map_location=self.device)
        logger.setLevel(ori_level)

        self.inception_net.eval()

        self.use_kid = use_kid
        self.use_pr = use_pr
        self.use_is = use_is
        self.kid_num_subsets = kid_num_subsets
        self.kid_max_subset_size = kid_max_subset_size
        self.real_feats_np = None

        self.pr_k = pr_k
        self.pr_row_batch_size = pr_row_batch_size
        self.pr_col_batch_size = pr_col_batch_size
        self.cached_precision = None
        self.cached_recall = None

        self.is_splits = is_splits
        self.is_shuffle = is_shuffle

        self.prefix = prefix

    def prepare(self):
        self.real_feats = []
        self.real_feats_np = None
        self.fake_feats = []
        self.preds = []
        self.cached_precision = None
        self.cached_recall = None
        if self.reference_pkl is not None:
            assert mmcv.is_filepath(self.reference_pkl)
            if self.reference_pkl.startswith('huggingface://'):
                self.reference_pkl = download_from_huggingface(self.reference_pkl)
            elif self.reference_pkl.startswith(('http://', 'https://')):
                self.reference_pkl = download_from_url(self.reference_pkl)
            with open(self.reference_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                self.real_feats_np = reference['real_feats_np']
                self.real_feats = [torch.from_numpy(reference['real_feats_np'])]
                self.num_real_feeded = reference['num_real_feeded']

    def _gather_feats(self):
        real_feats = torch.cat(self.real_feats, dim=0)
        fake_feats = torch.cat(self.fake_feats, dim=0)
        if self.num_images is not None:
            assert fake_feats.shape[0] >= self.num_images
            fake_feats = fake_feats[:self.num_images]
            if self.reference_pkl is None:
                assert real_feats.shape[0] >= self.num_images
                real_feats = real_feats[:self.num_images]
        return real_feats.contiguous(), fake_feats.contiguous()

    def _broadcast_feature_matrix(self, features):
        rank = dist.get_rank()

        if rank == 0:
            shape = torch.tensor(features.shape, device=self.device, dtype=torch.long)
        else:
            shape = torch.empty(2, device=self.device, dtype=torch.long)
        dist.broadcast(shape, src=0)
        num_rows, num_cols = shape.tolist()

        if rank == 0:
            features = features.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
        else:
            features = torch.empty((num_rows, num_cols), device=self.device, dtype=torch.float32)
        dist.broadcast(features, src=0)
        return features

    def _maybe_finalize_pr(self):
        if (self.use_pr and self.num_images is not None and self.device == 'cuda'
                and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
                and (self.cached_precision is None or self.cached_recall is None)
                and self.num_real_feeded >= self.num_real_need and self.num_fake_feeded >= self.num_fake_need):
            rank = dist.get_rank()

            real_feats = fake_feats = None
            if rank == 0:
                real_feats, fake_feats = self._gather_feats()

            real_feats = self._broadcast_feature_matrix(real_feats)
            fake_feats = self._broadcast_feature_matrix(fake_feats)

            precision = compute_pr_score_distributed(
                real_feats,
                fake_feats,
                pr_k=self.pr_k,
                pr_row_batch_size=self.pr_row_batch_size,
                pr_col_batch_size=self.pr_col_batch_size)
            recall = compute_pr_score_distributed(
                fake_feats,
                real_feats,
                pr_k=self.pr_k,
                pr_row_batch_size=self.pr_row_batch_size,
                pr_col_batch_size=self.pr_col_batch_size)

            self.cached_precision = precision
            self.cached_recall = recall

    @staticmethod
    def _calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
        """Refer to the implementation from:

        https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
        """
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm(
                (sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(
            real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return fid, mean_norm, trace

    @staticmethod
    def _calc_kid(real_feat, fake_feat, num_subsets, max_subset_size):
        """Refer to the implementation from:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py#L18  # noqa
        Args:
            real_feat (np.array): Features of the real samples.
            fake_feat (np.array): Features of the fake samples.
            num_subsets (int): Number of subsets to calculate KID.
            max_subset_size (int): The max size of each subset.
        Returns:
            float: The calculated kid metric.
        """
        n = real_feat.shape[1]
        m = min(min(real_feat.shape[0], fake_feat.shape[0]), max_subset_size)
        t = 0
        for _ in range(num_subsets):
            x = fake_feat[np.random.choice(
                fake_feat.shape[0], m, replace=False)]
            y = real_feat[np.random.choice(
                real_feat.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
            b = (x @ y.T / n + 1)**3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    def extract_features(self, batch):
        if self.center_crop:
            crop_size = 299
            h, w = batch.shape[2], batch.shape[3]
            assert h >= crop_size and w >= crop_size
            h_offset = (h - crop_size) // 2
            w_offset = (w - crop_size) // 2
            batch = batch[:, :, h_offset:h_offset + crop_size, w_offset:w_offset + crop_size]
        elif self.resize:
            batch = F.interpolate(
                batch, size=(299, 299), mode=self.resize_mode,
                align_corners=False, antialias=self.resize_mode == 'bicubic').clamp(min=-1, max=1)
        assert self.inception_style == 'StyleGAN'
        batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        feat = self.inception_net(batch, return_features=True)
        pred = F.linear(feat, self.inception_net.output.weight).softmax(dim=1)
        return feat, pred

    @torch.no_grad()
    def feed_op(self, batch, mode):
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0]]
        batch = batch.to(self.device)

        feat, pred = self.extract_features(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(feat) for _ in range(ws)]
            dist.all_gather(placeholder, feat)
            feat = torch.stack(placeholder, dim=1).reshape(feat.size(0) * ws, *feat.shape[1:])
            if mode == 'fakes':
                placeholder = [torch.zeros_like(pred) for _ in range(ws)]
                dist.all_gather(placeholder, pred)
                pred = torch.stack(placeholder, dim=1).reshape(pred.size(0) * ws, *pred.shape[1:])

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            if mode == 'reals':
                self.real_feats.append(feat.cpu())
            elif mode == 'fakes':
                self.fake_feats.append(feat.cpu())
                self.preds.append(pred.cpu().numpy())
            else:
                raise ValueError(
                    f"The expected mode should be set to 'reals' or 'fakes,\
                    but got '{mode}'")

    def feed(self, batch, mode):
        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()
            if mode == 'reals':
                if self.num_real_feeded == self.num_real_need:
                    return 0

                if isinstance(batch, dict):
                    batch_size = len(list(batch.values())[0])
                    end = min(batch_size, self.num_real_need - self.num_real_feeded)
                    batch_to_feed = {k: v[:end] for k, v in batch.items()}
                else:
                    batch_size = batch.shape[0]
                    end = min(batch_size, self.num_real_need - self.num_real_feeded)
                    batch_to_feed = batch[:end]

                global_end = min(batch_size * ws,
                                 self.num_real_need - self.num_real_feeded)
                self.feed_op(batch_to_feed, mode)
                self.num_real_feeded += global_end
                self._maybe_finalize_pr()
                return end

            elif mode == 'fakes':
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
                self._maybe_finalize_pr()
                return end

            else:
                raise ValueError(
                    'The expected mode should be set to \'reals\' or \'fakes\','
                    f'but got \'{mode}\'')

    @torch.no_grad()
    def summary(self):
        real_feats, fake_feats = self._gather_feats()

        if self.real_feats_np is None:
            real_feats_np = real_feats.numpy()
            self.real_feats_np = real_feats_np
            self.real_mean = np.mean(real_feats_np, 0)
            self.real_cov = np.cov(real_feats_np, rowvar=False)

        self._result_dict = dict()

        prefix = self.prefix + '_' if len(self.prefix) > 0 else ''

        # FID
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)
        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)
        self._result_dict.update({f'{prefix}fid': fid})
        _result_str = f'{prefix}FID: {fid:.4f} ({mean:.4f}/{cov:.4f})'

        # KID
        if self.use_kid:
            kid = self._calc_kid(self.real_feats_np, fake_feats_np, self.kid_num_subsets,
                                 self.kid_max_subset_size) * 1000
            self._result_dict.update({f'{prefix}kid': kid})
            _result_str += f', {prefix}KID: {kid:.4f}'
        else:
            kid = None

        # PR
        if self.use_pr:
            if self.cached_precision is not None and self.cached_recall is not None:
                precision = self.cached_precision
                recall = self.cached_recall
            else:
                real_feats = real_feats.to(device=self.device, dtype=torch.float32, non_blocking=True)
                fake_feats = fake_feats.to(device=self.device, dtype=torch.float32, non_blocking=True)
                precision = compute_pr_score(
                    real_feats,
                    fake_feats,
                    pr_k=self.pr_k,
                    pr_row_batch_size=self.pr_row_batch_size,
                    pr_col_batch_size=self.pr_col_batch_size)
                recall = compute_pr_score(
                    fake_feats,
                    real_feats,
                    pr_k=self.pr_k,
                    pr_row_batch_size=self.pr_row_batch_size,
                    pr_col_batch_size=self.pr_col_batch_size)
            self._result_dict[f'{prefix}precision'] = precision
            self._result_dict[f'{prefix}recall'] = recall
            _result_str += f', {prefix}Precision: {precision:.5f}, {prefix}Recall:{recall:.5f}'
        else:
            precision = recall = None

        # IS
        if self.use_is:
            split_scores = []
            self.preds = np.concatenate(self.preds, axis=0)
            if self.num_images is not None:
                assert self.preds.shape[0] >= self.num_images
                self.preds = self.preds[:self.num_images]
            num_preds = self.preds.shape[0]
            if self.is_shuffle:
                np.random.shuffle(self.preds)
            for k in range(self.is_splits):
                part = self.preds[k * (num_preds // self.is_splits):(k + 1) * (num_preds // self.is_splits), :]
                py = np.mean(part, axis=0)
                scores = []
                for i in range(part.shape[0]):
                    pyx = part[i, :]
                    scores.append(entropy(pyx, py))
                split_scores.append(np.exp(np.mean(scores)))
            is_mean = np.mean(split_scores)
            self._result_dict.update({f'{prefix}is': is_mean})
            _result_str += f', {prefix}IS: {is_mean:.2f}'
        else:
            is_mean = None

        self._result_str = _result_str

        return fid, kid, precision, recall, is_mean

    def clear_fake_data(self):
        self.fake_feats = []
        self.preds = []
        self.num_fake_feeded = 0
        self.cached_precision = None
        self.cached_recall = None

    def clear(self, clear_reals=False):
        self.clear_fake_data()
        if clear_reals:
            self.real_feats = []
            self.real_feats_np = None
            self.num_real_feeded = 0

    def load_to_gpu(self):
        """Move models to GPU."""
        if torch.cuda.is_available():
            self.inception_net.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        """Move models to CPU."""
        self.inception_net.cpu()
        self.device = 'cpu'


@METRICS.register_module()
class ColorStats(Metric):
    name = 'ColorStats'

    def __init__(self,
                 num_images=None):
        super().__init__(num_images)

    def prepare(self):
        self.stats = []

    @staticmethod
    def srgb_to_linear(c):
        threshold = 0.04045
        below = c <= threshold
        out = torch.where(
            below, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return out

    @staticmethod
    def linear_to_srgb(c):
        threshold = 0.0031308
        below = c <= threshold
        out = torch.where(
            below, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)
        return out

    def rgb_to_grayscale_srgb(self, img_srgb):
        img_lin = self.srgb_to_linear(img_srgb)
        R_lin, G_lin, B_lin = img_lin.unbind(dim=1)
        Y_lin = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
        gray_srgb = self.linear_to_srgb(Y_lin)
        return gray_srgb

    @staticmethod
    def srgb_to_hsv_saturation(img_srgb):
        c_max = torch.amax(img_srgb, dim=1)
        c_min = torch.amin(img_srgb, dim=1)
        delta = c_max - c_min
        sat = delta / c_max.clamp(min=1e-5)
        return sat

    def compute_stats(self, batch):
        batch = (batch / 2 + 0.5).clamp(0, 1)
        gray = self.rgb_to_grayscale_srgb(batch).flatten(1)
        contrast, brightness = torch.std_mean(gray, dim=1)
        saturation = self.srgb_to_hsv_saturation(batch).flatten(1).mean(dim=1)
        return torch.stack([brightness, contrast, saturation], dim=-1)

    @torch.no_grad()
    def feed_op(self, batch, mode):
        stats = self.compute_stats(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(stats) for _ in range(ws)]
            dist.all_gather(placeholder, stats)
            stats = torch.stack(placeholder, dim=1).reshape(stats.size(0) * ws, *stats.shape[1:])

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.stats.append(stats.cpu())

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
        stats = torch.cat(self.stats, dim=0)
        if self.num_images is not None:
            assert stats.shape[0] >= self.num_images
            stats = stats[:self.num_images]
        stats = stats.mean(dim=0)
        brightness, contrast, saturation = stats.tolist()
        self._result_dict = dict(
            brightness=brightness, contrast=contrast, saturation=saturation)
        self._result_str = f'Brightness: {brightness:.4f}, Contrast: {contrast:.4f}, Saturation: {saturation:.4f}'
        return brightness, contrast, saturation

    def clear_fake_data(self):
        self.stats = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()


@METRICS.register_module()
class HPSv2(Metric):
    name = 'HPSv2'
    requires_prompt = True

    def __init__(self,
                 num_images=None,
                 hps_version='v2.1'):
        super().__init__(num_images)
        self.hps_version = hps_version
        self.device = 'cpu'  # Initialize on CPU
        self.dtype = torch.float16
        self.model, self.tokenizer = load_hpsv2(hps_version, device=self.device, precision='fp16')
        self.model.eval()
        image_size = self.model.visual.image_size
        if isinstance(image_size, tuple):
            assert len(image_size) == 2 and image_size[0] == image_size[1]
            image_size = image_size[0]
        self.image_size = image_size
        self.image_mean = torch.tensor(self.model.visual.image_mean, device=self.device).view(3, 1, 1)
        self.image_std = torch.tensor(self.model.visual.image_std, device=self.device).view(3, 1, 1)

    def prepare(self):
        self.scores = []

    def resize(self, imgs):
        h, w = imgs.shape[2:]
        scale = self.image_size / float(max(h, w))
        if scale != 1.0:
            h = int(round(h * scale))
            w = int(round(w * scale))
            imgs = F.interpolate(imgs, size=(h, w), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        if h != w:
            pad_h = self.image_size - h
            pad_w = self.image_size - w
            imgs = F.pad(
                imgs, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)
        return imgs

    @torch.no_grad()
    def feed_op(self, batch, mode):
        imgs = batch['imgs']
        prompts = batch['prompts']

        imgs = (imgs.to(device=self.device, dtype=torch.float32) / 2 + 0.5).clamp(0, 1)
        imgs = ((self.resize(imgs) - self.image_mean) / self.image_std).to(dtype=self.dtype)
        prompts = self.tokenizer(prompts).to(device=self.device)

        outputs = self.model(imgs, prompts)
        image_features, text_features = outputs['image_features'], outputs['text_features']
        hps_scores = (image_features * text_features).sum(dim=-1)  # (bs, )

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.empty_like(hps_scores) for _ in range(ws)]
            dist.all_gather(placeholder, hps_scores)
            hps_scores = torch.stack(placeholder, dim=1).reshape(hps_scores.size(0) * ws)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.scores.append(hps_scores.float().cpu())

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
        self._result_dict = dict(hpsv2=mean_score)
        self._result_str = f'HPSv2: {mean_score:.4f}'
        return mean_score

    def clear_fake_data(self):
        self.scores = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()

    def load_to_gpu(self):
        if torch.cuda.is_available():
            self.model.cuda()
            self.image_mean = self.image_mean.cuda()
            self.image_std = self.image_std.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        self.model.cpu()
        self.image_mean = self.image_mean.cpu()
        self.image_std = self.image_std.cpu()
        self.device = 'cpu'


@METRICS.register_module()
class CLIPSimilarity(Metric):
    """
    Average image–text CLIP cosine similarity (↑ better).
    Preprocess emulates OpenAI CLIP for ViT-L/14@336:
      - Resize so min(H, W) = 336 (bicubic, antialias), keep aspect ratio
      - Center crop to 336x336
      - Normalize with model.visual.image_mean/std
    Expects batch = {'imgs': (B,3,H,W) in [-1,1], 'prompts': List[str]}
    """
    name = 'CLIPSimilarity'
    requires_prompt = True

    def __init__(
        self,
        num_images=None,
        model_name='ViT-L-14-336-quickgelu',
        pretrained='openai',
        precision='fp16',   # 'fp16' | 'fp32' | 'bf16'
    ):
        super().__init__(num_images)
        self.model_name = model_name
        self.pretrained = pretrained
        self.precision = precision

        self.device = 'cpu'
        self.dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32
        }.get(precision, torch.float16)

        self.model, self.tokenizer = load_openclip(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
            precision=precision,
        )
        self.model.eval()

        # OpenAI ViT-L/14@336 uses square 336 input
        image_size = self.model.visual.image_size
        if isinstance(image_size, tuple):
            assert len(image_size) == 2 and image_size[0] == image_size[1]
            image_size = image_size[0]
        self.image_size = int(image_size)  # 336

        # Use the model's own stats for normalization
        self.image_mean = torch.tensor(self.model.visual.image_mean, device=self.device).view(3, 1, 1)
        self.image_std = torch.tensor(self.model.visual.image_std, device=self.device).view(3, 1, 1)

    def prepare(self):
        self.scores = []

    def _resize_min_side_then_center_crop(self, imgs):
        """
        imgs: (B,3,H,W) in [0,1], float32, on self.device
        1) Resize so min(H,W) == self.image_size, preserve AR (bicubic, antialias)
        2) Center-crop to (self.image_size, self.image_size)
        3) Normalize with model mean/std
        4) Cast to self.dtype
        """
        _, _, H, W = imgs.shape
        target = self.image_size

        # Scale factor so that the shorter side becomes 'target'
        short, long = (H, W) if H < W else (W, H)
        if short == 0:
            raise ValueError("Invalid image with zero dimension.")
        scale = target / float(short)

        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        if new_h != H or new_w != W:
            imgs = F.interpolate(
                imgs, size=(new_h, new_w),
                mode='bicubic', align_corners=False, antialias=True
            ).clamp(0, 1)

        # Center crop to target x target
        top = max(0, (new_h - target) // 2)
        left = max(0, (new_w - target) // 2)
        imgs = imgs[:, :, top:top + target, left:left + target]

        imgs = (imgs - self.image_mean) / self.image_std
        return imgs.to(dtype=self.dtype)

    @torch.no_grad()
    def feed_op(self, batch, mode):
        if mode == 'reals':
            return 0

        imgs = batch['imgs']
        prompts = batch['prompts']

        # [-1,1] -> [0,1]
        imgs = (imgs.to(device=self.device, dtype=torch.float32) / 2 + 0.5).clamp(0, 1)
        imgs = self._resize_min_side_then_center_crop(imgs)

        # Tokenize on device
        text = self.tokenizer(prompts).to(device=self.device)

        # Forward (create_model(..., output_dict=True)) => dict w/ features
        out = self.model(imgs, text)
        if isinstance(out, dict) and ('image_features' in out and 'text_features' in out):
            img_feat = out['image_features']
            txt_feat = out['text_features']
        else:
            img_feat = self.model.encode_image(imgs)
            txt_feat = self.model.encode_text(text)

        # Cosine similarity per pair
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        sim = (img_feat * txt_feat).sum(dim=-1).to(torch.float32)  # (B,)

        # DDP gather
        if dist.is_initialized():
            ws = dist.get_world_size()
            bucket = [torch.empty_like(sim) for _ in range(ws)]
            dist.all_gather(bucket, sim)
            sim = torch.stack(bucket, dim=1).reshape(sim.size(0) * ws)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.scores.append(sim.cpu())

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

            global_end = min(batch_size * ws, self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    @torch.no_grad()
    def summary(self):
        sims = torch.cat(self.scores, dim=0)
        if self.num_images is not None:
            assert sims.shape[0] >= self.num_images
            sims = sims[:self.num_images]
        mean_sim = sims.mean().item()

        self._result_dict = dict(clipsim=mean_sim)  # raw cosine in [-1,1]
        self._result_str = f'CLIPSim: {mean_sim:.4f}'
        return mean_sim

    def clear_fake_data(self):
        self.scores = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()

    def load_to_gpu(self):
        if torch.cuda.is_available():
            self.model.cuda()
            self.image_mean = self.image_mean.cuda()
            self.image_std = self.image_std.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        self.model.cpu()
        self.image_mean = self.image_mean.cpu()
        self.image_std = self.image_std.cpu()
        self.device = 'cpu'


class BenchmarkImageExport(Metric):
    requires_prompt = True

    def __init__(self,
                 prompt_path=None,
                 img_root_dir='work_dirs/benchmark_exports',
                 samples_per_prompt=4,
                 clear_existing=True,
                 num_images=None,
                 prompt_items=None,
                 prompt_key='prompt',
                 export_size=1024):
        if prompt_items is None:
            assert os.path.exists(prompt_path), f'prompt_path {prompt_path} does not exist.'
            prompt_items = []
            with open(prompt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompt_items.append(json.loads(line))
        self.prompt_path = prompt_path
        self.img_root_dir = img_root_dir
        self.samples_per_prompt = samples_per_prompt
        self.clear_existing = clear_existing
        self.prompt_items = prompt_items
        self.prompt_key = prompt_key
        self.export_size = export_size
        assert self.prompt_items, f'No prompts found in {prompt_path}.'
        self.prompt_list = [item[self.prompt_key] for item in self.prompt_items]
        self.img_buffer = []
        self.current_idx = 0
        self.executor = None
        super().__init__(num_images=num_images or len(self.prompt_items) * samples_per_prompt)

    def prepare(self):
        ddp_active = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp_active else 0
        self.img_buffer = []
        self.current_idx = 0
        self.executor = None
        if rank == 0:
            if self.clear_existing and os.path.exists(self.img_root_dir):
                shutil.rmtree(self.img_root_dir)
            os.makedirs(self.img_root_dir, exist_ok=True)
            self.executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 4)
        if ddp_active:
            dist.barrier()

    @abstractmethod
    def _save_prompt_images(self, imgs, prompt_item, prompt_idx):
        """please implement in subclass."""

    @torch.no_grad()
    def feed_op(self, batch, mode):
        imgs = batch['imgs']
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        if (self.export_size is not None
                and imgs.shape[-2:] != (self.export_size, self.export_size)):
            imgs = F.interpolate(
                imgs,
                size=(self.export_size, self.export_size),
                mode='bicubic',
                align_corners=False,
                antialias=True).clamp(0, 1)
        imgs = (imgs.permute(0, 2, 3, 1) * 255).round().to(dtype=torch.uint8).contiguous()
        prompts = batch['prompts']

        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.empty_like(imgs) for _ in range(ws)]
            dist.all_gather(placeholder, imgs)
            imgs = torch.stack(placeholder, dim=1).reshape(imgs.size(0) * ws, *imgs.shape[1:])

            placeholder_prompts = [None for _ in range(ws)]
            dist.all_gather_object(placeholder_prompts, prompts)
            prompts = [p for prompt_batch in zip(*placeholder_prompts) for p in prompt_batch]

        imgs = imgs.cpu().numpy()

        for img, prompt in zip(imgs, prompts):
            self.img_buffer.append((img, prompt))
            if len(self.img_buffer) == self.samples_per_prompt:
                expected_prompt = self.prompt_list[self.current_idx]
                for _, p in self.img_buffer:
                    assert p == expected_prompt, (
                        f'Unexpected prompt order at index {self.current_idx}: '
                        f'{p!r} != {expected_prompt!r}')

                if self.executor is not None:
                    imgs_to_save = np.stack([x[0] for x in self.img_buffer], axis=0)
                    prompt_item = self.prompt_items[self.current_idx]
                    prompt_idx = self.current_idx
                    self.executor.submit(
                        self._save_prompt_images, imgs_to_save, prompt_item, prompt_idx)

                self.img_buffer = []
                self.current_idx += 1
                if self.current_idx >= len(self.prompt_items):
                    break

    def feed(self, batch, mode):
        if mode == 'reals':
            return 0

        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()

            if self.num_fake_feeded == self.num_fake_need:
                return 0

            batch_size = len(list(batch.values())[0])
            end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
            batch_to_feed = {k: v[:end] for k, v in batch.items()}

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    def clear_fake_data(self):
        self.img_buffer = []
        self.current_idx = 0
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()

    @torch.no_grad()
    def summary(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        self._result_dict = dict()
        self._result_str = ''
        return


@METRICS.register_module()
class DPGBenchExport(BenchmarkImageExport):
    name = 'DPGBenchExport'

    def __init__(self,
                 prompt_path='data/dpgbench-prompts/prompts.jsonl',
                 img_root_dir='work_dirs/benchmark_exports/dpgbench',
                 samples_per_prompt=4,
                 clear_existing=True,
                 export_size=1024):
        assert samples_per_prompt == 4, 'DPGBenchExport expects 4 images for a 2x2 grid.'
        super().__init__(
            prompt_path=prompt_path,
            img_root_dir=img_root_dir,
            samples_per_prompt=samples_per_prompt,
            clear_existing=clear_existing,
            export_size=export_size)

    def _save_prompt_images(self, imgs, prompt_item, prompt_idx):
        _, h, w, c = imgs.shape
        assert h == w, (
            f'DPGBenchExport expects square samples for scorer cropping, got {h}x{w}.')
        grid = imgs.reshape(2, 2, h, w, c).transpose(
            0, 2, 1, 3, 4).reshape(2 * h, 2 * w, c)
        item_id = prompt_item.get('id', f'{prompt_idx:05d}')
        Image.fromarray(grid).save(os.path.join(self.img_root_dir, f'{item_id}.png'))


@METRICS.register_module()
class GenEvalExport(BenchmarkImageExport):
    name = 'GenEvalExport'

    def __init__(self,
                 prompt_path='data/geneval-prompts/prompts.jsonl',
                 metadata_path='data/geneval-prompts/evaluation_metadata.jsonl',
                 img_root_dir='work_dirs/benchmark_exports/geneval',
                 samples_per_prompt=4,
                 save_grid=True,
                 clear_existing=True,
                 export_size=1024):
        self.metadata_path = metadata_path
        self.save_grid = save_grid
        assert os.path.exists(metadata_path), f'metadata_path {metadata_path} does not exist.'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata_items = [json.loads(line) for line in f if line.strip()]
        super().__init__(
            prompt_path=prompt_path,
            img_root_dir=img_root_dir,
            samples_per_prompt=samples_per_prompt,
            clear_existing=clear_existing,
            export_size=export_size)
        assert len(self.metadata_items) == len(self.prompt_items), (
            f'GenEval metadata count {len(self.metadata_items)} does not match '
            f'prompt count {len(self.prompt_items)}.')

    def _save_prompt_images(self, imgs, prompt_item, prompt_idx):
        prompt_dir = os.path.join(self.img_root_dir, f'{prompt_idx:05d}')
        sample_dir = os.path.join(prompt_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        metadata = self.metadata_items[prompt_idx]
        with open(os.path.join(prompt_dir, 'metadata.jsonl'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

        for sample_idx, img in enumerate(imgs):
            Image.fromarray(img).save(os.path.join(sample_dir, f'{sample_idx:04d}.png'))

        if self.save_grid:
            grid_size = int(np.sqrt(imgs.shape[0]))
            if grid_size * grid_size == imgs.shape[0]:
                _, h, w, c = imgs.shape
                grid = imgs.reshape(grid_size, grid_size, h, w, c).transpose(
                    0, 2, 1, 3, 4).reshape(grid_size * h, grid_size * w, c)
                Image.fromarray(grid).save(os.path.join(prompt_dir, 'grid.png'))


@METRICS.register_module()
class HPSv3BenchmarkExport(BenchmarkImageExport):
    name = 'HPSv3BenchmarkExport'

    def __init__(self,
                 benchmark_dataset_kwargs,
                 img_root_dir='work_dirs/benchmark_exports/hpsv3',
                 prompt_key='caption',
                 clear_existing=True,
                 export_size=1024):
        benchmark_dataset = load_dataset(**benchmark_dataset_kwargs)
        if isinstance(benchmark_dataset, DatasetDict):
            split = 'train' if 'train' in benchmark_dataset else list(benchmark_dataset.keys())[0]
            benchmark_dataset = benchmark_dataset[split]
        prompt_items = [benchmark_dataset[i] for i in range(len(benchmark_dataset))]
        super().__init__(
            prompt_path=None,
            img_root_dir=img_root_dir,
            samples_per_prompt=1,
            clear_existing=clear_existing,
            num_images=len(prompt_items),
            prompt_items=prompt_items,
            prompt_key=prompt_key,
            export_size=export_size)

    def _save_prompt_images(self, imgs, prompt_item, prompt_idx):
        assert imgs.shape[0] == 1, 'HPSv3BenchmarkExport expects one image per prompt.'
        category = prompt_item['category']
        image_file = prompt_item['image_file']
        image_stem = os.path.splitext(os.path.basename(image_file))[0]
        category_dir = os.path.join(self.img_root_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        Image.fromarray(imgs[0]).save(os.path.join(category_dir, f'{image_stem}.png'))
        prompt = prompt_item[self.prompt_key]
        with open(os.path.join(category_dir, f'{image_stem}.txt'), 'w', encoding='utf-8') as f:
            f.write(prompt)
