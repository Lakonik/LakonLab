# Copyright (c) 2026 Hansheng Chen

import os
import re
import hashlib
import shutil
import time
import mimetypes
import tempfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from io import BytesIO
from functools import wraps
from typing import Generator, Union, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import av
import imageio
import boto3
from PIL import Image, PngImagePlugin
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from torch.hub import download_url_to_file
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import WeakFileLock

import mmcv
from mmcv.fileio import BaseStorageBackend
from mmcv.fileio import FileClient as MMCVFileClient
from mmcv.fileio.file_client import HTTPBackend
from mmcv.fileio.file_client import HardDiskBackend as MMCVHardDiskBackend


S3_MULTIPART_THRESHOLD = 5 * 2**30  # 5GB
S3_MULTIPART_CHUNKSIZE = 5 * 2**30  # 5GB
VIDEO_EXTENSIONS = (
    '.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv', '.flv', '.ogv')

# Pillow's decompression-bomb guard warns past 89 MP and refuses past 179 MP, which would spam a
# dataloader worker and then fail a sample outright.
# Loading with a target size drafts them down before decoding, so the memory stays bounded.
Image.MAX_IMAGE_PIXELS = None
# the same guard applies to a png's text and profile chunks, where a fat embedded ICC profile fails
# the sample while `_open` is still reading the header. These are our own shards, not hostile input.
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 1024 * 1024 * 1024

TMP_DIR = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
S3_TRANSFER_CONFIG = TransferConfig(
    preferred_transfer_client='classic',  # avoiding crt, which may be incompatible with `fork` in dataloader workers
    multipart_threshold=S3_MULTIPART_THRESHOLD,
    multipart_chunksize=S3_MULTIPART_CHUNKSIZE,
)

LAKONLAB_CACHE = os.getenv('LAKONLAB_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'lakonlab'))
AWS_SHARED_CREDENTIALS_FILE = os.path.join(os.path.expanduser('~'), '.aws', 'credentials')

HF_REPO_TYPES = {
    'datasets': 'dataset',
    'spaces': 'space'
}


class _S3ObjectChangedError(RuntimeError):
    pass


def retry(tries=5, delay=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == tries:
                        print(f"Attempt {attempt} failed: {e}. No more retries.")
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


def _refresh_s3_client(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._refresh_client_if_stale()
        return func(self, *args, **kwargs)
    return wrapper


@retry()
def _download_from_url(url, dest_path, hash_prefix):
    download_url_to_file(url, dest_path, hash_prefix, progress=True)


@contextmanager
def locked_cache_path(path, expected_size=None):
    path = str(path)
    with WeakFileLock(f'{path}.lock'):
        if os.path.exists(path):
            if (expected_size is None
                    or os.path.getsize(path) == expected_size):
                yield path, True
                return
            os.remove(path)

        tmp_path = f'{path}.tmp.{os.getpid()}'
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        elif os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            yield tmp_path, False
            os.rename(tmp_path, path)
        finally:
            if os.path.isdir(tmp_path):
                shutil.rmtree(tmp_path, ignore_errors=True)
            elif os.path.exists(tmp_path):
                os.remove(tmp_path)


def download_from_url(url,
                      dest_path=None,
                      dest_dir=None,
                      hash_prefix=None):
    """Modified from MMGeneration.
    """
    if dest_dir is None:
        dest_dir = os.path.join(LAKONLAB_CACHE, 'downloads')

    # get the exact destination path
    if dest_path is None:
        filename = url.split('/')[-1]
        dest_path = os.path.join(dest_dir, filename)

    if dest_path.startswith('~'):
        dest_path = os.path.expanduser('~') + dest_path[1:]

    # advoid downloading existing file
    if os.path.exists(dest_path):
        return dest_path

    _dir = os.path.dirname(dest_path)
    mmcv.mkdir_or_exist(_dir)
    with locked_cache_path(dest_path) as (local_path, exists):
        if not exists:
            _download_from_url(url, local_path, hash_prefix)

    return dest_path


@retry()
def download_from_huggingface(filename):
    filename = filename.removeprefix('huggingface://').split('/')
    repo_type = None
    if filename[0] in HF_REPO_TYPES:
        repo_type = HF_REPO_TYPES[filename[0]]
        filename = filename[1:]
    repo_id = '/'.join(filename[:2])
    repo_filename = '/'.join(filename[2:])
    cached_file = hf_hub_download(repo_id=repo_id, filename=repo_filename, repo_type=repo_type)
    return cached_file


class HardDiskBackend(MMCVHardDiskBackend):

    @staticmethod
    def get_range(filepath: Union[str, Path], offset: int, length: int) -> bytes:
        with open(filepath, 'rb') as f:
            f.seek(offset)
            return f.read(length)


class S3Backend(BaseStorageBackend):

    _allow_symlink = True

    def __init__(self, anonymous: bool = False):
        self.anonymous = anonymous
        self._credentials_file = None
        self._credentials_file_mtime = None
        self._client = self._new_client()

    def __del__(self):
        if hasattr(self, '_client'):
            self._client.close()

    def _new_client(self):
        self._pid = os.getpid()
        region = os.getenv('AWS_REGION')
        profile = os.getenv('AWS_PROFILE')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        session_token = os.getenv('AWS_SESSION_TOKEN')
        credentials_file = (
            os.getenv('AWS_SHARED_CREDENTIALS_FILE')
            or AWS_SHARED_CREDENTIALS_FILE
        )
        self._credentials_file = credentials_file
        self._credentials_file_mtime = self._get_credentials_file_mtime()

        if self.anonymous:
            config = Config(region_name=region, signature_version=UNSIGNED)
            return boto3.Session().client('s3', config=config)

        config = Config(region_name=region)
        if profile:
            session = boto3.Session(profile_name=profile)
            return session.client('s3', config=config)

        client_kwargs = dict(config=config)
        if access_key and secret_key:
            client_kwargs.update(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
            )
        return boto3.Session().client('s3', **client_kwargs)

    def _get_credentials_file_mtime(self):
        if not self._credentials_file or not os.path.exists(self._credentials_file):
            return None
        return os.path.getmtime(self._credentials_file)

    def _refresh_client(self) -> None:
        self._client.close()
        self._client = self._new_client()

    def _refresh_client_if_stale(self) -> None:
        # botocore does not support sharing a client, or its open sockets, across a fork, so a
        # worker process builds its own on first use
        if self._pid != os.getpid():
            self._client = self._new_client()
            return
        credentials_file = (
            os.getenv('AWS_SHARED_CREDENTIALS_FILE')
            or AWS_SHARED_CREDENTIALS_FILE
        )
        if credentials_file != self._credentials_file:
            self._credentials_file = credentials_file
            self._refresh_client()
            return
        mtime = self._get_credentials_file_mtime()
        if mtime != self._credentials_file_mtime:
            self._refresh_client()

    @staticmethod
    def _split_s3_url(s3_url):
        s3_url = s3_url.removeprefix('s3://')
        bucket, _, prefix = s3_url.partition('/')
        return bucket, prefix

    @staticmethod
    def _infer_s3_extra_args(filepath: str) -> dict:
        extra_args = dict()
        ctype, enc = mimetypes.guess_type(str(filepath), strict=False)
        if ctype is not None:
            extra_args.update(ContentType=ctype)
        if enc == 'gzip':
            extra_args['ContentEncoding'] = 'gzip'
        return extra_args

    @staticmethod
    def _format_path(filepath: str) -> str:
        return re.sub(r'\\+', '/', filepath)

    @retry()
    @_refresh_s3_client
    def get(self, filepath: Union[str, Path]) -> bytes:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        bytesio = BytesIO()
        self._client.download_fileobj(bucket, prefix, bytesio, Config=S3_TRANSFER_CONFIG)
        bytesio.seek(0)
        return bytesio.read()

    @retry()
    @_refresh_s3_client
    def get_range(self, filepath: Union[str, Path], offset: int, length: int) -> bytes:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        response = self._client.get_object(
            Bucket=bucket, Key=prefix, Range=f'bytes={offset}-{offset + length - 1}')
        return response['Body'].read()

    def get_text(self, filepath: Union[str, Path], encoding: str = 'utf-8') -> str:
        return self.get(filepath).decode(encoding)

    @retry()
    @_refresh_s3_client
    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        extra_args = self._infer_s3_extra_args(filepath)
        if len(obj) < S3_MULTIPART_THRESHOLD:
            self._client.upload_fileobj(
                BytesIO(obj),
                bucket,
                prefix,
                Config=S3_TRANSFER_CONFIG,
                ExtraArgs=extra_args,
            )
        else:
            ext = os.path.splitext(filepath)[-1].lower()
            with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=ext, delete=False) as tmp:
                cached_file = tmp.name
                tmp.write(obj)
            try:
                self._client.upload_file(
                    cached_file,
                    bucket,
                    prefix,
                    Config=S3_TRANSFER_CONFIG,
                    ExtraArgs=extra_args,
                )
            finally:
                os.remove(cached_file)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        self.put(bytes(obj, encoding=encoding), filepath)

    @retry()
    @_refresh_s3_client
    def remove(self, filepath: Union[str, Path]) -> None:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        self._client.delete_object(Bucket=bucket, Key=prefix)

    @retry()
    @_refresh_s3_client
    def exists(self, filepath: Union[str, Path]) -> bool:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        if filepath[-1] == '/':
            s3_objects = self._client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/',
                MaxKeys=2)
            files = []
            if 'Contents' in s3_objects:
                files += [obj['Key'] for obj in s3_objects['Contents']]
            if 'CommonPrefixes' in s3_objects:
                files += [obj['Prefix'] for obj in s3_objects['CommonPrefixes']]
            exist_status = len(files) > 0
        else:
            try:
                self._client.head_object(Bucket=bucket, Key=prefix)
                exist_status = True
            except ClientError as e:
                code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
                err = e.response.get('Error', {}).get('Code')
                if code == 404 or err in ('404', 'NoSuchKey', 'NotFound'):
                    exist_status = False
                else:
                    raise
        return exist_status

    def isdir(self, filepath: Union[str, Path]) -> bool:
        filepath = str(filepath)
        if not filepath.endswith('/'):
            filepath += '/'
        return self.exists(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        filepath = str(filepath)
        return filepath[-1] != '/' and self.exists(filepath)

    def join_path(self, filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        filepath = self._format_path(filepath)
        if filepath.endswith('/'):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_paths.append(self._format_path(path))
        return '/'.join(formatted_paths)

    @retry()
    @_refresh_s3_client
    def download_file(self, filepath: Union[str, Path], local_path: Union[str, Path]) -> None:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        self._client.download_file(
            bucket,
            prefix,
            str(local_path),
            Config=S3_TRANSFER_CONFIG,
        )

    @retry()
    @_refresh_s3_client
    def upload_file(self, local_path: Union[str, Path], filepath: Union[str, Path]) -> None:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        extra_args = self._infer_s3_extra_args(filepath)
        self._client.upload_file(
            str(local_path),
            bucket,
            prefix,
            Config=S3_TRANSFER_CONFIG,
            ExtraArgs=extra_args,
        )

    @retry()
    @_refresh_s3_client
    def _head_object(self, filepath: Union[str, Path]) -> dict:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        return self._client.head_object(Bucket=bucket, Key=prefix)

    # Retry if the S3 object is overwritten between the initial HEAD and the
    # download, so new contents are not cached under the previous ETag.
    @retry(exceptions=(_S3ObjectChangedError,))
    def _get_cached_file(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        filepath_hash = hashlib.sha256(filepath.encode()).hexdigest()
        suffix = os.path.splitext(filepath)[-1]

        metadata = self._head_object(filepath)
        etag = metadata['ETag']
        content_length = metadata['ContentLength']
        revision = hashlib.sha256(f'{etag}:{content_length}'.encode()).hexdigest()
        cache_dir = os.path.join(LAKONLAB_CACHE, 's3', filepath_hash)
        cached_file = os.path.join(cache_dir, revision + suffix)
        mmcv.mkdir_or_exist(cache_dir)

        with locked_cache_path(
                cached_file, expected_size=content_length) as (local_path, exists):
            if not exists:
                self.download_file(filepath, local_path)
                current_metadata = self._head_object(filepath)
                if (current_metadata['ETag'] != etag
                        or current_metadata['ContentLength'] != content_length):
                    raise _S3ObjectChangedError(
                        f'S3 object changed while downloading: {filepath}')
                if os.path.getsize(local_path) != content_length:
                    raise IOError(
                        f'Downloaded file size does not match S3 '
                        f'object: {filepath}')

        return cached_file

    @contextmanager
    def get_local_path(
            self,
            filepath: Union[str, Path]) -> Generator[Union[str, Path], None, None]:
        yield self._get_cached_file(filepath)

    @retry()
    @_refresh_s3_client
    def list_dir_or_file(
            self,
            dir_path: Union[str, Path],
            list_dir: bool = True,
            list_file: bool = True,
            suffix: Optional[Union[str, Tuple[str]]] = None,
            recursive: bool = False):
        assert list_dir and list_file and suffix is None

        dir_path = str(dir_path)
        if not dir_path.endswith('/'):
            dir_path += '/'

        bucket, prefix = self._split_s3_url(dir_path)
        prefix_len = len(prefix)
        names = []
        paginator = self._client.get_paginator('list_objects_v2')
        pagination_config = dict(Bucket=bucket, Prefix=prefix)
        if not recursive:
            pagination_config['Delimiter'] = '/'

        for page in paginator.paginate(**pagination_config):
            if not recursive:
                for obj in page.get('CommonPrefixes', []):
                    key = obj['Prefix']
                    name = key[prefix_len:].rstrip('/')
                    if name:
                        names.append(name)

            for obj in page.get('Contents', []):
                key = obj['Key']
                name = key[prefix_len:]
                if name:
                    names.append(name)

        return names


class HuggingFaceBackend(BaseStorageBackend):

    def get(self, filepath):
        local_path = download_from_huggingface(filepath)
        with open(local_path, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath, encoding='utf-8'):
        local_path = download_from_huggingface(filepath)
        with open(local_path, encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    @contextmanager
    def get_local_path(self, filepath: str):
        yield download_from_huggingface(filepath)


class FileClient(MMCVFileClient):
    """mmcv's FileClient plus `get_range`, so shard formats can address one record without
    fetching the whole object.

    The registry is declared here rather than registered onto mmcv's, so that mmcv's own
    client is left alone and the backends needing infrastructure we do not have are dropped
    (ceph, memcached, lmdb, and petrel, which mmcv maps `s3` to by default). mmcv keys its
    instance cache by backend/prefix only, so a private cache keeps this subclass from
    handing back a base-class instance created elsewhere in the process.
    """

    _backends = {
        'disk': HardDiskBackend,
        'http': HTTPBackend,
        's3': S3Backend,
        'huggingface': HuggingFaceBackend,
    }

    _prefix_to_backends = {
        'http': HTTPBackend,
        'https': HTTPBackend,
        's3': S3Backend,
        'huggingface': HuggingFaceBackend,
    }

    _instances = dict()

    def get_range(self, filepath: Union[str, Path], offset: int, length: int) -> bytes:
        return self.client.get_range(filepath, offset, length)


def save_image(image, filepath, file_client):
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    file_client.put(buffer.getvalue(), filepath)


def save_video(video, filepath, file_client, fps=16, quality=5, bitrate=None, macro_block_size=1):
    imageio.plugins.ffmpeg.get_exe()
    buffer = BytesIO()
    with imageio.get_writer(
            buffer, format='mp4', mode='I', fps=fps,
            quality=quality, bitrate=bitrate, macro_block_size=macro_block_size) as writer:
        for frame in video:
            writer.append_data(frame)
    file_client.put(buffer.getvalue(), filepath)


def rgb_to_yuv420p(video):
    """Convert float RGB video [T,C,H,W] in [0, 1] to uint8 planar YUV420p [T,H*3/2,W] (the Y plane, then the U and
    V planes), with the BT.601 limited-range matrix FFmpeg's swscale applies to rgb24 input by default. Running it on
    the GPU spares the CPU conversion and halves the host copy."""
    yuv = torch.einsum(
        'ij,tjhw->tihw',
        video.new_tensor([
            [65.481, 128.553, 24.966],
            [-37.797, -74.203, 112.0],
            [112.0, -93.786, -18.214]
        ]),
        video
    ) + video.new_tensor([16.0, 128.0, 128.0]).view(1, 3, 1, 1)
    y, uv = yuv[:, 0], F.avg_pool2d(yuv[:, 1:], 2)
    t, h, w = y.shape
    return torch.cat([y, uv.reshape(t, h // 2, w)], dim=1).round().clamp(0, 255).to(torch.uint8)


def save_video_audio(video, audio, filepath, file_client, fps, sample_rate, pix_fmt='rgb24', options=None,
                     metadata=None):
    """Save uint8 video and float audio [C,S] as an H.264/AAC MP4. The video is RGB [T,H,W,C] with
    ``pix_fmt='rgb24'``, or planar YUV420p [T,H*3/2,W] (see ``rgb_to_yuv420p``) with ``pix_fmt='yuv420p'``.
    ``options`` are libx264 options with string values, e.g. ``{'preset': 'veryfast', 'crf': '18'}`` (libx264's own
    defaults, preset medium and crf 23, if None). ``metadata`` are container tags with string values, e.g.
    ``{'comment': '...'}``. The index is placed before the
    media data (faststart), so playback can start while the file downloads. Faststart rewrites the finished file,
    so only local and S3 paths are supported: a local file is written in place, an S3 one uploads a temporary file."""
    local = isinstance(file_client.client, HardDiskBackend)
    if local:
        mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(filepath)))
        path = str(filepath)
    elif isinstance(file_client.client, S3Backend):
        with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix='.mp4', delete=False) as tmp:
            path = tmp.name
    else:
        raise ValueError('save_video_audio only supports local and S3 paths.')
    try:
        with av.open(path, mode='w', format='mp4', options={'movflags': 'faststart'}) as container:
            container.metadata.update(metadata or {})
            video_stream = container.add_stream(
                'libx264', rate=fps, options=options)
            video_stream.width = video.shape[2]
            video_stream.height = video.shape[1] if pix_fmt == 'rgb24' else video.shape[1] * 2 // 3
            video_stream.pix_fmt = 'yuv420p'
            audio_stream = container.add_stream('aac', rate=sample_rate)
            audio_stream.layout = {1: 'mono', 2: 'stereo'}[audio.shape[0]]
            for frame in video:
                for packet in video_stream.encode(av.VideoFrame.from_ndarray(frame, format=pix_fmt)):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(audio, dtype=np.float32), format='fltp', layout=audio_stream.layout)
            frame.sample_rate = sample_rate
            frame.pts = 0
            for packet in audio_stream.encode(frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
        if not local:
            file_client.client.upload_file(path, filepath)
    finally:
        if not local:
            os.remove(path)


def resize_and_crop(image: Image, target_hw: tuple[int, int], method='lanczos'):
    tgt_h, tgt_w = target_hw
    w, h = image.size
    scale = max(tgt_h / h, tgt_w / w)
    new_h, new_w = round(h * scale), round(w * scale)
    if new_h != h or new_w != w:
        image = image.resize((new_w, new_h), Image.Resampling[method.upper()])
    left, top = (new_w - tgt_w) // 2, (new_h - tgt_h) // 2
    return image.crop((left, top, left + tgt_w, top + tgt_h)), scale


def load_image_or_video(filepath_or_bytes: Union[str, Path, bytes], file_client=None,
                        target_size=None, num_frames=None, resize_method='lanczos', is_video=None,
                        frame_indices=None):
    if isinstance(filepath_or_bytes, (str, Path)):
        if file_client is None:
            file_client = FileClient.infer_client(uri=filepath_or_bytes)
        img_bytes = file_client.get(filepath_or_bytes)
        extension = os.path.splitext(filepath_or_bytes)[-1].lower()
        if is_video is None:
            is_video = extension in VIDEO_EXTENSIONS
    elif isinstance(filepath_or_bytes, bytes):
        img_bytes = filepath_or_bytes
        extension = None
        if is_video is None:
            raise ValueError('is_video must be specified when loading bytes.')
    else:
        raise TypeError(f'Expected a file path or bytes, got {type(filepath_or_bytes)}.')

    if is_video:
        video_kwargs = dict(extension=extension or '.mp4')
        if frame_indices is not None:
            frames = islice(imageio.v3.imiter(BytesIO(img_bytes), **video_kwargs), max(frame_indices) + 1)
            frames = [frame for index, frame in enumerate(frames) if index in frame_indices]
        elif num_frames is not None:  # bounded decoding of a long source
            frames = list(islice(
                imageio.v3.imiter(BytesIO(img_bytes), **video_kwargs), num_frames))
        else:
            frames = imageio.v3.imread(BytesIO(img_bytes), **video_kwargs)

        if target_size is not None:
            frames = [
                np.asarray(resize_and_crop(
                    Image.fromarray(frame), target_size, resize_method)[0])
                for frame in frames
            ]
        return np.stack(frames) if isinstance(frames, list) else frames

    elif target_size is not None:
        # bounded decoding: draft lets the jpeg decoder work at 1/2, 1/4 or 1/8 scale, so a source
        # far larger than the target never materializes at full resolution. It only ever decodes
        # at or above the requested size, which is what resize_and_crop needs to cover the box.
        with Image.open(BytesIO(img_bytes)) as image:
            image.draft('RGB', (target_size[1], target_size[0]))
            arr = np.asarray(image.convert('RGB'))
        pil, _ = resize_and_crop(Image.fromarray(arr), target_size, resize_method)
        return np.array(pil)

    else:
        return imageio.v3.imread(BytesIO(img_bytes), extension=extension, mode='RGB')


def load_images_parallel(filepaths, file_client):
    futures = []
    results = [None] * len(filepaths)
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 4) as pool:
        for idx, abs_path in enumerate(filepaths):
            fut = pool.submit(load_image_or_video, abs_path, file_client)
            futures.append((idx, fut))

        for idx, fut in futures:
            arr = fut.result()
            results[idx] = arr
    return results


def _decode_av_frames(container, **streams):
    """Decode interleaved packets, replacing invalid audio packets with timed silence."""
    for packet in container.demux(**streams):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            if packet.stream.type != 'audio' or packet.pts is None or not packet.duration:
                raise
            codec = packet.stream.codec_context
            frame = av.AudioFrame(
                format=codec.format.name, layout=codec.layout.name,
                samples=round(packet.duration * packet.time_base * codec.sample_rate))
            for plane in frame.planes:
                plane.update(bytes([128 if codec.format.name.startswith('u8') else 0]) * plane.buffer_size)
            frame.sample_rate = codec.sample_rate
            frame.pts = packet.pts
            frame.time_base = packet.time_base
            frames = [frame]
        yield from frames


class _AVFrameFilter:
    """Feed a stream into FFmpeg filters and drain its available output frames."""

    def __init__(self, first, filters):
        self.graph = av.filter.Graph()
        is_audio = isinstance(first, av.AudioFrame)
        if is_audio:
            source = self.graph.add_abuffer(
                sample_rate=first.sample_rate, format=first.format.name,
                layout=first.layout.name, time_base=first.time_base)
        else:
            source = self.graph.add_buffer(template=first)
        self.graph.link_nodes(source, *(self.graph.add(name, args) for name, args in filters),
                              self.graph.add('abuffersink' if is_audio else 'buffersink'))
        self.graph.configure()

    def push(self, frame):
        self.graph.push(frame)
        while True:
            try:
                yield self.graph.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                return


def load_video_audio(filepath_or_bytes, fps, num_frames=None, file_client=None,
                     sample_rate=32000, audio_channels=2, target_size=None):
    """Decode paired media entirely in memory with PyAV and FFmpeg filters.

    Returns uint8 video [T, H, W, 3] and float32 audio [C, S], starting at the
    first video timestamp. num_frames=None loads the full video at fps; otherwise
    video is truncated or last-frame padded. Audio is truncated or silence padded
    to the video duration. Lanczos resizing runs in the source pixel format before
    RGB conversion and a center crop.
    """
    if isinstance(filepath_or_bytes, (str, Path)):
        if file_client is None:
            file_client = FileClient.infer_client(uri=filepath_or_bytes)
        media_bytes = file_client.get(filepath_or_bytes)
    else:
        media_bytes = filepath_or_bytes

    frames, audio_chunks = [], []
    origin = audio_start = audio_end = None
    video_filter = audio_filter = None
    duration = num_frames / fps if num_frames is not None else None
    with av.open(BytesIO(media_bytes)) as container:
        container.streams.video[0].thread_type = 'AUTO'
        streams = dict(video=0)
        if container.streams.audio:
            streams['audio'] = 0
        video_done, audio_done = False, not container.streams.audio

        # Interleaving preserves audio codec priming without decoding the whole track
        # or seeking back into it after video decoding.
        for frame in _decode_av_frames(container, **streams):
            if isinstance(frame, av.VideoFrame):
                if video_done:
                    continue
                if video_filter is None:
                    if frame.pts is None:
                        raise ValueError('Video needs a frame with a presentation timestamp.')
                    origin = frame.pts * frame.time_base
                    filters = [('setpts', 'PTS-STARTPTS'),
                               ('fps', f'{fps}:start_time=0:round=near:eof_action=pass')]
                    if target_size is not None:
                        height, width = target_size
                        scale = max(height / frame.height, width / frame.width)
                        scaled_h, scaled_w = round(frame.height * scale), round(frame.width * scale)
                        filters += [
                            ('scale', f'{scaled_w}:{scaled_h}:flags=lanczos'),
                            ('format', 'rgb24'),
                            ('crop', f'{width}:{height}:{(scaled_w - width) // 2}:{(scaled_h - height) // 2}:exact=1')
                        ]
                    else:
                        filters.append(('format', 'rgb24'))
                    video_filter = _AVFrameFilter(frame, filters)
                for output in video_filter.push(frame):
                    if num_frames is None or len(frames) < num_frames:
                        frames.append(output.to_ndarray(format='rgb24'))
                video_done = num_frames is not None and len(frames) == num_frames
            else:
                if audio_done:
                    continue
                if audio_filter is None:
                    mono = len(frame.layout.channels) == 1
                    layout = 'mono' if mono or audio_channels == 1 else 'stereo'
                    filters = [
                        ('aresample', f'{sample_rate}:async=1'),
                        ('aformat', f'sample_fmts=fltp:channel_layouts={layout}')
                    ]
                    audio_filter = _AVFrameFilter(frame, filters)
                for output in audio_filter.push(frame):
                    if audio_start is None:
                        audio_start = output.pts * output.time_base
                    audio_chunks.append(output.to_ndarray())
                    audio_end = output.pts * output.time_base + output.samples / sample_rate

            if duration is not None and origin is not None:
                audio_done |= audio_end is not None and audio_end >= origin + duration
                if video_done and audio_done:
                    break

        if video_filter is not None and not video_done:
            for output in video_filter.push(None):
                if num_frames is None or len(frames) < num_frames:
                    frames.append(output.to_ndarray(format='rgb24'))
        if audio_filter is not None and not audio_done:
            for output in audio_filter.push(None):
                if audio_start is None:
                    audio_start = output.pts * output.time_base
                audio_chunks.append(output.to_ndarray())
        if not frames:
            raise ValueError('Video produced no frames at the requested FPS.')
        if num_frames is not None:
            frames.extend([frames[-1]] * (num_frames - len(frames)))

    video = np.stack(frames)
    audio = np.zeros((audio_channels, round(len(video) / fps * sample_rate)), dtype=np.float32)
    if audio_chunks:
        waveform = np.concatenate(audio_chunks, axis=1)
        start = round((audio_start - origin) * sample_rate)
        dest_start, source_start = max(start, 0), max(-start, 0)
        count = min(audio.shape[1] - dest_start, waveform.shape[1] - source_start)
        if count > 0:
            # Broadcasting duplicates mono without attenuating either channel.
            audio[:, dest_start:dest_start + count] = waveform[:, source_start:source_start + count]
    return video, audio


@retry()
def hf_model_loader(model_class, repo_id, **kwargs):
    model = model_class.from_pretrained(repo_id, **kwargs)
    return model


@retry()
def hf_modular_model_loader(blocks, repo_id, **kwargs):
    model = blocks.init_pipeline(repo_id)
    model.load_components(**kwargs)
    # Diffusers logs component-loading failures without raising; propagate them for retry.
    if model.null_component_names:
        raise RuntimeError(f'Failed to load components: {model.null_component_names}')
    return model
