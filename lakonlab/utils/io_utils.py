# Copyright (c) 2026 Hansheng Chen

import os
import time
import mimetypes
import tempfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO
from functools import wraps
from typing import Generator, Union, Optional, Tuple

import numpy as np
import imageio
import boto3
from PIL import Image
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import torch.distributed as dist
from torch.hub import download_url_to_file
from huggingface_hub import hf_hub_download

import mmcv
from mmcv.fileio import BaseStorageBackend, FileClient


S3_MULTIPART_THRESHOLD = 5 * 2**30  # 5GB
S3_MULTIPART_CHUNKSIZE = 5 * 2**30  # 5GB

TMP_DIR = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
S3_TRANSFER_CONFIG = TransferConfig(
    preferred_transfer_client='classic',  # avoiding crt, which may be incompatible with `fork` in dataloader workers
    multipart_threshold=S3_MULTIPART_THRESHOLD,
    multipart_chunksize=S3_MULTIPART_CHUNKSIZE,
)

LAKONLAB_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'lakonlab')
AWS_SHARED_CREDENTIALS_FILE = os.path.join(os.path.expanduser('~'), '.aws', 'credentials')


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
        self._refresh_client_if_credentials_file_changed()
        return func(self, *args, **kwargs)
    return wrapper


@retry()
def _download_from_url(url, dest_path, hash_prefix):
    download_url_to_file(url, dest_path, hash_prefix, progress=True)


def download_from_url(url,
                      dest_path=None,
                      dest_dir=LAKONLAB_CACHE_DIR,
                      hash_prefix=None):
    """Modified from MMGeneration.
    """
    # get the exact destination path
    if dest_path is None:
        filename = url.split('/')[-1]
        dest_path = os.path.join(dest_dir, filename)

    if dest_path.startswith('~'):
        dest_path = os.path.expanduser('~') + dest_path[1:]

    # advoid downloading existed file
    if os.path.exists(dest_path):
        return dest_path

    is_dist = dist.is_available() and dist.is_initialized()

    if is_dist:
        local_rank = dist.get_node_local_rank()
    else:
        local_rank = 0

    # only download from the master process
    if local_rank == 0:
        # mkdir
        _dir = os.path.dirname(dest_path)
        mmcv.mkdir_or_exist(_dir)
        _download_from_url(url, dest_path, hash_prefix)

    # sync the other processes
    if is_dist:
        dist.barrier()

    return dest_path


@retry()
def download_from_huggingface(filename):
    filename = filename.replace('huggingface://', '').split('/')
    repo_id = '/'.join(filename[:2])
    repo_filename = '/'.join(filename[2:])
    cached_file = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    return cached_file


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

    def _refresh_client_if_credentials_file_changed(self) -> None:
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

    @retry()
    @_refresh_s3_client
    def get(self, filepath: Union[str, Path]) -> bytes:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        bytesio = BytesIO()
        self._client.download_fileobj(bucket, prefix, bytesio, Config=S3_TRANSFER_CONFIG)
        bytesio.seek(0)
        return bytesio.read()

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

    @staticmethod
    def join_path(filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        return os.path.join(str(filepath), *(str(p) for p in filepaths))

    @contextmanager
    def get_local_path(
            self,
            filepath: Union[str, Path],
            **kwargs) -> Generator[Union[str, Path], None, None]:
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False, **kwargs)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

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
        try:
            yield download_from_huggingface(filepath)
        finally:
            pass


FileClient.register_backend(name='s3', backend=S3Backend, force=True, prefixes='s3')
FileClient.register_backend(name='huggingface', backend=HuggingFaceBackend, prefixes='huggingface')


def save_image(image, filepath, file_client):
    img_byte_arr = BytesIO()
    Image.fromarray(image).save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    file_client.put(img_byte_arr, filepath)


def save_video(video, filepath, file_client, fps=16, quality=5, bitrate=None, macro_block_size=16):
    imageio.plugins.ffmpeg.get_exe()
    img_byte_arr = BytesIO()
    with imageio.get_writer(
            img_byte_arr, format='mp4', mode='I', fps=fps,
            quality=quality, bitrate=bitrate, macro_block_size=macro_block_size) as writer:
        for frame in video:
            writer.append_data(frame)
    img_byte_arr = img_byte_arr.getvalue()
    file_client.put(img_byte_arr, filepath)


def resize_and_crop(image: Image, target_hw: tuple[int, int]):
    tgt_h, tgt_w = target_hw
    w, h = image.size
    scale = max(tgt_h / h, tgt_w / w)
    new_h, new_w = round(h * scale), round(w * scale)
    if new_h != h or new_w != w:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - tgt_w) // 2, (new_h - tgt_h) // 2
    return image.crop((left, top, left + tgt_w, top + tgt_h)), scale


def load_image(filepath, file_client, target_size=None):
    img_bytes = file_client.get(filepath)
    extension = os.path.splitext(filepath)[-1].lower()
    arr = imageio.v3.imread(BytesIO(img_bytes), extension=extension)  # (H,W,C) or (H,W)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    if target_size is not None:
        assert arr.ndim == 3
        pil = Image.fromarray(arr)
        pil, _ = resize_and_crop(pil, target_size)
        arr = np.asarray(pil)
    return arr


def load_images_parallel(filepaths, file_client):
    futures = []
    results = [None] * len(filepaths)
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 4) as pool:
        for idx, abs_path in enumerate(filepaths):
            fut = pool.submit(load_image, abs_path, file_client)
            futures.append((idx, fut))

        for idx, fut in futures:
            arr = fut.result()
            results[idx] = arr
    return results


@retry()
def hf_model_loader(model_class, repo_id, **kwargs):
    model = model_class.from_pretrained(repo_id, **kwargs)
    return model
