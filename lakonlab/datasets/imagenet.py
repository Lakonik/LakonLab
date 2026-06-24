# Copyright (c) 2026 Hansheng Chen

import os
import pickle
import tarfile
import tempfile

import numpy as np
import torch
import mmcv

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from mmcv.fileio import FileClient
from mmcv.parallel import DataContainer as DC
from lakonlab.utils import get_root_logger, locked_cache_path
from .builder import DATASETS


def image_preproc(pil_image, image_size, random_flip=False):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    if random_flip and np.random.rand() < 0.5:
        arr = np.ascontiguousarray(arr[:, ::-1])

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        else:
            assert arr.shape[2] == 3
    else:
        raise ValueError(f'Unexpected number of dimensions: {arr.ndim}')
    return arr


@DATASETS.register_module()
class ImageNet(Dataset):
    def __init__(
            self,
            data_root='data/imagenet/train',
            datalist_path='data/imagenet/train.txt',
            label2name_path='data/imagenet/imagenet1000_clsidx_to_labels.txt',
            ignore_cached_latents=False,
            random_flip=True,
            negative_label=1000,
            image_size=256,
            return_image_bytes=False,
            tar_extract_dir=None,
            latent_size=(4, 32, 32),
            test_label_repeat=1,
            test_label_sampling='random',
            test_mode=False,
            num_test_images=50000):
        super().__init__()
        self.tar_extract_dir = tar_extract_dir
        self.original_data_root = data_root
        self.data_root = data_root if test_mode else self.prepare_data_root(data_root)
        self._file_client = None
        self._data_root_from_tar = self.data_root != self.original_data_root

        self.datalist_path = datalist_path
        self.label2name_path = label2name_path
        self.ignore_cached_latents = ignore_cached_latents
        self.random_flip = random_flip
        self.negative_label = negative_label
        self.image_size = image_size
        self.return_image_bytes = return_image_bytes
        self.latent_size = latent_size
        self.test_label_repeat = test_label_repeat
        self.test_label_sampling = test_label_sampling
        self.test_mode = test_mode
        self.num_test_images = num_test_images

        self.label2name = {}
        label2name_text = FileClient.infer_client(uri=self.label2name_path).get_text(self.label2name_path)
        for line in label2name_text.split('\n'):
            line = line.strip()
            if len(line) == 0:
                continue
            idx, name = line.split(':')
            idx, name = idx.strip(), name.strip()
            if name[-1] == ',':
                name = name[:-1]
            if name[0] == '"' and name[-1] == '"':
                name = name[1:-1]
            if name[0] == "'" and name[-1] == "'":
                name = name[1:-1]
            self.label2name[int(idx)] = name

        if not test_mode:
            self.all_paths = []
            self.all_labels = []
            datalist_text = FileClient.infer_client(uri=self.datalist_path).get_text(self.datalist_path)
            for line in datalist_text.split('\n'):
                line = line.strip()
                if len(line) == 0:
                    continue
                path_label = line.split(' ')
                self.all_paths.append(path_label[0])
                if len(path_label) > 1:
                    self.all_labels.append(int(path_label[1]))

            logger = get_root_logger()
            if self._data_root_from_tar:
                mmcv.print_log(f'Packed data root: {self.original_data_root}', logger=logger)
            mmcv.print_log(f'Data root: {self.data_root}', logger=logger)
            mmcv.print_log(f'Data list path: {self.datalist_path}', logger=logger)
            mmcv.print_log(f'Number of images: {len(self.all_paths)}', logger=logger)

    def prepare_data_root(self, data_root):
        if not data_root.lower().endswith(('.tar', '.tar.gz', '.tgz')):
            return data_root

        if self.tar_extract_dir is None:
            tar_name = os.path.basename(data_root)
            for suffix in ('.tar.gz', '.tgz', '.tar'):
                if tar_name.endswith(suffix):
                    tar_name = tar_name[:-len(suffix)]
                    break
            extract_dir = os.path.join(tempfile.gettempdir(), tar_name)
        else:
            extract_dir = self.tar_extract_dir

        extract_parent = os.path.dirname(extract_dir)
        assert extract_parent
        os.makedirs(extract_parent, exist_ok=True)
        with locked_cache_path(extract_dir) as (local_path, exists):
            if not exists:
                os.makedirs(local_path, exist_ok=True)
                file_client = FileClient.infer_client(uri=data_root)
                with file_client.get_local_path(data_root) as local_tar_path:
                    with tarfile.open(local_tar_path, mode='r:*') as tar:
                        tar.extractall(local_path)

        return extract_dir

    @property
    def file_client(self):
        if self._file_client is None:
            self._file_client = FileClient.infer_client(uri=self.data_root)
        return self._file_client

    def __len__(self):
        return self.num_test_images if self.test_mode else len(self.all_paths)

    def __getitem__(self, idx):
        data = dict(ids=DC(idx, cpu_only=True))

        if self.test_mode:
            label_sample_idx = idx // self.test_label_repeat
            if self.test_label_sampling == 'equal':
                label = torch.tensor(label_sample_idx % 1000, dtype=torch.long)
            elif self.test_label_sampling == 'random':
                label_generator = torch.Generator().manual_seed(label_sample_idx)
                label = torch.randint(0, 1000, (), generator=label_generator).long()
            else:
                raise ValueError(f'Unsupported test_label_sampling: {self.test_label_sampling}')
            noise_generator = torch.Generator().manual_seed(idx + 1000)
            noise = torch.randn(self.latent_size, generator=noise_generator)
            data.update(noise=noise)

        else:
            rel_data_path = self.all_paths[idx]
            data.update(paths=DC(rel_data_path, cpu_only=True))
            data_path = self.file_client.join_path(self.data_root, rel_data_path)
            data_bytes = self.file_client.get(data_path)
            data_bytesio = BytesIO(data_bytes)
            ext = os.path.splitext(data_path)[-1]
            if ext.lower() in ('.pth', '.pt', '.pkl'):
                if ext.lower() == '.pkl':
                    torch_data = pickle.load(data_bytesio)
                else:
                    torch_data = torch.load(data_bytesio, map_location='cpu')
                label = torch.as_tensor(torch_data['y'], dtype=torch.long)
                if self.ignore_cached_latents:
                    assert 'image_bytes' in torch_data
                    img_data = Image.open(BytesIO(torch_data['image_bytes'])).convert('RGB')
                    data.update(
                        images=torch.from_numpy(image_preproc(
                            img_data, self.image_size, random_flip=self.random_flip
                        )).float().permute(2, 0, 1) / 255.0)
                else:
                    if 'x_std' in torch_data:
                        mean = torch_data['x_mean'].float()
                        std = torch_data['x_std'].float()
                        data.update(latents=mean + std * torch.randn_like(mean))
                    else:
                        data.update(latents=torch_data['x'].float())
                    if 'image_bytes' in torch_data:
                        assert not self.random_flip
                        img_data = Image.open(BytesIO(torch_data['image_bytes'])).convert('RGB')
                        data.update(
                            images=torch.from_numpy(image_preproc(
                                img_data, self.image_size, random_flip=self.random_flip
                            )).float().permute(2, 0, 1) / 255.0)
            elif ext.lower() in ('.jpg', '.jpeg', '.png'):
                label = torch.tensor(self.all_labels[idx], dtype=torch.long)
                if self.return_image_bytes:
                    data.update(image_bytes=DC(data_bytes, cpu_only=True))
                img_data = Image.open(data_bytesio).convert('RGB')
                data.update(
                    images=torch.from_numpy(image_preproc(
                        img_data, self.image_size, random_flip=self.random_flip
                    )).float().permute(2, 0, 1) / 255.0)
            else:
                raise ValueError(f'Unsupported file extension: {ext}')

        name = self.label2name[label.item()]
        data.update(labels=label, name=DC(name, cpu_only=True))

        if self.negative_label is not None:
            if isinstance(self.negative_label, int):
                data.update(negative_labels=torch.tensor(self.negative_label, dtype=torch.long))
            else:
                raise ValueError(f'Unsupported negative label: {self.negative_label}')

        return data
