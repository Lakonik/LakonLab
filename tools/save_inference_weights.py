import os
import argparse
from io import BytesIO
from collections import OrderedDict

import torch
from mmcv.fileio import FileClient
from lakonlab.runner.checkpoint import write_checkpoint_to_file


def parse_args():
    parser = argparse.ArgumentParser(description='Save the inference weights of a checkpoint')
    parser.add_argument('path', help='path to the checkpoint')
    parser.add_argument('--out-path', help='path to save the inference weights')
    parser.add_argument('--dtype', default='float16', help='dtype of inference weights')
    parser.add_argument('--ema-only', action='store_true', help='save only the EMA weights')
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path
    if args.out_path is not None:
        out_path = args.out_path
    else:
        root, ext = os.path.splitext(path)
        out_path = root + '_inference' + ext
    dtype = args.dtype
    in_file_client = FileClient.infer_client(uri=path)
    checkpoint = torch.load(BytesIO(in_file_client.get(path)), map_location='cpu')
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if args.ema_only:
        ema_keys = [key for key in checkpoint['state_dict'].keys() if '_ema' in key]
        exclude_keys = [key.replace('_ema', '') for key in ema_keys]
    else:
        exclude_keys = []
    out_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key not in exclude_keys:
            out_dict[key] = value.to(getattr(torch, dtype))
    checkpoint['state_dict'] = out_dict
    write_checkpoint_to_file(checkpoint, out_path)


if __name__ == '__main__':
    main()
