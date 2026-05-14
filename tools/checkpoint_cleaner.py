import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Remove all checkpoints except the latests')
    parser.add_argument('workdir', help='directory of checkpoints')
    parser.add_argument('--save-inf', action='store_true', help='save inference weights')
    parser.add_argument('--dtype', default='float16', help='dtype of inference weights')
    return parser.parse_args()


def save_inference(path, dtype):
    a = torch.load(path, map_location='cpu')
    if 'optimizer' in a:
        del a['optimizer']
        out_dict = OrderedDict()
        for key, value in a['state_dict'].items():
            out_dict[key] = value.to(getattr(torch, dtype))
        a['state_dict'] = out_dict
        torch.save(a, path)
        print('Saved inference weights for {}'.format(path))
    else:
        print('Skipping {}'.format(path))


def main():
    args = parse_args()
    workdir = args.workdir
    operation = 'remove' if not args.save_inf else 'prune'
    print('This will {} all the non-latest checkpoints in {}'.format(operation, Path(workdir).resolve()))
    answer = None
    while answer not in ('y', 'n'):
        answer = input('continue? [y/n]')
        if answer == 'n':
            exit()
    for dirpath, dirnames, filenames in os.walk(workdir):
        pth_filenames = [f for f in filenames if f.endswith('.pth')]
        if 'latest.pth' in pth_filenames:
            latest_path = os.path.join(dirpath, 'latest.pth')
            if os.path.islink(latest_path):
                latest_path_tgt = Path(latest_path).resolve()
                # os.remove(latest_path)
                # os.rename(latest_path_tgt, latest_path)
                pth_filenames.remove(latest_path_tgt.name)
            pth_filenames.remove('latest.pth')
        for f in pth_filenames:
            path = os.path.join(dirpath, f)
            if args.save_inf:
                save_inference(path, args.dtype)
            else:
                os.remove(path)
                print('Removed {}'.format(path))


if __name__ == '__main__':
    main()
