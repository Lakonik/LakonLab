import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Count the number of parameters of a checkpoint')
    parser.add_argument('path', help='path to the checkpoint')
    parser.add_argument('--ema-only', action='store_true', help='count only the EMA weights')
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path
    a = torch.load(path, map_location='cpu')
    if args.ema_only:
        ema_keys = [key for key in a['state_dict'].keys() if '_ema' in key]
        exclude_keys = [key.replace('_ema', '') for key in ema_keys]
    else:
        exclude_keys = []
    count = 0
    for key, value in a['state_dict'].items():
        if key not in exclude_keys:
            count += value.numel()
    print(f'Total number of parameters: {count}')
    

if __name__ == '__main__':
    main()
