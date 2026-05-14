import os
import random

import numpy as np
import torch
from mmcv.runner import get_dist_info


_MAX_SEED = 2 ** 32


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    seed = int(seed) % _MAX_SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
