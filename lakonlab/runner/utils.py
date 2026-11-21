import os
import contextlib
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


@contextlib.contextmanager
def preserve_rng_state():
    """Snapshot and restore every global RNG stream (python, numpy, torch CPU, all
    CUDA devices). Wrap any code that is not part of the training trajectory but may
    consume or reseed the global streams -- bucket warmup steps, evaluation, metric
    libraries. A module-level `torch.manual_seed` in an external benchmark once
    re-synchronized the per-rank training streams that `--diff_seed` decorrelates,
    collapsing the noise/timestep diversity of every batch after the first eval.
    """
    python_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_states)
