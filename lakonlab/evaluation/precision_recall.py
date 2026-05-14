# Copyright (c) 2026 Hansheng Chen

import torch
import torch.distributed as dist


def compute_pr_distances(row_features,
                         col_features,
                         col_batch_size=10000):
    dist_batches = []
    for col_batch in col_features.split(col_batch_size):
        dist_batch = torch.cdist(
            row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        dist_batches.append(dist_batch)
    return torch.cat(dist_batches, dim=1)


def _get_rank_slice(num_items, rank, world_size):
    chunk = num_items // world_size
    remainder = num_items % world_size
    start = rank * chunk + min(rank, remainder)
    end = start + chunk + int(rank < remainder)
    return start, end


def _all_gather_variable_1d(local_tensor):
    size = torch.tensor([local_tensor.numel()], device=local_tensor.device, dtype=torch.long)
    size_list = [torch.empty_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, size)
    sizes = [int(item.item()) for item in size_list]
    max_size = max(sizes)

    if local_tensor.numel() < max_size:
        padded = torch.zeros(max_size, device=local_tensor.device, dtype=local_tensor.dtype)
        if local_tensor.numel() > 0:
            padded[:local_tensor.numel()] = local_tensor
    else:
        padded = local_tensor

    gathered = [torch.empty(max_size, device=local_tensor.device, dtype=local_tensor.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    return torch.cat([tensor[:size] for tensor, size in zip(gathered, sizes)], dim=0)


def compute_pr_score_distributed(manifold, probes, pr_k=3, pr_row_batch_size=10000, pr_col_batch_size=10000):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = manifold.device

    manifold_start, manifold_end = _get_rank_slice(manifold.shape[0], rank, world_size)
    local_manifold = manifold[manifold_start:manifold_end]

    kth = manifold.new_empty((0,), dtype=torch.float32)
    if local_manifold.shape[0] > 0:
        kth_chunks = []
        for manifold_batch in local_manifold.split(pr_row_batch_size):
            distance = compute_pr_distances(
                row_features=manifold_batch,
                col_features=manifold,
                col_batch_size=pr_col_batch_size)
            kth_chunks.append(
                distance.to(torch.float32).kthvalue(pr_k + 1, dim=1).values)
        kth = torch.cat(kth_chunks, dim=0)
    kth = _all_gather_variable_1d(kth)

    probe_start, probe_end = _get_rank_slice(probes.shape[0], rank, world_size)
    local_probes = probes[probe_start:probe_end]
    local_true = torch.zeros(1, device=device, dtype=torch.long)
    local_total = torch.tensor([local_probes.shape[0]], device=device, dtype=torch.long)

    manifold_chunks = manifold.split(pr_col_batch_size)
    kth_chunks = kth.split(pr_col_batch_size)
    for probes_batch in local_probes.split(pr_row_batch_size):
        matched = torch.zeros(probes_batch.shape[0], device=device, dtype=torch.bool)
        for manifold_batch, kth_batch in zip(manifold_chunks, kth_chunks):
            distance = torch.cdist(
                probes_batch.unsqueeze(0), manifold_batch.unsqueeze(0))[0]
            matched |= (distance <= kth_batch.unsqueeze(0)).any(dim=1)
            if matched.all():
                break
        local_true += matched.sum(dtype=torch.long)

    global_counts = torch.cat([local_true, local_total], dim=0)
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    return float(global_counts[0].to(torch.float32) / global_counts[1].clamp_min(1).to(torch.float32))


def compute_pr_score(manifold, probes, pr_k=3, pr_row_batch_size=10000, pr_col_batch_size=10000):
    kth = []
    for manifold_batch in manifold.split(pr_row_batch_size):
        distance = compute_pr_distances(
            row_features=manifold_batch,
            col_features=manifold,
            col_batch_size=pr_col_batch_size)
        kth.append(
            distance.to(torch.float32).kthvalue(pr_k + 1, dim=1).values)
    kth = torch.cat(kth, dim=0)

    pred = []
    manifold_chunks = manifold.split(pr_col_batch_size)
    kth_chunks = kth.split(pr_col_batch_size)
    for probes_batch in probes.split(pr_row_batch_size):
        matched = torch.zeros(probes_batch.shape[0], device=manifold.device, dtype=torch.bool)
        for manifold_batch, kth_batch in zip(manifold_chunks, kth_chunks):
            distance = torch.cdist(
                probes_batch.unsqueeze(0), manifold_batch.unsqueeze(0))[0]
            matched |= (distance <= kth_batch.unsqueeze(0)).any(dim=1)
            if matched.all():
                break
        pred.append(matched)
    return float(torch.cat(pred).to(torch.float32).mean())
