# Copyright (c) 2026 Hansheng Chen

import torch


def _per_sample_grad_norm(grads):
    """Joint per-sample L2 norm over (possibly None) batch-first gradients."""
    return sum(
        g.float().flatten(1).square().sum(dim=1) for g in grads if g is not None
    ).sqrt()


class _AdaptiveGradNormSplit(torch.autograd.Function):
    """Identity forward; merge main/auxiliary gradients with adaptive scaling.

    Outputs are block-ordered: (main_0, ..., main_n, aux_0, ..., aux_n), each a
    distinct view so the two branches occupy separate autograd edges.
    """

    @staticmethod
    def forward(ctx, ratio, min_scale, max_scale, *tensors):
        ctx.ratio = ratio
        ctx.min_scale = min_scale
        ctx.max_scale = max_scale
        ctx.set_materialize_grads(False)
        return (*(t.view_as(t) for t in tensors),
                *(t.view_as(t) for t in tensors))

    @staticmethod
    def backward(ctx, *grads):
        num_tensors = len(grads) // 2
        main_grads = grads[:num_tensors]
        aux_grads = grads[num_tensors:]

        if all(g is None for g in aux_grads):  # aux branch unused (e.g. warmup)
            return None, None, None, *main_grads
        if all(g is None for g in main_grads):  # no reference; zero the aux gradient
            return None, None, None, *(
                None if g is None else torch.zeros_like(g) for g in aux_grads)

        scale = (
            ctx.ratio * _per_sample_grad_norm(main_grads)
            / _per_sample_grad_norm(aux_grads)
        ).clamp(min=ctx.min_scale, max=ctx.max_scale).nan_to_num(nan=ctx.min_scale)

        input_grads = []
        for main_grad, aux_grad in zip(main_grads, aux_grads):
            if aux_grad is not None:
                aux_grad = aux_grad * scale.reshape(
                    -1, *((aux_grad.dim() - 1) * [1])).to(aux_grad.dtype)
                main_grad = (
                    aux_grad if main_grad is None
                    else main_grad + aux_grad)
            input_grads.append(main_grad)
        return None, None, None, *input_grads


def adaptive_grad_norm_split(tensors, ratio, min_scale=1e-4, max_scale=1e4):
    """Return value-identical main and auxiliary views.

    During backward, preserve the primary gradient and scale the auxiliary gradient toward
    ratio × ||primary||, with its multiplier clamped to [min_scale, max_scale].
    """
    if isinstance(tensors, torch.Tensor):
        return _AdaptiveGradNormSplit.apply(float(ratio), float(min_scale), float(max_scale), tensors)
    if not isinstance(tensors, dict):
        raise TypeError(
            'Adaptive split expects a Tensor or a flat dict of Tensors.')
    grad_keys = [
        key for key, value in tensors.items()
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    if not grad_keys:
        return tensors, tensors

    outputs = _AdaptiveGradNormSplit.apply(
        float(ratio), float(min_scale), float(max_scale), *(tensors[key] for key in grad_keys))
    main_tensors = dict(tensors)
    aux_tensors = dict(tensors)
    for index, key in enumerate(grad_keys):
        main_tensors[key] = outputs[index]
        aux_tensors[key] = outputs[len(grad_keys) + index]
    return main_tensors, aux_tensors
