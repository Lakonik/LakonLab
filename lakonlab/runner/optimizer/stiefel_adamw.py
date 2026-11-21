import torch

from torch.optim import Optimizer

from mmcv.runner.optimizer.builder import OPTIMIZERS


def _tangent_project(basis, x):
    in_subspace = basis.T @ x
    return x - basis @ (0.5 * (in_subspace + in_subspace.T))


def _qr_retract(x):
    q, r = torch.linalg.qr(x, mode='reduced')
    # deterministic sign convention on diag(R) to avoid arbitrary column sign flips
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs


class _StiefelAdam(Optimizer):
    """Projection-transport Stiefel Adam: ambient-Euclidean Riemannian gradient,
    scalar second moment, QR retraction, projection-based vector transport.

    The scalar second moment is the mean of Adam's entrywise second moments, so the
    per-entry step scale (and hence the lr semantics) roughly matches ordinary
    elementwise Adam; the lr only rescales the shared tangent direction instead of
    preconditioning entries individually.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group['weight_decay'] != 0.0:
                raise ValueError(
                    'StiefelAdam requires weight_decay=0: L2 decay is normal to the manifold '
                    '(removed by tangent projection) and decoupled decay is a scaling undone '
                    'by retraction.')
            for p in group['params']:
                if p.ndim != 2 or p.size(0) < p.size(1):
                    raise ValueError(
                        'Stiefel parameters must be tall 2D matrices with orthonormal columns, '
                        f'got shape {tuple(p.shape)}.')

    @torch.no_grad()
    def project_loaded_momentum(self, tol=1e-5):
        # re-establish the on-manifold invariants after loading: fp16/bf16 checkpoint
        # rounding leaves the basis slightly off the manifold (retract it before the
        # first forward) and the momentum slightly off the tangent space. Only repair
        # when a violation exceeds tol, so loading a clean fp32 checkpoint stays
        # bitwise-exact (bit-exact resume)
        for group in self.param_groups:
            for p in group['params']:
                eye = torch.eye(p.size(1), device=p.device, dtype=p.dtype)
                if (p.T @ p - eye).abs().max() > tol:
                    p.copy_(_qr_retract(p))
                state = self.state.get(p, None)
                if state is not None and 'exp_avg' in state:
                    exp_avg = state['exp_avg']
                    in_subspace = p.T @ exp_avg
                    in_subspace = 0.5 * (in_subspace + in_subspace.T)
                    if in_subspace.abs().max() > tol * exp_avg.abs().max():
                        exp_avg.copy_(exp_avg - p @ in_subspace.to(exp_avg.dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros((), dtype=p.dtype, device=p.device)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                grad = _tangent_project(p, p.grad)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.square().mean(), alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq / bias_correction2).sqrt() + group['eps']
                p.addcdiv_(exp_avg, denom, value=-group['lr'] / bias_correction1)

                # the update is tangent by construction, for which QR is a valid
                # (cheaper-than-polar) retraction
                p.copy_(_qr_retract(p))

                # transport stored momentum to the new tangent space
                exp_avg.copy_(_tangent_project(p, exp_avg))
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.project_loaded_momentum()


@OPTIMIZERS.register_module()
class StiefelAdamW(Optimizer):
    """Composite optimizer using projection-transport Stiefel Adam for parameters
    flagged with ``lakonlab_stiefel`` and AdamW otherwise.

    Not a canonical-metric Riemannian Adam: it uses the ambient Euclidean metric,
    a scalar per-parameter second moment, QR retraction, and projection as
    approximate vector transport.
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            weight_decay=0.0,
            stiefel_kwargs=None,
            adamw_kwargs=None):
        if stiefel_kwargs is None:
            stiefel_kwargs = dict()
        if adamw_kwargs is None:
            adamw_kwargs = dict()

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            stiefel_kwargs=stiefel_kwargs,
            adamw_kwargs=adamw_kwargs,
        )

        stiefel_param_groups, fallback_param_groups = self._split_param_groups(params)
        if not stiefel_param_groups and not fallback_param_groups:
            raise ValueError('optimizer got an empty parameter list')

        self._num_stiefel_param_groups = len(stiefel_param_groups)

        self._stiefel = None
        if stiefel_param_groups:
            self._stiefel = _StiefelAdam(
                stiefel_param_groups,
                lr=lr,
                **stiefel_kwargs)

        self._adamw = None
        if fallback_param_groups:
            self._adamw = torch.optim.AdamW(
                fallback_param_groups,
                lr=lr,
                weight_decay=weight_decay,
                **adamw_kwargs)

        super().__init__(self._child_param_groups(), defaults)
        self._share_state()

    @staticmethod
    def _as_param_list(params):
        if isinstance(params, torch.Tensor):
            return [params]
        return list(params)

    @classmethod
    def _split_param_groups(cls, params):
        stiefel_param_groups = []
        fallback_param_groups = []

        param_groups = cls._as_param_list(params)
        if param_groups and not isinstance(param_groups[0], dict):
            param_groups = [dict(params=param_groups)]

        for param_group in param_groups:
            if isinstance(param_group, dict):
                param_group = param_group.copy()
                group_params = cls._as_param_list(param_group.pop('params'))
            else:
                group_params = [param_group]
                param_group = {}

            stiefel_params = [p for p in group_params if getattr(p, 'lakonlab_stiefel', False)]
            fallback_params = [p for p in group_params if not getattr(p, 'lakonlab_stiefel', False)]

            if stiefel_params:
                stiefel_param_groups.append(dict(param_group, params=stiefel_params))
            if fallback_params:
                fallback_param_groups.append(dict(param_group, params=fallback_params))

        return stiefel_param_groups, fallback_param_groups

    def _child_param_groups(self):
        param_groups = []
        if self._stiefel is not None:
            param_groups.extend(self._stiefel.param_groups)
        if self._adamw is not None:
            param_groups.extend(self._adamw.param_groups)
        return param_groups

    def _share_state(self):
        if self._stiefel is not None:
            self._stiefel.state = self.state
        if self._adamw is not None:
            self._adamw.state = self.state

    def _sync_child_param_groups(self):
        if self._stiefel is not None:
            self._stiefel.param_groups = self.param_groups[:self._num_stiefel_param_groups]
        if self._adamw is not None:
            self._adamw.param_groups = self.param_groups[self._num_stiefel_param_groups:]

    def zero_grad(self, set_to_none=False):
        if self._stiefel is not None:
            self._stiefel.zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if self._stiefel is not None:
            loss = self._stiefel.step(closure=closure)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def state_dict(self):
        return Optimizer.state_dict(self)

    def load_state_dict(self, state_dict):
        Optimizer.load_state_dict(self, state_dict)
        self._sync_child_param_groups()
        self._share_state()
        if self._stiefel is not None:
            self._stiefel.project_loaded_momentum()
