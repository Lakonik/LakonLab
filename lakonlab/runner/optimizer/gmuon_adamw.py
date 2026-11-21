from collections import defaultdict

import torch

from torch.optim import Optimizer

from mmcv.runner.optimizer.builder import OPTIMIZERS


@OPTIMIZERS.register_module()
class GMuonAdamW(Optimizer):
    """Composite optimizer using Gram-Muon for matrices and AdamW otherwise.

    GMuon is applied to 2D parameters, while AdamW handles biases, norms, and
    other non-matrix parameters, as well as any parameter flagged with
    ``lakonlab_no_muon`` -- embeddings and output heads, which Muon is not meant
    for.

    A parameter flagged with ``lakonlab_muon_chunks`` is a fused matrix, e.g. a
    qkv projection or a SwiGLU gate/up pair, whose blocks are functionally
    separate. Those go to their own param group with GMuon's split/recombine
    hooks, so each block is orthogonalized on its own instead of being coupled
    through a joint Gram matrix, and each is scaled by its own shape.
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            weight_decay=0.0,
            muon_kwargs=None,
            adamw_kwargs=None):
        if muon_kwargs is None:
            muon_kwargs = dict()
        if adamw_kwargs is None:
            adamw_kwargs = dict()

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            muon_kwargs=muon_kwargs,
            adamw_kwargs=adamw_kwargs,
        )

        muon_param_groups, fallback_param_groups = self._split_param_groups(params)
        if not muon_param_groups and not fallback_param_groups:
            raise ValueError('optimizer got an empty parameter list')

        self._num_muon_param_groups = len(muon_param_groups)

        self._muon = None
        if muon_param_groups:
            from gram_newton_schulz import Muon as GMuon

            self._muon = GMuon(
                muon_param_groups,
                lr=lr,
                weight_decay=weight_decay,
                **muon_kwargs)

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

    @staticmethod
    def _chunk_fns(num_chunks):
        """GMuon's hooks to orthogonalize a fused matrix blockwise, applied to the gradient."""
        return dict(
            param_split_fn=lambda matrix: list(matrix.chunk(num_chunks, dim=0)),
            param_recombine_fn=lambda matrices: torch.cat(list(matrices), dim=0))

    @classmethod
    def _split_param_groups(cls, params):
        muon_param_groups = []
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

            muon_params = [p for p in group_params if p.ndim == 2 and not getattr(p, 'lakonlab_no_muon', False)]
            fallback_params = [p for p in group_params if p.ndim != 2 or getattr(p, 'lakonlab_no_muon', False)]

            # one group per chunk count, since the split hooks are per group
            muon_params_by_chunks = defaultdict(list)
            for p in muon_params:
                muon_params_by_chunks[int(getattr(p, 'lakonlab_muon_chunks', 1))].append(p)

            for num_chunks, chunk_params in sorted(muon_params_by_chunks.items()):
                chunk_group = dict(
                    param_group, params=chunk_params, lakonlab_muon_chunks=num_chunks)
                if num_chunks > 1:
                    for p in chunk_params:
                        if p.shape[0] % num_chunks != 0:
                            raise ValueError(
                                f'A parameter of shape {tuple(p.shape)} is flagged as '
                                f'{num_chunks} fused blocks, which does not divide it.')
                    chunk_group.update(cls._chunk_fns(num_chunks))
                muon_param_groups.append(chunk_group)
            if fallback_params:
                fallback_param_groups.append(dict(param_group, params=fallback_params))

        return muon_param_groups, fallback_param_groups

    def _child_param_groups(self):
        param_groups = []
        if self._muon is not None:
            param_groups.extend(self._muon.param_groups)
        if self._adamw is not None:
            param_groups.extend(self._adamw.param_groups)
        return param_groups

    def _share_state(self):
        if self._muon is not None:
            self._muon.state = self.state
        if self._adamw is not None:
            self._adamw.state = self.state

    def _sync_child_param_groups(self):
        if self._muon is not None:
            muon_param_groups = self.param_groups[:self._num_muon_param_groups]
            self._muon.param_groups = muon_param_groups
            if hasattr(self._muon, '_combined_param_groups'):
                self._muon._combined_param_groups = muon_param_groups
        if self._adamw is not None:
            self._adamw.param_groups = self.param_groups[self._num_muon_param_groups:]

    def zero_grad(self, set_to_none=False):
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if self._muon is not None:
            loss = self._muon.step(closure=closure)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def state_dict(self):
        state_dict = Optimizer.state_dict(self)
        for param_group in state_dict['param_groups']:
            param_group.pop('param_split_fn', None)
            param_group.pop('param_recombine_fn', None)
        return state_dict

    def _restore_chunk_fns(self):
        for param_group in self.param_groups[:self._num_muon_param_groups]:
            param_group.pop('param_split_fn', None)
            param_group.pop('param_recombine_fn', None)
            num_chunks = param_group.get('lakonlab_muon_chunks', 1)
            if num_chunks > 1:
                param_group.update(self._chunk_fns(num_chunks))

    def load_state_dict(self, state_dict):
        Optimizer.load_state_dict(self, state_dict)
        self._restore_chunk_fns()
        self._sync_child_param_groups()
        self._share_state()
