from .dx import DXPolicy
from .gmflow import GMFlowPolicy


POLICY_CLASSES = dict(
    DX=DXPolicy,
    GMFlow=GMFlowPolicy,
)
