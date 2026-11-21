import torch
from peft.tuners.lora.layer import Linear as LoraLinear

from .activation import silu_mul_


class InplaceLoraLinear(LoraLinear):
    """peft's LoRA Linear that, with grad disabled, adds the LoRA update into the base output in place.

    peft computes ``result + lora_B(lora_A(x)) * scaling``, which holds two extra output-sized tensors next to the
    base output. Here only the rank-sized ``lora_A`` output is extra. Vanilla LoRA only; anything else (grad, extra
    forward arguments, disabled or merged adapters, LoRA variants, LoRA bias) runs peft's forward.
    """

    def _peft_forward(self):
        return (torch.is_grad_enabled() or self.disable_adapters or self.merged or self.lora_variant
                or any(self.lora_bias.values()))

    def _add_lora_(self, result, x):
        for name in self.active_adapters:
            if name in self.lora_A:
                h = self.lora_A[name](self.lora_dropout[name](x.to(self.lora_A[name].weight.dtype)))
                result.view(-1, result.size(-1)).addmm_(
                    h.view(-1, h.size(-1)), self.lora_B[name].weight.T, alpha=self.scaling[name])
        return result

    def forward(self, x, *args, **kwargs):
        if args or kwargs or self._peft_forward():
            return super().forward(x, *args, **kwargs)
        return self._add_lora_(self.base_layer(x), x)

    def forward_swiglu(self, x, gate_first=True):
        """``self(silu(gate) * up)`` without grad for a contiguous SwiGLU projection x = [gate, up], or [up, gate]
        with ``gate_first=False``. The base layer's forward_swiglu leaves the product over the first half of x,
        which the LoRA reads."""
        hidden = x[..., :x.size(-1) // 2]
        if self._peft_forward() or not hasattr(self.base_layer, 'forward_swiglu'):
            silu_mul_(x, gate_first=gate_first)
            return self(hidden)
        return self._add_lora_(self.base_layer.forward_swiglu(x, gate_first, store_product=True), hidden)

    @classmethod
    def convert(cls, module):
        """Convert every peft LoRA Linear in ``module`` into an InplaceLoraLinear in place. Parameters and
        training behavior are unchanged."""
        for m in module.modules():
            if type(m) is LoraLinear:
                m.__class__ = cls
        return module
