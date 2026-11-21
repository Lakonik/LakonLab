from diffusers.models.attention import FeedForward

from lakonlab.ops import swiglu_linear


class SwiGLUFeedForward(FeedForward):
    """diffusers' SwiGLU FeedForward ([SwiGLU, dropout, out]) that in eval mode without grad takes the SwiGLU
    product inside the out projection's activation quantization (see swiglu_linear), or in place."""

    def forward(self, hidden_states, *args, **kwargs):
        if self.training or len(self.net) != 3:
            return super().forward(hidden_states, *args, **kwargs)
        hidden_states = self.net[0].proj(hidden_states)
        if hidden_states.requires_grad:
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            return self.net[2](hidden_states * self.net[0].activation(gate))
        return swiglu_linear(self.net[2], hidden_states, gate_first=False)
