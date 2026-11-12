"""
LoRA: Low-Rank Adaptation (Hu et al., 2021, https://arxiv.org/abs/2106.09685),
written from scratch.

Idea: fine-tuning changes a pretrained weight matrix W (d_out x d_in) by some
dW. LoRA bets that dW has low rank and writes it as

    dW = (alpha / r) * B @ A,     A: (r x d_in),  B: (d_out x r),  r << d_in

W stays frozen; only A and B are trained. For GPT-2's attention projection
c_attn (768 -> 2304) that is r * (768 + 2304) = 3,072 r parameters instead of
1.77M.

Initialisation matters: A is random and B is zero, so dW = 0 at the start
(the model begins exactly as the pretrained one) but the gradient of B,
which is proportional to (A x), is not zero, so training can move away from 0.
(If both were zero, neither would ever receive a gradient; if both were
random, the fine-tuned model would start from a corrupted backbone.)

After training, dW can be folded into W (`merge`), so a LoRA model is exactly
as fast as the original one at inference time.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update."""
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same as nn.Linear
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if not self.merged:
            # x A^T B^T never forms the (d_out x d_in) matrix: it costs
            # r * (d_in + d_out) per token instead of d_in * d_out
            y = y + (self.dropout(x) @ self.A.T @ self.B.T) * self.scale
        return y

    @torch.no_grad()
    def merge(self):
        """W <- W + (alpha / r) B A; afterwards the LoRA branch is skipped."""
        if not self.merged:
            self.base.weight += self.scale * (self.B @ self.A)
            self.merged = True


def apply_lora(model: nn.Module, targets=('c_attn', 'c_proj', 'c_fc'),
               rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
    """
    Replace every nn.Linear whose attribute name is in `targets` by a
    LoRALinear. In GPT-2 (gpt2.py): attn.c_attn (queries, keys, values),
    attn.c_proj (attention output), mlp.c_fc and mlp.c_proj (the MLP).
    (In version2's GPT the same layers are called qkv, proj and fc.)
    """
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name,
                        LoRALinear(child, rank, alpha, dropout))
    return model


def merge_lora(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def lora_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]
