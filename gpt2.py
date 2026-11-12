"""
GPT-2 written from scratch, so that OpenAI's pretrained weights can be loaded
into it, plus a sequence-classification head.

This is version2's GPT with the details GPT-2 fixes (all are one-line
changes):
  * the query/key/value projection has a bias,
  * GELU uses the tanh approximation,
  * the output layer has no bias and shares its weight with the token
    embedding ("weight tying"),
  * LayerNorm everywhere as in version2 (pre-norm), context 1024, vocabulary
    of 50,257 byte-level BPE tokens,
  * attention uses PyTorch's fused scaled_dot_product_attention, which
    computes exactly version2's softmax(q k^T / sqrt(hs)) v with a causal mask.
The attribute names (transformer.h.0.attn.c_attn, ...) match OpenAI's
checkpoint, which is what lets `from_pretrained` copy the weights.

Adapted from Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT,
MIT licence), which follows OpenAI's GPT-2 code
(https://github.com/openai/gpt-2/blob/master/src/model.py) and Hugging Face's
modeling_gpt2.py.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q, k, v
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, nh, T, hs), as in version2
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate='tanh')))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def hidden_states(self, idx):
        """Output of the last block after ln_f, shape (B, T, n_embd)."""
        B, T = idx.size()
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx)
                                  + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)

    def forward(self, idx):
        return self.lm_head(self.hidden_states(idx))  # (B, T, vocab)

    @classmethod
    def from_pretrained(cls, model_type='gpt2', dropout=0.0):
        """Build our GPT2 and copy OpenAI's weights into it."""
        from transformers import GPT2LMHeadModel
        sizes = {'gpt2': dict(n_layer=12, n_head=12, n_embd=768),          # 124M
                 'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
                 'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
                 'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)}      # 1558M
        model = cls(GPT2Config(**sizes[model_type], dropout=dropout))
        sd = model.state_dict()
        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
        keys_hf = [k for k in sd_hf if not k.endswith(('.attn.bias',
                                                        '.attn.masked_bias'))]
        assert len(keys_hf) == len(sd), (len(keys_hf), len(sd))
        # OpenAI stored these layers as "Conv1D" with weight (in, out);
        # nn.Linear stores (out, in), so they are transposed on the way in
        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight')
        with torch.no_grad():
            for k in keys_hf:
                w = sd_hf[k].t() if k.endswith(transposed) else sd_hf[k]
                assert w.shape == sd[k].shape, k
                sd[k].copy_(w)
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.block_size:])[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class GPT2Classifier(nn.Module):
    """
    GPT-2 as a sentence classifier: pool the hidden states of the real
    (non-padding) tokens and apply a linear layer.
      pool='last': the last real token, the only one that has attended to the
                   whole sentence (Hugging Face's GPT2ForSequenceClassification);
      pool='mean': the average over all real tokens.
    """
    def __init__(self, gpt: GPT2, n_classes: int, pool: str = 'last'):
        super().__init__()
        self.gpt, self.pool = gpt, pool
        self.head = nn.Linear(gpt.cfg.n_embd, n_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx, mask):
        h = self.gpt.hidden_states(idx)  # (B, T, C); padding is on the right
        if self.pool == 'mean':
            m = mask.unsqueeze(-1).to(h.dtype)
            h = (h * m).sum(1) / m.sum(1)
        else:
            h = h[torch.arange(len(h), device=h.device), mask.sum(1) - 1]
        return self.head(h)


def get_tokenizer(model_type='gpt2'):
    """GPT-2's byte-level BPE tokenizer (version3's idea at 50k merges)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_type)
