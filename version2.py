"""
Version 2: the full decoder-only transformer (GPT), character level.

What is NEW compared to version1.py
-----------------------------------
Four ideas turn the single attention head into a GPT. Each fixes a concrete
weakness of version1:

1. Multi-head attention. One head computes one weighted average, i.e. it can
   follow one kind of relation at a time. n_head heads of size C / n_head run
   in parallel (one may track "the previous vowel", another "the speaker's
   name") and their outputs are concatenated and mixed by a linear layer.
   Instead of a Python list of `Head`s we compute all heads with ONE linear
   layer for the queries, keys and values and fold the heads into a batch
   dimension, (B, n_head, T, head_size): the maths is the same, it is faster.

2. A feed-forward network (MLP) after attention. Attention is communication
   between positions; the MLP is computation, applied to each position on its
   own (C -> 4C -> C with a GELU).

3. Residual connections. Each sub-layer ADDS to a running "residual stream",
   x = x + attn(x) and x = x + mlp(x), so the gradient has a direct path from
   the loss to the first layers and deep stacks become trainable.

4. Layer normalisation (pre-norm, as in GPT-2): each sub-layer sees a
   normalised copy of the stream, and a final LayerNorm sits before the output
   layer. Dropout regularises the attention weights and every sub-layer.

Stacking n_layer such blocks gives the decoder-only transformer of GPT-2
(Radford et al., 2019) at a small scale. With the default size (6 layers,
384 dimensions, 6 heads, 256 characters of context, 10.8M parameters) the
validation loss drops from ~2.5 (bigram) to ~1.5 nats per character and the
samples look like Shakespeare.

`GPT` here is the model that every later version reuses: version3.py trains it
on BPE tokens, version4.py adds a KV cache to it, and version5.py fine-tunes
it with LoRA.
"""

import argparse
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (CharTokenizer, Dataset, add_training_args, count_parameters,
                   finish_args, load_checkpoint, read_text, set_seed, train)


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256  # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class MultiHeadAttention(nn.Module):
    """All heads of causal self-attention, computed in one batched matmul."""
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        # one projection for the queries, keys and values of every head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer('tril', torch.tril(
            torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)))

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, C = t.shape  # (B, T, C) -> (B, nh, T, hs)
        return t.view(B, T, self.n_head, self.head_size).transpose(1, 2)

    def merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = t.shape  # (B, nh, T, hs) -> (B, T, C)
        return t.transpose(1, 2).reshape(B, T, self.n_head * self.head_size)

    def attend(self, q, k, v, mask):
        # exactly version1's Head, with an extra head dimension
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B,nh,Tq,Tk)
        wei = wei.masked_fill(~mask, float('-inf'))
        wei = self.attn_dropout(F.softmax(wei, dim=-1))
        return wei @ v  # (B, nh, Tq, hs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        q, k, v = map(self.split_heads, self.qkv(x).chunk(3, dim=-1))
        out = self.attend(q, k, v, self.tril[:T, :T])
        return self.resid_dropout(self.proj(self.merge_heads(out)))


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> C."""
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformer block: communication (attention), then computation (MLP)."""
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # the residual stream is only added to
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # GPT-2 initialisation: small normal weights, zero biases
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """The output of the last block (after ln_f), shape (B, T, C)."""
        B, T = idx.shape
        assert T <= self.block_size, f'sequence of {T} > block_size'
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden_states(idx))  # (B, T, V)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, greedy=False):
        for _ in range(max_new_tokens):
            # the position table has block_size rows: crop the context
            logits = self(idx[:, -self.block_size:])[:, -1, :]
            if greedy:
                idx_next = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def build_from_checkpoint(path, device, model_cls=GPT):
    """Rebuild a model (and its tokenizer state) saved by train()."""
    ckpt = load_checkpoint(path, device)
    model = model_cls(GPTConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model'])
    return model, ckpt


def add_model_args(parser, **defaults):
    d = dict(n_layer=6, n_head=6, n_embd=384, dropout=0.2)
    d.update(defaults)
    parser.add_argument('--n_layer', type=int, default=d['n_layer'])
    parser.add_argument('--n_head', type=int, default=d['n_head'])
    parser.add_argument('--n_embd', type=int, default=d['n_embd'])
    parser.add_argument('--dropout', type=float, default=d['dropout'])
    return parser


def config_from_args(args, vocab_size) -> GPTConfig:
    return GPTConfig(vocab_size=vocab_size, block_size=args.block_size,
                     n_layer=args.n_layer, n_head=args.n_head,
                     n_embd=args.n_embd, dropout=args.dropout)


def main(args):
    set_seed(args.seed)
    text = read_text(args.data_path)
    tok = CharTokenizer.from_text(text)
    data = Dataset(tok.encode(text))

    cfg = config_from_args(args, tok.vocab_size)
    model = GPT(cfg).to(args.device)
    print(f'Parameters: {count_parameters(model) / 1e6:.2f}M')

    if args.train or not os.path.exists(args.ckpt):
        train(model, data, args, args.ckpt,
              extra_state={'config': asdict(cfg),
                           'tokenizer': tok.state_dict()})
    else:
        print(f'Loading {args.ckpt} (pass --train to retrain)')
        model.load_state_dict(load_checkpoint(args.ckpt, args.device)['model'])

    model.eval()
    prompt = torch.tensor([tok.encode(args.prompt)], device=args.device)
    out = model.generate(prompt, args.max_new_tokens, args.temperature)
    print('-' * 80)
    print(tok.decode(out[0].tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Character-level GPT')
    add_training_args(parser)
    add_model_args(parser)
    parser.add_argument('--prompt', default='\n')
    parser.add_argument('--ckpt', default='checkpoints/gpt_char.pt')
    main(finish_args(parser.parse_args()))
