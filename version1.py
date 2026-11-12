"""
Version 1: one head of causal self-attention.

What is NEW compared to version0.py
-----------------------------------
The bigram model predicts the next character from the current character
only. To use more context, every position has to gather information from the
positions before it. Self-attention does this with a learned, data-dependent
weighted average:

  * every position t emits a query q_t ("what am I looking for?"), a key k_t
    ("what do I contain?") and a value v_t ("what I pass on if chosen");
  * the affinity of t for an earlier position s is q_t . k_s / sqrt(H);
  * a causal mask sets the affinities for future positions (s > t) to -inf,
    so after the softmax they get weight 0 and t cannot cheat by looking at
    the character it is supposed to predict;
  * the output at t is sum_s softmax(affinities)_s v_s.

Dividing by sqrt(H) keeps the scores at unit variance at initialisation, so
the softmax does not start out saturated (one-hot).

Attention on its own is order-blind (it is a weighted average over a set), so
we also add a learned position embedding to every token embedding.

The model is: token embedding + position embedding -> one attention head ->
linear layer to the vocabulary. It already beats the bigram model.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (CharTokenizer, Dataset, add_training_args, count_parameters,
                   finish_args, load_checkpoint, read_text, set_seed, train)


class Head(nn.Module):
    """One head of causal self-attention."""
    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular matrix: position t may only look at positions <= t
        self.register_buffer('tril',
                             torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # each row sums to 1 over the past
        return wei @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)


class SingleHeadLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.sa_head(x)
        return self.lm_head(x)  # (B, T, V)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # the position table has block_size rows: crop the context
            logits = self(idx[:, -self.block_size:])[:, -1, :] / temperature
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def main(args):
    set_seed(args.seed)
    text = read_text(args.data_path)
    tok = CharTokenizer.from_text(text)
    data = Dataset(tok.encode(text))

    model = SingleHeadLanguageModel(
        tok.vocab_size, args.block_size, args.n_embd).to(args.device)
    print(f'Parameters: {count_parameters(model):,}')

    if args.train or not os.path.exists(args.ckpt):
        train(model, data, args, args.ckpt)
    else:
        print(f'Loading {args.ckpt} (pass --train to retrain)')
        model.load_state_dict(load_checkpoint(args.ckpt, args.device)['model'])

    model.eval()
    start = torch.tensor([tok.encode('\n')], device=args.device)
    out = model.generate(start, args.max_new_tokens, args.temperature)
    print('-' * 80)
    print(tok.decode(out[0].tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single-head self-attention')
    add_training_args(parser, batch_size=32, block_size=32, max_iters=5000,
                      learning_rate=1e-3, eval_interval=1000, eval_iters=200)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--ckpt', default='checkpoints/single_head.pt')
    main(finish_args(parser.parse_args()))
