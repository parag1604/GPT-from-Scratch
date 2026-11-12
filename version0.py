"""
Version 0: a character-level GPT trained on tinyshakespeare.

What this file teaches
----------------------
* Tokenization at its simplest: every distinct character is a token.
* Language modelling as next-token prediction: the target is the input
  shifted by one position.
* The decoder-only transformer, built from scratch:
    - token embeddings + learned position embeddings,
    - causal (masked) self-attention, one head at a time,
    - multi-head attention = several heads run side by side and concatenated,
    - a position-wise feed-forward network,
    - pre-LayerNorm residual blocks.
* Autoregressive sampling: predict a distribution over the next character,
  sample from it, append the sample to the input, repeat.

This is the starting point; the next versions only change one thing at a time.
"""

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Dataset, count_parameters, get_dataset, get_device, \
    set_seed, train


class Mapper:
    """Character-level tokenizer: a lookup table between characters and ids."""
    def __init__(self, text: str) -> None:
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.ch2idx, self.idx2ch = {}, {}
        for i, c in enumerate(self.vocab):
            self.ch2idx[c] = i
            self.idx2ch[i] = c

    def encode(self, text: str) -> List[int]:
        return [self.ch2idx[c] for c in text]

    def decode(self, x: List[int]) -> str:
        return ''.join([self.idx2ch[i] for i in x])


class SelfAttention(nn.Module):
    """One head of causal self-attention."""
    def __init__(
            self, head_size: int, n_embed: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # lower-triangular matrix: position t may only look at positions <= t
        self.register_buffer(
            'mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k = self.key(x)    # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)
        H = k.shape[-1]
        # compute the attention scores; dividing by sqrt(head size) keeps the
        # variance of the scores ~1 so that the softmax does not saturate
        attn_w = q @ k.transpose(-2, -1) * (H ** -0.5)  # (B, T, T)
        attn_w = attn_w.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn_w = F.softmax(attn_w, dim=-1)  # (B, T, T)
        attn_w = self.dropout(attn_w)
        # perform the weighted aggregation of values
        out = attn_w @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    """Several heads in parallel; their outputs are concatenated."""
    def __init__(self, num_heads, head_size, embed_dim, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(head_size, embed_dim, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, n_hdim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_hdim),
            nn.ReLU(),
            nn.Linear(n_hdim, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) then computation (MLP)."""
    def __init__(
            self, n_embed: int, num_heads: int, block_size: int,
            n_ff_hdim: int, dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            num_heads, n_embed // num_heads, n_embed, block_size, dropout)
        self.feed_forward = FeedForward(n_embed, n_ff_hdim, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm residual connections: the "residual stream" x is only ever
        # added to, which keeps gradients flowing through deep stacks
        x = x + self.self_attn(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(
            self, vocab_size: int, block_size: int, n_embed: int, n_head: int,
            n_layers: int, dropout: float, n_ff_hdim: int,
    ) -> None:
        super().__init__()
        assert n_embed % n_head == 0
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head, block_size, n_ff_hdim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(n_embed)
        self.out_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T = x.shape
        token_emb = self.token_emb(x)  # (B, T, H)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))  # (T, H)
        x = token_emb + pos_emb  # (B, T, H)
        x = self.blocks(x)  # (B, T, H)
        x = self.ln_final(x)
        logits = self.out_head(x)  # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(
            self, x: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        # generate a sequence of tokens given the input x of shape (B, T)
        for _ in range(max_new_tokens):
            # the position embedding only has block_size rows, so the model can
            # never look further back than block_size tokens
            idx_cond = x[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            new_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            x = torch.cat([x, new_tok], dim=-1)  # (B, T+1)
        return x


def main(args):
    set_seed(args.seed)

    # get the dataset
    text = get_dataset(args.data_path)

    # creating a mapping from unique characters to indices
    chmap = Mapper(text)
    print(f'Vocabulary size: {chmap.vocab_size} characters')

    # creating the training and validation sets
    data = Dataset(chmap.encode(text), split=args.train_test_split)

    model = Model(
        chmap.vocab_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_head=args.n_head,
        dropout=args.dropout,
        n_layers=args.n_layers,
        n_ff_hdim=args.ff_dim,
    ).to(args.device)
    print(f'Number of parameters: {count_parameters(model) / 1e6:.3f}M')

    if args.train or not os.path.exists(args.ckpt):
        train(model, data, args, args.ckpt)
    else:
        print(f'Loading the weights from {args.ckpt} (pass --train to retrain)')
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    # generate a sequence of characters
    model.eval()
    prompt = torch.tensor(
        [chmap.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(prompt, args.max_gen_len, args.temperature)
    print('-' * 80)
    print(chmap.decode(out[0].tolist()))


def get_args():
    parser = argparse.ArgumentParser(description='Character-level GPT')
    parser.add_argument('--data_path', default='data/tinyshakespeare.txt')
    parser.add_argument('--ckpt', default='models/gpt_char.pth')
    parser.add_argument('--train', action='store_true',
                        help='train even if a checkpoint already exists')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--n_embed', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--ff_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train_test_split', type=float, default=0.9)
    parser.add_argument('--max_iters', type=int, default=4000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--prompt', default='HERMIONE:')
    parser.add_argument('--max_gen_len', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    args.device = get_device()
    return args


if __name__ == '__main__':
    main(get_args())
