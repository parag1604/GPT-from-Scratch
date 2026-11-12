"""
Version 1: the same GPT, but on Byte-Pair-Encoding (BPE) tokens.

What is NEW compared to version0.py
-----------------------------------
1. A BPE tokenizer, written from scratch (class `BPETokenizer`).
   Characters are a tiny vocabulary (65 symbols for tinyshakespeare), so the
   model has to spend its context window and its capacity on spelling.
   BPE starts from the characters and repeatedly merges the most frequent
   adjacent pair of tokens into a new token ("t"+"h" -> "th", "th"+"e" ->
   "the", ...) until the vocabulary reaches `vocab_size`. The same text then
   needs ~3x fewer tokens, so a context of 128 BPE tokens sees much more text
   than 128 characters.
   Encoding new text = replaying the learned merges in the order they were
   learned. The learned merges and the tokenized corpus are cached in
   ./preprocessed/ because learning them takes a minute or two.

2. Multi-head attention computed for all heads at once.
   version0 loops over a Python list of heads. Here one Linear layer produces
   the queries, keys and values of every head, and the heads are folded into
   a batch dimension (B, n_head, T, head_size), so a single batched matmul
   does the work of the loop. The maths is identical; it is just faster.

Everything else (data windows, training loop, sampling) is unchanged.
"""

import argparse
import os
import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import Dataset, count_parameters, get_dataset, get_device, \
    set_seed, train
from version0 import FeedForward


class BPETokenizer:
    """
    Byte-pair encoding over characters.

    vocab[i] is the string that token i stands for; merges[j] = (a, b) means
    that the j-th learned token is the concatenation of tokens a and b.
    """
    def __init__(self) -> None:
        self.base_chars: List[str] = []
        self.merges: List[tuple] = []
        self.vocab: List[str] = []
        self.ch2idx: dict = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _merge(ids: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
        """Replace every occurrence of the pair (a, b) in ids by new_id."""
        hits = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
        if len(hits) == 0:
            return ids
        if a == b:
            # overlapping occurrences, e.g. "aaa" contains (a, a) twice; like a
            # left-to-right scan, merge the first and skip the one it overlaps
            keep, last = [], -2
            for h in hits:
                if h != last + 1:
                    keep.append(h)
                    last = h
            hits = np.array(keep)
        ids = ids.copy()
        ids[hits] = new_id
        return np.delete(ids, hits + 1)

    def train(self, text: str, vocab_size: int) -> np.ndarray:
        """Learn the merges; returns the tokenized training text."""
        self.base_chars = sorted(list(set(text)))
        self.ch2idx = {c: i for i, c in enumerate(self.base_chars)}
        self.vocab = list(self.base_chars)
        self.merges = []
        ids = np.array([self.ch2idx[c] for c in text], dtype=np.int64)

        n_merges = vocab_size - len(self.vocab)
        for _ in tqdm(range(n_merges), desc='Learning BPE merges'):
            # count every adjacent pair (a, b) by encoding it as a * V + b
            pair_codes = ids[:-1] * vocab_size + ids[1:]
            counts = np.bincount(pair_codes)
            best = int(counts.argmax())
            if counts[best] < 2:  # nothing left that is worth merging
                break
            a, b = divmod(best, vocab_size)
            new_id = len(self.vocab)
            ids = self._merge(ids, a, b, new_id)
            self.merges.append((a, b))
            self.vocab.append(self.vocab[a] + self.vocab[b])
        return ids

    def encode(self, text: str) -> List[int]:
        unknown = set(text) - set(self.ch2idx)
        if unknown:
            raise ValueError(f'Characters not in the training text: {unknown}')
        ids = np.array([self.ch2idx[c] for c in text], dtype=np.int64)
        # replay the merges in the order in which they were learned
        for j, (a, b) in enumerate(self.merges):
            ids = self._merge(ids, a, b, len(self.base_chars) + j)
        return ids.tolist()

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.vocab[i] for i in ids)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'base_chars': self.base_chars,
                         'merges': self.merges}, f)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        tok = cls()
        tok.base_chars = state['base_chars']
        tok.ch2idx = {c: i for i, c in enumerate(tok.base_chars)}
        tok.merges = state['merges']
        tok.vocab = list(tok.base_chars)
        for a, b in tok.merges:
            tok.vocab.append(tok.vocab[a] + tok.vocab[b])
        return tok


def get_tokenizer_and_tokens(text: str, vocab_size: int, cache_dir: str):
    """Learn BPE once and cache it (the merges and the tokenized corpus)."""
    vocab_path = os.path.join(cache_dir, 'bpe_vocab.pkl')
    tokens_path = os.path.join(cache_dir, 'tokenized_text.pkl')
    if os.path.exists(vocab_path) and os.path.exists(tokens_path):
        tok = BPETokenizer.load(vocab_path)
        with open(tokens_path, 'rb') as f:
            ids = pickle.load(f)
        if tok.vocab_size == vocab_size:
            return tok, ids
        print('Cached tokenizer has a different vocab size, re-learning it')
    os.makedirs(cache_dir, exist_ok=True)
    tok = BPETokenizer()
    ids = tok.train(text, vocab_size)
    tok.save(vocab_path)
    with open(tokens_path, 'wb') as f:
        pickle.dump(ids, f)
    print(f'Vocab size: {tok.vocab_size}')
    print(f'Length of the text: {len(text)} characters -> {len(ids)} tokens')
    print('Some learned tokens:', [tok.vocab[i] for i in range(
        tok.vocab_size - 10, tok.vocab_size)])
    return tok, ids


class MultiHeadAttention(nn.Module):
    """All heads of causal self-attention computed in one batched matmul."""
    def __init__(self, num_heads, embed_dim, block_size, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        # one projection for the queries, keys and values of every head
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.tril(torch.ones(block_size, block_size)))

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, C = t.shape  # (B, T, C) -> (B, nh, T, hs)
        return t.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

    def merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = t.shape  # (B, nh, T, hs) -> (B, T, C)
        return t.transpose(1, 2).reshape(B, T, self.num_heads * self.head_size)

    def attend(self, q, k, v, mask):
        # exactly version0's single head, but with an extra head dimension
        attn_w = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        attn_w = attn_w.masked_fill(mask == 0, float('-inf'))
        attn_w = self.attn_dropout(F.softmax(attn_w, dim=-1))
        return attn_w @ v  # (B, nh, T, hs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 3 x (B, T, C)
        q, k, v = map(self.split_heads, (q, k, v))
        out = self.attend(q, k, v, self.mask[:T, :T])
        return self.resid_dropout(self.proj(self.merge_heads(out)))


class Block(nn.Module):
    def __init__(
            self, n_embed: int, num_heads: int, block_size: int,
            n_ff_hdim: int, dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            num_heads, n_embed, block_size, dropout)
        self.feed_forward = FeedForward(n_embed, n_ff_hdim, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(
            self, vocab_size: int, block_size: int, n_embed: int, n_head: int,
            n_layers: int, dropout: float, n_ff_hdim: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([
            Block(n_embed, n_head, block_size, n_ff_hdim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(n_embed)
        self.out_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        return self.out_head(self.ln_final(x))  # (B, T, V)

    @torch.no_grad()
    def generate(
            self, x: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(x[:, -self.block_size:])[:, -1, :] / temperature
            new_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            x = torch.cat([x, new_tok], dim=-1)
        return x


def build_model(args, vocab_size: int, model_cls=Model):
    return model_cls(
        vocab_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_head=args.n_head,
        dropout=args.dropout,
        n_layers=args.n_layers,
        n_ff_hdim=args.ff_dim,
    ).to(args.device)


def main(args):
    set_seed(args.seed)
    text = get_dataset(args.data_path)

    # learn (or load) the byte-pair encoding of the corpus
    bpe, ids = get_tokenizer_and_tokens(text, args.vocab_size, args.cache_dir)
    data = Dataset(ids, split=args.train_test_split)

    model = build_model(args, bpe.vocab_size)
    print(f'Number of parameters: {count_parameters(model) / 1e6:.3f}M')

    if args.train or not os.path.exists(args.ckpt):
        train(model, data, args, args.ckpt)
    else:
        print(f'Loading the weights from {args.ckpt} (pass --train to retrain)')
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    model.eval()
    prompt = torch.tensor(
        [bpe.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(prompt, args.max_gen_len, args.temperature)
    print('-' * 80)
    print(bpe.decode(out[0].tolist()))


def get_parser(description='GPT on BPE tokens'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_path', default='data/tinyshakespeare.txt')
    parser.add_argument('--cache_dir', default='preprocessed')
    parser.add_argument('--ckpt', default='models/gpt_bpe.pth')
    parser.add_argument('--train', action='store_true',
                        help='train even if a checkpoint already exists')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--vocab_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--n_embed', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--ff_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train_test_split', type=float, default=0.9)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--prompt', default='HERMIONE:\n')
    parser.add_argument('--max_gen_len', type=int, default=300)
    # Parag's original sampled from softmax(1.5 * logits), i.e. T = 1/1.5
    parser.add_argument('--temperature', type=float, default=0.67)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    args.device = get_device()
    main(args)
