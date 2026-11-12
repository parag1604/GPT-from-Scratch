"""
Version 3 (optional): the same GPT on byte-pair-encoding (BPE) tokens.

What is NEW compared to version2.py
-----------------------------------
Only the tokenizer. Characters are a tiny vocabulary, so the model spends its
context window and its capacity on spelling. BPE (Sennrich et al., 2016; the
scheme behind GPT-2's tokenizer) starts from the characters and repeatedly
merges the most frequent adjacent pair of tokens into a new token
("t"+"h" -> "th", "th"+"e" -> "the", ...) until the vocabulary has
`vocab_size` tokens. Encoding new text = replaying the learned merges in the
order they were learned. The tokenizer below is written from scratch with
numpy; learning 448 merges on tinyshakespeare takes well under a minute.

Comparing losses across tokenizers: a loss per BPE token is not comparable
with a loss per character, because one BPE token covers several characters.
The script therefore also reports the validation loss in nats per CHARACTER:
    loss_per_char = loss_per_token * (number of tokens) / (number of chars).

Everything else (model, training loop, sampling) is version2.py unchanged.
The other versions stay at character level, which keeps the KV-cache and the
LoRA experiments simple.
"""

import argparse
import os
import pickle
from dataclasses import asdict
from typing import List

import numpy as np
import torch

from utils import (Dataset, add_training_args, count_parameters, finish_args,
                   load_checkpoint, read_text, set_seed, train)
from version2 import GPT, add_model_args, config_from_args


class BPETokenizer:
    """
    Byte-pair encoding over characters.

    vocab[i] is the string that token i stands for; merges[j] = (a, b) means
    that token len(base_chars) + j is the concatenation of tokens a and b.
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
        self.base_chars = sorted(set(text))
        self.ch2idx = {c: i for i, c in enumerate(self.base_chars)}
        self.vocab = list(self.base_chars)
        self.merges = []
        ids = np.array([self.ch2idx[c] for c in text], dtype=np.int64)
        for _ in range(vocab_size - len(self.vocab)):
            # count every adjacent pair (a, b) by encoding it as a * V + b
            counts = np.bincount(ids[:-1] * vocab_size + ids[1:])
            best = int(counts.argmax())
            if counts[best] < 2:  # nothing left that is worth merging
                break
            a, b = divmod(best, vocab_size)
            ids = self._merge(ids, a, b, len(self.vocab))
            self.merges.append((a, b))
            self.vocab.append(self.vocab[a] + self.vocab[b])
        return ids

    def encode(self, text: str) -> List[int]:
        ids = np.array([self.ch2idx[c] for c in text], dtype=np.int64)
        for j, (a, b) in enumerate(self.merges):  # replay, in learned order
            ids = self._merge(ids, a, b, len(self.base_chars) + j)
        return ids.tolist()

    def decode(self, ids) -> str:
        return ''.join(self.vocab[i] for i in ids)

    def state_dict(self) -> dict:
        return {'base_chars': self.base_chars, 'merges': self.merges}

    @classmethod
    def from_state_dict(cls, state: dict) -> 'BPETokenizer':
        tok = cls()
        tok.base_chars = state['base_chars']
        tok.ch2idx = {c: i for i, c in enumerate(tok.base_chars)}
        tok.merges = [tuple(m) for m in state['merges']]
        tok.vocab = list(tok.base_chars)
        for a, b in tok.merges:
            tok.vocab.append(tok.vocab[a] + tok.vocab[b])
        return tok


def get_tokenizer_and_tokens(text: str, vocab_size: int, cache_path: str):
    """Learn BPE once and cache the merges and the tokenized corpus."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            state = pickle.load(f)
        tok = BPETokenizer.from_state_dict(state['tokenizer'])
        if tok.vocab_size == vocab_size:
            return tok, state['ids']
    tok = BPETokenizer()
    ids = tok.train(text, vocab_size)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'tokenizer': tok.state_dict(), 'ids': ids}, f)
    return tok, ids


def main(args):
    set_seed(args.seed)
    text = read_text(args.data_path)
    tok, ids = get_tokenizer_and_tokens(text, args.vocab_size, args.bpe_cache)
    chars_per_token = len(text) / len(ids)
    print(f'{len(text):,} characters -> {len(ids):,} BPE tokens '
          f'({chars_per_token:.2f} characters per token)')
    print('Last tokens learned:',
          [tok.vocab[i] for i in range(tok.vocab_size - 12, tok.vocab_size)])
    longest = sorted(tok.vocab, key=len)[-8:]
    print('Longest tokens:', longest)
    data = Dataset(ids)

    cfg = config_from_args(args, tok.vocab_size)
    model = GPT(cfg).to(args.device)
    print(f'Parameters: {count_parameters(model) / 1e6:.2f}M; the context of '
          f'{args.block_size} tokens covers ~{args.block_size * chars_per_token:.0f}'
          ' characters')

    if args.train or not os.path.exists(args.ckpt):
        history = train(model, data, args, args.ckpt, extra_state={
            'config': asdict(cfg), 'bpe': tok.state_dict()})
    else:
        print(f'Loading {args.ckpt} (pass --train to retrain)')
        ckpt = load_checkpoint(args.ckpt, args.device)
        model.load_state_dict(ckpt['model'])
        history = ckpt['history']
    val = history[-1]['val']
    print(f'Final validation loss: {val:.4f} nats/token = '
          f'{val / chars_per_token:.4f} nats/character')

    model.eval()
    prompt = torch.tensor([tok.encode(args.prompt)], device=args.device)
    out = model.generate(prompt, args.max_new_tokens, args.temperature)
    print('-' * 80)
    print(tok.decode(out[0].tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT on BPE tokens')
    add_training_args(parser, block_size=128, max_iters=3000,
                      eval_interval=500)
    add_model_args(parser)
    parser.add_argument('--vocab_size', type=int, default=512)
    parser.add_argument('--bpe_cache', default=None,
                        help='default: checkpoints/bpe_<vocab_size>.pkl')
    parser.add_argument('--prompt', default='\n')
    parser.add_argument('--ckpt', default='checkpoints/gpt_bpe.pt')
    parser.set_defaults(max_new_tokens=200)
    args = finish_args(parser.parse_args())
    args.bpe_cache = args.bpe_cache or f'checkpoints/bpe_{args.vocab_size}.pkl'
    main(args)
