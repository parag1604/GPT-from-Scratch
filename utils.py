"""
Helpers shared by all the versions: device selection, seeding, reading text,
a character-level tokenizer, random training windows, and the training loop.

Nothing in here depends on the model, which is why the same code trains the
bigram model of version0.py and the full GPT of version2.py.
"""

import math
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    # prefer an NVIDIA GPU, then Apple silicon, then the CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found (see README.md, "Data").')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class CharTokenizer:
    """
    Character-level tokenizer: every distinct character is a token.

    `unk` (optional) is a character that stands in for any character that was
    not seen when the vocabulary was built (needed in version5.py, where the
    model meets tweets with emoji, accents, ...).
    """
    def __init__(self, chars: List[str], unk: Optional[str] = None) -> None:
        self.chars = list(chars)
        self.unk = unk
        if unk is not None and unk not in self.chars:
            self.chars.append(unk)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str, unk: Optional[str] = None,
                  min_count: int = 1) -> 'CharTokenizer':
        counts = {}
        for c in text:
            counts[c] = counts.get(c, 0) + 1
        chars = sorted(c for c, n in counts.items() if n >= min_count)
        return cls(chars, unk)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        if self.unk is None:
            return [self.stoi[c] for c in text]
        u = self.stoi[self.unk]
        return [self.stoi.get(c, u) for c in text]

    def decode(self, ids) -> str:
        return ''.join(self.chars[i] for i in ids)

    def state_dict(self) -> dict:
        return {'chars': self.chars, 'unk': self.unk}

    @classmethod
    def from_state_dict(cls, state: dict) -> 'CharTokenizer':
        return cls(state['chars'], state['unk'])


class Dataset:
    """
    Holds the whole corpus as one long array of token ids (first 90% for
    training, last 10% for validation).
    A training example is a random window of `block_size` tokens (x) together
    with the same window shifted one step to the right (y): at every position
    the model has to predict the next token.
    """
    def __init__(self, encoded, split: float = 0.9) -> None:
        self.encoded = np.asarray(encoded, dtype=np.int64)
        n = int(len(self.encoded) * split)
        self.train_data = self.encoded[:n]
        self.val_data = self.encoded[n:]

    def get_batch(self, split: str, batch_size: int, block_size: int,
                  device) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        idx = np.random.randint(0, len(data) - block_size - 1, batch_size)
        x = np.stack([data[i:i + block_size] for i in idx])
        y = np.stack([data[i + 1:i + block_size + 1] for i in idx])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits (B, T, V), targets (B, T): average next-token cross-entropy
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1))


@torch.no_grad()
def estimate_loss(model, data: Dataset, args) -> dict:
    # average the loss over several random batches, so that the printed
    # numbers are not dominated by the noise of a single batch
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(args.eval_iters):
            x, y = data.get_batch(split, args.batch_size, args.block_size,
                                  args.device)
            losses.append(lm_loss(model(x), y).item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def get_lr(it: int, args) -> float:
    """Linear warm-up, then cosine decay to lr / 10 (as in nanoGPT)."""
    warmup = getattr(args, 'warmup_iters', 0)
    if it < warmup:
        return args.learning_rate * (it + 1) / warmup
    if not getattr(args, 'cosine', False):
        return args.learning_rate
    progress = (it - warmup) / max(1, args.max_iters - warmup)
    min_lr = args.learning_rate / 10
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (
        args.learning_rate - min_lr)


def train(model, data: Dataset, args, ckpt_path: str,
          extra_state: Optional[dict] = None) -> List[dict]:
    """
    Plain next-token prediction with AdamW. Prints and returns the train /
    validation losses every `eval_interval` iterations and saves the weights
    (plus `extra_state`, e.g. the tokenizer) at the end.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=getattr(args, 'weight_decay', 0.0))
    history, t0 = [], time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, data, args)
            history.append({'iter': it, **losses})
            print(f'iter {it:5d} | train loss {losses["train"]:.4f} '
                  f'| val loss {losses["val"]:.4f} | {time.time() - t0:.0f}s',
                  flush=True)
        if it == args.max_iters:
            break
        for g in optimizer.param_groups:
            g['lr'] = get_lr(it, args)
        x, y = data.get_batch('train', args.batch_size, args.block_size,
                              args.device)
        loss = lm_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if getattr(args, 'grad_clip', 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    torch.save({'model': model.state_dict(), 'history': history,
                **(extra_state or {})}, ckpt_path)
    print(f'Saved the weights to {ckpt_path}')
    return history


def load_checkpoint(path: str, device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def count_parameters(model, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def add_training_args(parser, **defaults):
    """Arguments shared by every language-model version."""
    d = dict(batch_size=64, block_size=256, max_iters=5000,
             learning_rate=3e-4, eval_interval=500, eval_iters=100)
    d.update(defaults)
    parser.add_argument('--data_path', default='data/tinyshakespeare.txt')
    parser.add_argument('--train', action='store_true',
                        help='train even if a checkpoint already exists')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_size', type=int, default=d['batch_size'])
    parser.add_argument('--block_size', type=int, default=d['block_size'],
                        help='context length (number of tokens)')
    parser.add_argument('--max_iters', type=int, default=d['max_iters'])
    parser.add_argument('--learning_rate', type=float,
                        default=d['learning_rate'])
    parser.add_argument('--eval_interval', type=int,
                        default=d['eval_interval'])
    parser.add_argument('--eval_iters', type=int, default=d['eval_iters'])
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', default=None,
                        help='cuda | mps | cpu (default: the best available)')
    return parser


def finish_args(args):
    args.device = torch.device(args.device) if args.device else get_device()
    return args
