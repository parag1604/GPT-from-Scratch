"""
Helpers shared by all the versions: device selection, seeding, reading the
text file, sampling random training windows, and the training loop.

Nothing in here is specific to the tokenizer or to the model, which is why it
can be reused unchanged by version0.py (characters) and version1.py (BPE).
"""

import os
import random
from typing import Tuple

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


def get_dataset(fname: str) -> str:
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f'{fname} not found. Download tinyshakespeare first (see README.md, '
            'section "Data").')
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


class Dataset:
    """
    Holds the whole corpus as one long array of token ids.
    A training example is a random window of `block_size` tokens (x) together
    with the same window shifted one step to the right (y): at every position
    the model has to predict the next token.
    """
    def __init__(self, encoded: np.ndarray, split: float = 0.9) -> None:
        self.encoded = np.asarray(encoded, dtype=np.int64)
        n = int(len(self.encoded) * split)
        self.train_data = self.encoded[:n]
        self.val_data = self.encoded[n:]

    def get_batch(
            self, split: str = 'train', batch_size: int = 4, block_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        idx = np.random.randint(0, len(data) - block_size, batch_size)
        x = np.array([data[i:i+block_size] for i in idx])
        y = np.array([data[i+1:i+block_size+1] for i in idx])
        return x, y


@torch.no_grad()
def estimate_loss(model, data: Dataset, args) -> dict:
    # average the loss over several random batches, so that the printed
    # numbers are not dominated by the noise of a single batch
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        total = 0.0
        for _ in range(args.eval_iters):
            xb, yb = data.get_batch(split, args.batch_size, args.block_size)
            xb = torch.from_numpy(xb).to(args.device)
            yb = torch.from_numpy(yb).to(args.device)
            logits = model(xb)
            total += F.cross_entropy(logits.transpose(1, 2), yb).item()
        losses[split] = total / args.eval_iters
    model.train()
    return losses


def train(model, data: Dataset, args, ckpt_path: str) -> None:
    """Plain next-token prediction with AdamW, saving the weights at the end."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for i in range(args.max_iters + 1):
        if i % args.eval_interval == 0 or i == args.max_iters:
            losses = estimate_loss(model, data, args)
            print(f'Iteration {i:5d} | Train loss {losses["train"]:.4f} '
                  f'| Val loss {losses["val"]:.4f}')
        if i == args.max_iters:
            break

        xb, yb = data.get_batch('train', args.batch_size, args.block_size)
        xb = torch.from_numpy(xb).to(args.device)
        yb = torch.from_numpy(yb).to(args.device)

        logits = model(xb)  # (B, T, V)
        # cross_entropy expects the class dimension second: (B, V, T)
        loss = F.cross_entropy(logits.transpose(1, 2), yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f'Saved the weights to {ckpt_path}')


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
