"""
Version 4: fast generation with a key-value (KV) cache.

What is NEW compared to version2.py
-----------------------------------
Generation in version2 re-runs the whole model on the whole context for every
new token. But causal attention never lets a position look at the future, so
the keys and values of the earlier positions are exactly the same as in the
previous step: generating T tokens recomputes them O(T^2) times in total.

A KV cache stores, for every layer, the keys and values computed so far.
Generation then has two phases:
  * prefill: run the prompt through the model once, filling the cache;
  * decode : push only the ONE newest token through the network. Its query
             attends to the cached keys/values plus its own; its key and value
             are appended to the cache.
Each decode step now costs O(T) (one query against T keys) instead of
O(T^2), and the token-wise layers (MLP, projections) run on one position
instead of T. The price is memory: 2 x n_layer x T x n_embd numbers per
sequence (1.2M numbers, 4.7 MB in float32, for the default model at
T = 256).

A subtlety with learned absolute position embeddings: the model only knows
positions 0 .. block_size-1. When the cache is full we cannot simply drop the
oldest entry, because every cached key was computed at its old position. So
we "re-prefill": rebuild the cache from the most recent block_size // 2
tokens and carry on. (Models with relative or rotary position encodings can
slide the window instead.) The uncached reference crops its context in the
same way, so the two must produce identical tokens.

The model has exactly the same parameters as version2's GPT, so this script
loads the checkpoint written by version2.py. It
  1. checks that greedy decoding with and without the cache gives identical
     tokens (also across a re-prefill), and reports the largest logit gap;
  2. times generation of N tokens with and without the cache for growing N,
     for the trained model and for a 1024-token-context model with random
     weights (the speed does not depend on the values of the weights), and
     plots the result to static/kv_cache_timing.png;
  3. prints a sample generated with the cache.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import version2
from utils import finish_args, set_seed
from utils import CharTokenizer
from version2 import GPTConfig, build_from_checkpoint


class CachedMultiHeadAttention(version2.MultiHeadAttention):
    def forward(self, x, cache=None, start_pos=0):
        """
        x:     (B, T, C) new tokens, at positions start_pos .. start_pos+T-1
        cache: dict with this layer's past 'k' and 'v' (B, nh, T_past, hs),
               updated in place; None = no caching (plain version2).
        """
        T = x.shape[1]
        q, k, v = map(self.split_heads, self.qkv(x).chunk(3, dim=-1))
        if cache is not None:
            if 'k' in cache:  # append the new keys/values to the old ones
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        # query i sits at absolute position start_pos + i and may look at every
        # key up to and including that position
        T_k = k.shape[2]
        mask = (torch.arange(T_k, device=x.device)[None, :]
                <= (start_pos + torch.arange(T, device=x.device))[:, None])
        out = self.attend(q, k, v, mask)
        return self.resid_dropout(self.proj(self.merge_heads(out)))


class CachedBlock(version2.Block):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.attn = CachedMultiHeadAttention(cfg)  # same parameter names

    def forward(self, x, cache=None, start_pos=0):
        x = x + self.attn(self.ln1(x), cache, start_pos)
        x = x + self.mlp(self.ln2(x))
        return x


class CachedGPT(version2.GPT):
    def __init__(self, cfg):
        super().__init__(cfg)
        # same parameter names as version2.GPT, so its checkpoint loads as is
        self.blocks = nn.ModuleList([CachedBlock(cfg)
                                     for _ in range(cfg.n_layer)])
        self.apply(self._init_weights)

    def forward(self, idx, caches=None, start_pos=0):
        B, T = idx.shape
        assert start_pos + T <= self.block_size
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i], start_pos)
        return self.lm_head(self.ln_f(x))

    def _prefill(self, idx):
        caches = [{} for _ in self.blocks]
        logits = self(idx, caches, start_pos=0)
        return logits[:, -1, :], caches, idx.shape[1]

    @staticmethod
    def _pick(logits, greedy, temperature):
        if greedy:
            return logits.argmax(-1, keepdim=True)
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)

    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens, temperature=1.0,
                        greedy=False, record=None):
        logits, caches, pos = self._prefill(idx[:, -self.block_size:])
        for _ in range(max_new_tokens):
            if record is not None:
                record.append(logits.float().cpu())
            idx_next = self._pick(logits, greedy, temperature)
            idx = torch.cat([idx, idx_next], dim=1)
            if pos == self.block_size:
                # the cache is full: rebuild it from the most recent half window
                logits, caches, pos = self._prefill(
                    idx[:, -(self.block_size // 2):])
            else:
                # decode step: only the newest token goes through the model
                logits = self(idx_next, caches, start_pos=pos)[:, -1, :]
                pos += 1
        return idx

    @torch.no_grad()
    def generate_uncached(self, idx, max_new_tokens, temperature=1.0,
                          greedy=False, record=None):
        # version2's loop, but cropping the context exactly like the cached
        # version does, so that the two produce the same tokens
        start = max(0, idx.shape[1] - self.block_size)
        for _ in range(max_new_tokens):
            if idx.shape[1] - start > self.block_size:
                start = idx.shape[1] - self.block_size // 2
            logits = self(idx[:, start:])[:, -1, :]
            if record is not None:
                record.append(logits.float().cpu())
            idx_next = self._pick(logits, greedy, temperature)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def timed(fn, device):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return out, time.perf_counter() - t0


def time_curve(model, lengths, device, repeats):
    """Seconds to generate N tokens from a 1-token prompt, for N in lengths."""
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    model.generate_cached(prompt, 8, greedy=True)  # warm-up
    model.generate_uncached(prompt, 8, greedy=True)
    rows = []
    for n in lengths:
        tc = min(timed(lambda: model.generate_cached(prompt, n, greedy=True),
                       device)[1] for _ in range(repeats))
        tu = min(timed(lambda: model.generate_uncached(prompt, n, greedy=True),
                       device)[1] for _ in range(repeats))
        rows.append((n, tu, tc))
        print(f'  N = {n:5d} | no cache {tu:7.3f}s | KV cache {tc:7.3f}s '
              f'| speed-up x{tu / tc:5.1f}', flush=True)
    return rows


def plot(curves, path, device_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.8))
    for ax, (title, rows) in zip(axes, curves):
        n = [r[0] for r in rows]
        ax.plot(n, [r[1] for r in rows], 'o-', color='#d55e00',
                label='no cache (re-run the whole context)')
        ax.plot(n, [r[2] for r in rows], 's-', color='#0072b2',
                label='KV cache')
        ax.set_xlabel('tokens generated, N')
        ax.set_ylabel('seconds')
        ax.set_title(title, fontsize=10)
        ax.text(0.03, 0.70, f'speed-up at N = {n[-1]}: x{rows[-1][1] / rows[-1][2]:.1f}',
                transform=ax.transAxes, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(f'Greedy generation time ({device_name})', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f'Saved {path}')


def main(args):
    set_seed(args.seed)
    device = args.device
    model, ckpt = build_from_checkpoint(args.ckpt, device, CachedGPT)
    model.eval()
    tok = CharTokenizer.from_state_dict(ckpt['tokenizer'])
    print(f'Loaded the version2 model from {args.ckpt} '
          f'(context {model.block_size} characters)')

    # 1. correctness: identical greedy tokens, also across a re-prefill
    prompt = torch.tensor([tok.encode(args.prompt)], device=device)
    n = args.check_tokens
    rec_c, rec_u = [], []
    out_c = model.generate_cached(prompt, n, greedy=True, record=rec_c)
    out_u = model.generate_uncached(prompt, n, greedy=True, record=rec_u)
    gap = (torch.stack(rec_c) - torch.stack(rec_u)).abs().max().item()
    print(f'Greedy decoding of {n} tokens: identical outputs = '
          f'{torch.equal(out_c, out_u)}; max |logit difference| = {gap:.2e}')

    # 2. timing
    curves = []
    lengths = [n for n in args.lengths if n < model.block_size]
    print(f'Timing the trained model ({device}):')
    curves.append((f'trained char model, context {model.block_size}',
                   time_curve(model, lengths, device, args.repeats)))
    if args.long_context:
        cfg = GPTConfig(**{**ckpt['config'], 'block_size': args.long_context})
        long_model = CachedGPT(cfg).to(device).eval()
        long_lengths = [n for n in args.long_lengths if n < args.long_context]
        print(f'Timing a random-weight model with context {args.long_context}:')
        curves.append((f'same size, random weights, context {args.long_context}',
                       time_curve(long_model, long_lengths, device,
                                  args.repeats)))
    if args.plot:
        name = (torch.cuda.get_device_name() if device.type == 'cuda'
                else str(device))
        plot(curves, args.plot, name)

    # 3. a sample, generated with the cache
    out = model.generate_cached(prompt, args.max_new_tokens,
                                temperature=args.temperature)
    print('-' * 80)
    print(tok.decode(out[0].tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT generation with a KV cache')
    parser.add_argument('--ckpt', default='checkpoints/gpt_char.pt')
    parser.add_argument('--prompt', default='ROMEO:')
    parser.add_argument('--check_tokens', type=int, default=600,
                        help='> block_size, so that a re-prefill is tested too')
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=[16, 32, 64, 128, 192, 255])
    parser.add_argument('--long_context', type=int, default=1024,
                        help='0 to skip the random-weight long-context model')
    parser.add_argument('--long_lengths', type=int, nargs='+',
                        default=[64, 128, 256, 512, 768, 1023])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--plot', default='static/kv_cache_timing.png')
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default=None)
    main(finish_args(parser.parse_args()))
