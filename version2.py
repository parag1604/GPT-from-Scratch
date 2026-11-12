"""
Version 2: faster generation with a key-value (KV) cache.

What is NEW compared to version1.py
-----------------------------------
In version1, every generated token runs the model over the whole context
again, although the keys and values of all the earlier positions are exactly
the same as in the previous step (causal attention: a position never looks at
the future, so its k and v never change). That is O(T) redundant work per new
token.

A KV cache stores, for every layer, the keys and values computed so far. To
generate the next token we only push the ONE newest token through the network:
its query attends to the cached keys/values plus its own, and its k, v are
appended to the cache. Two phases:
  * prefill: run the whole prompt once, filling the cache;
  * decode : one token at a time, reusing the cache.

A subtlety with learned absolute position embeddings: the model only knows
positions 0 .. block_size-1. When the cache is full we cannot simply drop the
oldest entry, because the cached keys were computed with the old positions.
So we "re-prefill": recompute the cache from the most recent block_size // 2
tokens and continue. (Models with relative/rotary positions can slide the
window instead.)

The model has exactly the same parameters as version1, so this script loads
the checkpoint written by version1.py (models/gpt_bpe.pth). It then checks that
cached and uncached generation produce the same tokens, and times both.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_dataset, get_device, set_seed
import version1
from version1 import build_model, get_parser, get_tokenizer_and_tokens


class CachedMultiHeadAttention(version1.MultiHeadAttention):
    def forward(self, x, cache=None, start_pos=0):
        """
        x: (B, T, C) new tokens, sitting at positions start_pos .. start_pos+T-1
        cache: dict holding this layer's past 'k' and 'v' (B, nh, T_past, hs);
               it is updated in place. None = no caching (plain version1).
        """
        T = x.shape[1]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        if cache is not None:
            if 'k' in cache:  # append the new keys/values to the old ones
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        # query i is at absolute position start_pos + i and may look at every
        # key up to (and including) that position
        T_k = k.shape[2]
        mask = (torch.arange(T_k, device=x.device)[None, :]
                <= (start_pos + torch.arange(T, device=x.device))[:, None])
        out = self.attend(q, k, v, mask)
        return self.resid_dropout(self.proj(self.merge_heads(out)))


class CachedBlock(version1.Block):
    def __init__(self, n_embed, num_heads, block_size, n_ff_hdim, dropout):
        super().__init__(n_embed, num_heads, block_size, n_ff_hdim, dropout)
        self.self_attn = CachedMultiHeadAttention(
            num_heads, n_embed, block_size, dropout)

    def forward(self, x, cache=None, start_pos=0):
        x = x + self.self_attn(self.ln1(x), cache, start_pos)
        x = x + self.feed_forward(self.ln2(x))
        return x


class CachedModel(version1.Model):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layers,
                 dropout, n_ff_hdim):
        super().__init__(vocab_size, block_size, n_embed, n_head, n_layers,
                         dropout, n_ff_hdim)
        # same parameter names as version1, so its checkpoint loads directly
        self.blocks = nn.ModuleList([
            CachedBlock(n_embed, n_head, block_size, n_ff_hdim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, caches=None, start_pos=0):
        _, T = x.shape
        assert start_pos + T <= self.block_size
        pos = torch.arange(start_pos, start_pos + T, device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i], start_pos)
        return self.out_head(self.ln_final(x))

    def _prefill(self, x):
        caches = [{} for _ in self.blocks]
        logits = self(x, caches, start_pos=0)
        return logits[:, -1, :], caches, x.shape[1]

    @torch.no_grad()
    def generate_cached(self, x, max_new_tokens, temperature=1.0,
                        greedy=False):
        logits, caches, pos = self._prefill(x[:, -self.block_size:])
        for _ in range(max_new_tokens):
            if greedy:
                new_tok = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                new_tok = torch.multinomial(probs, 1)
            x = torch.cat([x, new_tok], dim=-1)
            if pos == self.block_size:
                # cache is full: rebuild it from the most recent half window
                logits, caches, pos = self._prefill(
                    x[:, -(self.block_size // 2):])
            else:
                # decode step: only the newest token goes through the model
                logits = self(new_tok, caches, start_pos=pos)[:, -1, :]
                pos += 1
        return x

    @torch.no_grad()
    def generate_uncached(self, x, max_new_tokens, greedy=False,
                          temperature=1.0):
        # version1's loop, but it must crop the context exactly like the cached
        # version does, so that the two produce the same tokens
        start = max(0, x.shape[1] - self.block_size)
        for _ in range(max_new_tokens):
            if x.shape[1] - start > self.block_size:
                start = x.shape[1] - self.block_size // 2
            logits = self(x[:, start:])[:, -1, :]
            if greedy:
                new_tok = logits.argmax(-1, keepdim=True)
            else:
                new_tok = torch.multinomial(
                    F.softmax(logits / temperature, dim=-1), 1)
            x = torch.cat([x, new_tok], dim=-1)
        return x


def timed(fn, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return out, time.time() - t0


def main(args):
    set_seed(args.seed)
    text = get_dataset(args.data_path)
    bpe, _ = get_tokenizer_and_tokens(text, args.vocab_size, args.cache_dir)

    model = build_model(args, bpe.vocab_size, model_cls=CachedModel)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
        print(f'Loaded the weights trained by version1.py from {args.ckpt}')
    else:
        print(f'{args.ckpt} not found: using random weights (run version1.py '
              'first to see meaningful text; the speed-up is the same)')
    model.eval()

    prompt = torch.tensor(
        [bpe.encode(args.prompt)], dtype=torch.long, device=args.device)

    # 1. correctness: greedy decoding must give identical tokens
    n = args.max_gen_len
    out_c, t_c = timed(
        lambda: model.generate_cached(prompt, n, greedy=True), args.device)
    out_u, t_u = timed(
        lambda: model.generate_uncached(prompt, n, greedy=True), args.device)
    same = torch.equal(out_c, out_u)
    print(f'Greedy decoding, {n} tokens: identical outputs = {same}')
    print(f'  without cache: {t_u:.2f}s | with KV cache: {t_c:.2f}s '
          f'| speed-up x{t_u / max(t_c, 1e-9):.1f}')

    # 2. a sample with the KV cache
    out = model.generate_cached(prompt, n, temperature=args.temperature)
    print('-' * 80)
    print(bpe.decode(out[0].tolist()))


if __name__ == '__main__':
    parser = get_parser('GPT generation with a KV cache')
    args = parser.parse_args()
    args.device = get_device()
    main(args)
