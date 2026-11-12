"""
Version 0: the bigram language model (the baseline).

What this file teaches
----------------------
* Tokenization at its simplest: every distinct character is a token
  (65 of them in tinyshakespeare).
* Language modelling as next-token prediction: the target sequence is the
  input sequence shifted by one position.
* The simplest possible neural language model: a V x V lookup table. Row i
  holds the logits of the character that follows character i. The model sees
  only ONE character of context, so it can learn letter-pair statistics
  ("q" is followed by "u") and nothing more.
* Autoregressive sampling: predict a distribution over the next character,
  sample from it, append the sample to the input, repeat.

A uniform guess costs ln(65) = 4.17 nats per character; the bigram model gets
to about 2.5. Every later version keeps this training loop and this sampling
loop and only changes the model.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (CharTokenizer, Dataset, add_training_args, count_parameters,
                   finish_args, load_checkpoint, read_text, set_seed, train)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token
        self.logits_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.logits_table(idx)  # (B, T) -> (B, T, V)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :] / temperature  # only the last step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def main(args):
    set_seed(args.seed)
    text = read_text(args.data_path)
    tok = CharTokenizer.from_text(text)
    print(f'{len(text):,} characters, vocabulary of {tok.vocab_size}: '
          f'{"".join(tok.chars)!r}')
    data = Dataset(tok.encode(text))

    model = BigramLanguageModel(tok.vocab_size).to(args.device)
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
    parser = argparse.ArgumentParser(description='Bigram language model')
    add_training_args(parser, batch_size=32, block_size=8, max_iters=5000,
                      learning_rate=1e-2, eval_interval=1000, eval_iters=200)
    parser.add_argument('--ckpt', default='checkpoints/bigram.pt')
    main(finish_args(parser.parse_args()))
