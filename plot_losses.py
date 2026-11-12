"""
Plot the validation loss of versions 0-3 (read from the training history that
utils.train() stores in each checkpoint), in nats per CHARACTER so that the
BPE model of version3 is comparable with the character-level ones.

    python plot_losses.py     # after running version0.py ... version3.py
"""

import argparse
import os
import pickle

import torch

RUNS = [('version0: bigram', 'checkpoints/bigram.pt', None),
        ('version1: one attention head', 'checkpoints/single_head.pt', None),
        ('version2: GPT, characters', 'checkpoints/gpt_char.pt', None),
        ('version3: GPT, BPE (512 tokens)', 'checkpoints/gpt_bpe.pt',
         'checkpoints/bpe_512.pkl')]


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    with open(args.data_path, encoding='utf-8') as f:
        n_chars = len(f.read())
    for (label, path, bpe), color in zip(RUNS, ['#999999', '#e69f00',
                                                '#0072b2', '#009e73']):
        if not os.path.exists(path):
            print(f'skipping {label}: {path} not found')
            continue
        hist = torch.load(path, map_location='cpu', weights_only=False)['history']
        scale = 1.0
        if bpe:  # nats per token -> nats per character
            with open(bpe, 'rb') as f:
                scale = len(pickle.load(f)['ids']) / n_chars
        ax.plot([h['iter'] for h in hist], [h['val'] * scale for h in hist],
                'o-', ms=3, color=color, label=label)
        best = min(h['val'] * scale for h in hist)
        print(f'{label:34s} best val {best:.3f} nats/char, '
              f'final {hist[-1]["val"] * scale:.3f}')
    ax.set_xlabel('training iteration')
    ax.set_ylabel('validation loss (nats per character)')
    ax.set_title('tinyshakespeare', fontsize=10)
    ax.set_ylim(1.2, 3.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='data/tinyshakespeare.txt')
    p.add_argument('--out', default='static/lm_val_loss.png')
    main(p.parse_args())
