"""
Version 5: OpenAI's pretrained GPT-2 inside our own GPT code.

What is NEW compared to version4.py
-----------------------------------
Our Shakespeare model has 10.8M parameters and has read 1 MB of text. GPT-2
small has 124M parameters and was pretrained on ~40 GB of web text (WebText).
Its architecture is the GPT of version2.py with a handful of small changes
(see gpt2.py), so we can write it ourselves and copy OpenAI's weights into it:

  * gpt2.py: GPT-2 from scratch; `GPT2.from_pretrained('gpt2')` downloads the
    official checkpoint through Hugging Face `transformers` and copies every
    tensor in, transposing the four matrices OpenAI stored as "Conv1D";
  * the tokenizer is GPT-2's byte-level BPE (version3's algorithm, run to
    50,257 tokens on bytes instead of characters).

The script
  1. checks that our implementation reproduces Hugging Face's GPT-2 logits;
  2. samples a continuation of a prompt (no training at all);
  3. tries ZERO-SHOT tweet sentiment, i.e. no training either: it appends
     "The sentiment of this tweet is" to every tweet and compares the
     probabilities GPT-2 gives to the words " negative", " neutral" and
     " positive". This is the floor that the fine-tuning of version6.py has
     to beat (GPT-2 small is not instruction-tuned, so expect little).

Data for step 3: data/tweet_sentiment/ (TweetEval sentiment; see README.md).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from gpt2 import GPT2, get_tokenizer
from utils import finish_args, set_seed

LABELS = ['negative', 'neutral', 'positive']


# --------------------------------------------------------------------------
# tweets
# --------------------------------------------------------------------------
def read_tweets(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found (see README.md, "Data").')
    texts, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            label, text = line.rstrip('\n').split('\t', 1)
            labels.append(int(label))
            texts.append(text)
    return texts, np.array(labels)


def encode_split(texts, labels, tokenizer, max_len):
    """Tokenize, truncate to max_len tokens and right-pad into one tensor."""
    ids = [tokenizer.encode(t)[:max_len] for t in texts]
    X = torch.full((len(ids), max_len), tokenizer.eos_token_id)
    M = torch.zeros(len(ids), max_len, dtype=torch.bool)
    for i, s in enumerate(ids):
        X[i, :len(s)] = torch.tensor(s)
        M[i, :len(s)] = True
    return X, M, torch.as_tensor(labels)


def batches(split, batch_size, shuffle, device):
    X, M, Y = split
    order = torch.randperm(len(Y)) if shuffle else torch.arange(len(Y))
    for i in range(0, len(Y), batch_size):
        j = order[i:i + batch_size]
        L = int(M[j].sum(1).max())  # trim the padding to this batch's longest
        yield X[j, :L].to(device), M[j, :L].to(device), Y[j].to(device)


def metrics(pred, y):
    """Accuracy and macro-averaged recall (TweetEval's official metric)."""
    pred, y = np.asarray(pred), np.asarray(y)
    recalls = [(pred[y == k] == k).mean() for k in range(len(LABELS))]
    return {'acc': float((pred == y).mean()),
            'macro_recall': float(np.mean(recalls))}


# --------------------------------------------------------------------------
# zero-shot classification by prompting
# --------------------------------------------------------------------------
@torch.no_grad()
def zero_shot(model, tokenizer, texts, labels, device, batch_size=64,
              prompt='\nThe sentiment of this tweet is'):
    model.eval()
    label_ids = [tokenizer.encode(' ' + w)[0] for w in LABELS]
    assert all(len(tokenizer.encode(' ' + w)) == 1 for w in LABELS)
    X, M, Y = encode_split([t + prompt for t in texts], labels, tokenizer,
                           max_len=128)
    preds = []
    for x, m, _ in batches((X, M, Y), batch_size, False, device):
        h = model.hidden_states(x)
        h_last = h[torch.arange(len(h), device=device), m.sum(1) - 1]
        logits = model.lm_head(h_last)[:, label_ids]  # 3 of the 50,257 words
        preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    return {**metrics(pred, labels),
            'predicted_class_counts': np.bincount(pred, minlength=3).tolist()}


def main(args):
    set_seed(args.seed)
    device = args.device
    tokenizer = get_tokenizer(args.model)
    model = GPT2.from_pretrained(args.model).to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'Loaded {args.model} into our GPT2 class: {n / 1e6:.1f}M parameters')

    # 1. our implementation == Hugging Face's
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    x = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    with torch.no_grad():
        gap = (model(x) - hf(x).logits).abs().max().item()
    print(f'Max |logit difference| vs Hugging Face GPT-2: {gap:.2e}')
    del hf

    # 2. a sample
    out = model.generate(x, args.max_new_tokens, temperature=0.8, top_k=50)
    print('-' * 80)
    print(tokenizer.decode(out[0].tolist()))
    print('-' * 80)

    # 3. zero-shot sentiment on the test set
    texts, labels = read_tweets(os.path.join(args.tweet_dir, 'test.tsv'))
    res = zero_shot(model, tokenizer, texts, labels, device)
    print(f'Zero-shot on {len(labels):,} test tweets: accuracy '
          f'{res["acc"]:.3f}, macro-recall {res["macro_recall"]:.3f}; '
          f'predicted class counts {res["predicted_class_counts"]} '
          f'(true {np.bincount(labels).tolist()})')
    os.makedirs(os.path.dirname(args.results) or '.', exist_ok=True)
    with open(args.results, 'w') as f:
        json.dump({'logit_gap_vs_hf': gap, 'zero_shot_test': res}, f, indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Pretrained GPT-2, from scratch')
    p.add_argument('--model', default='gpt2')
    p.add_argument('--prompt', default='The meaning of life is')
    p.add_argument('--max_new_tokens', type=int, default=60)
    p.add_argument('--tweet_dir', default='data/tweet_sentiment')
    p.add_argument('--results', default='checkpoints/zero_shot.json')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default=None)
    main(finish_args(p.parse_args()))
