"""
Version 6: fine-tuning GPT-2 for tweet sentiment, with LoRA.

What is NEW compared to version5.py
-----------------------------------
Zero-shot prompting of GPT-2 small barely works (version5). Now we train on
labelled tweets (negative / neutral / positive) and compare four ways of
adapting the pretrained network, from cheapest to most expensive:

  * linear probe:     freeze all 124M GPT-2 weights; train only a linear head
                      on the hidden state of the last token (2,307 params);
  * LoRA, rank r:     freeze GPT-2 and add a trainable low-rank update
                      dW = (alpha / r) B A to every attention and MLP matrix
                      (c_attn, c_proj, c_fc in all 12 blocks; lora.py), and
                      train those plus the head;
  * full fine-tuning: train every weight (the reference).

The classifier (gpt2.GPT2Classifier) reads the hidden state of the LAST real
token: GPT-2 is causal, so it is the only one that has seen the whole tweet.

LoRA mechanics, checked by the script:
  * at initialisation (A random, B = 0) the LoRA model gives exactly the same
    outputs as the frozen model; B receives a non-zero gradient and A a zero
    one (A starts moving once B is non-zero);
  * after training, merging W <- W + (alpha / r) B A into the frozen weights
    gives the same logits (up to float rounding) and the same predictions,
    with no extra cost at inference time.

All methods see the same data, the same number of epochs and the same head
initialisation; the validation set is evaluated every `--eval_every` steps
(the curves in static/), the step with the best validation macro-recall is
kept and evaluated ONCE on the test set.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from gpt2 import GPT2, GPT2Classifier, get_tokenizer
from lora import apply_lora, lora_modules, merge_lora
from utils import count_parameters, finish_args, set_seed
from version5 import LABELS, batches, encode_split, metrics, read_tweets


def autocast(args):
    if args.amp and args.device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build(method, args, pretrained_state):
    gpt = GPT2(pretrained_state['cfg'])
    gpt.load_state_dict(pretrained_state['weights'])
    if method != 'full':
        for p in gpt.parameters():
            p.requires_grad = False
    if method.startswith('lora'):
        r = int(method[4:])
        apply_lora(gpt, targets=('c_attn', 'c_proj', 'c_fc'), rank=r,
                   alpha=args.alpha_over_rank * r)
    torch.manual_seed(args.seed)  # the same head initialisation for everyone
    return GPT2Classifier(gpt, len(LABELS), args.pool).to(args.device)


@torch.no_grad()
def predict(model, split, args, amp=True):
    model.eval()
    out = []
    for x, m, _ in batches(split, 128, False, args.device):
        with (autocast(args) if amp else contextlib.nullcontext()):
            out.append(model(x, m).float().cpu())
    return torch.cat(out)


def evaluate(model, split, args):
    return metrics(predict(model, split, args).argmax(1), split[2])


def fit(model, data, lr, args, tag):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(data['train'][2]) + args.batch_size - 1) // args.batch_size
    total = args.epochs * steps_per_epoch
    # linear warm-up over the first 10% of the steps, then linear decay to 0
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm + 1))))
    curve = [{'step': 0, **evaluate(model, data['val'], args)}]
    best, best_state, step, t0 = curve[0]['macro_recall'], None, 0, time.time()
    for epoch in range(args.epochs):
        for x, m, y in batches(data['train'], args.batch_size, True,
                               args.device):
            model.train()
            with autocast(args):
                loss = F.cross_entropy(model(x, m).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % args.eval_every == 0 or step == total:
                val = evaluate(model, data['val'], args)
                curve.append({'step': step, **val})
                print(f'  [{tag}] step {step:5d}/{total} | loss {loss.item():.3f}'
                      f' | val acc {val["acc"]:.3f} macro-recall '
                      f'{val["macro_recall"]:.3f} | {time.time() - t0:.0f}s',
                      flush=True)
                if val['macro_recall'] > best:
                    best = val['macro_recall']
                    best_state = {n: p.detach().clone()
                                  for n, p in model.named_parameters()
                                  if p.requires_grad}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return curve, time.time() - t0


def check_lora_init(pretrained, data, args):
    """B = 0: same outputs as the frozen model; only B gets a gradient."""
    base = build('probe', args, pretrained).eval()
    lora = build(f'lora{args.merge_rank}', args, pretrained).eval()
    lora.head.load_state_dict(base.head.state_dict())
    x, m, y = next(batches(data['val'], 32, False, args.device))
    same = torch.equal(base(x, m), lora(x, m))
    F.cross_entropy(lora(x, m), y).backward()
    gA = sum(mod.A.grad.abs().sum().item() for mod in lora_modules(lora))
    gB = sum(mod.B.grad.abs().sum().item() for mod in lora_modules(lora))
    print(f'LoRA at initialisation: outputs identical to the frozen model = '
          f'{same}; sum |grad A| = {gA:.2e}, sum |grad B| = {gB:.2e}')
    del base, lora
    return {'identical_at_init': same, 'grad_A_sum': gA, 'grad_B_sum': gB}


def check_merge(model, data, args):
    """Compare logits before and after W <- W + (alpha/r) B A, in float32."""
    before = predict(model, data['test'], args, amp=False)
    t0 = time.perf_counter(); predict(model, data['test'], args, amp=False)
    t_lora = time.perf_counter() - t0
    merge_lora(model)
    after = predict(model, data['test'], args, amp=False)
    t0 = time.perf_counter(); predict(model, data['test'], args, amp=False)
    t_merged = time.perf_counter() - t0
    gap = (before - after).abs().max().item()
    same = bool((before.argmax(1) == after.argmax(1)).all())
    print(f'Merged W + (alpha/r) B A: max |logit difference| = {gap:.2e}, '
          f'identical test predictions = {same}; float32 test-set inference '
          f'{t_lora:.1f}s with the LoRA branch, {t_merged:.1f}s merged')
    return {'max_logit_diff': gap, 'same_predictions': same,
            'seconds_lora_branch': t_lora, 'seconds_merged': t_merged}


def plot(rows, zero_shot, path_curves, path_params):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    if zero_shot:
        ax.axhline(zero_shot['acc'], color='gray', ls=':', label='zero-shot (test)')
    for r, c in zip(rows, colors):
        ax.plot([p['step'] for p in r['curve']], [p['acc'] for p in r['curve']],
                '-', color=c, label=f'{r["label"]} ({r["trainable"]:,} params)')
    ax.set_xlabel('training step (batch of 32 tweets)')
    ax.set_ylabel('validation accuracy')
    ax.set_title('GPT-2 (124M) on TweetEval sentiment (3 classes)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(path_curves, dpi=110)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    if zero_shot:
        ax.axhline(zero_shot['acc'], color='gray', ls=':', label='zero-shot')
    for r, c in zip(rows, colors):
        ax.scatter(r['trainable'], r['test_acc'], color=c, zorder=3, s=40)
        ax.annotate(r['label'], (r['trainable'], r['test_acc']), fontsize=8,
                    textcoords='offset points', xytext=(6, -10))
    ax.set_xscale('log')
    ax.set_xlabel('trainable parameters (log scale)')
    ax.set_ylabel('test accuracy')
    ax.set_title('Accuracy vs trainable parameters', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(path_params, dpi=110)
    print(f'Saved {path_curves} and {path_params}')


def main(args):
    set_seed(args.seed)
    tokenizer = get_tokenizer(args.model)
    raw = {s: read_tweets(os.path.join(args.tweet_dir, f'{s}.tsv'))
           for s in ['train', 'val', 'test']}
    data = {s: encode_split(t, y, tokenizer, args.max_len)
            for s, (t, y) in raw.items()}
    for s, (t, y) in raw.items():
        print(f'{s:5s}: {len(y):6,} tweets, class counts '
              f'{np.bincount(y, minlength=3).tolist()}')

    gpt = GPT2.from_pretrained(args.model, dropout=args.dropout)
    pretrained = {'cfg': gpt.cfg, 'weights': gpt.state_dict()}
    del gpt
    report = {'init_check': check_lora_init(pretrained, data, args), 'rows': []}

    methods = ['probe'] + [f'lora{r}' for r in args.ranks] + ['full']
    for method in methods:
        set_seed(args.seed)
        model = build(method, args, pretrained)
        lr = {'probe': args.lr_probe, 'full': args.lr_full}.get(method, args.lr_lora)
        label = ('linear probe' if method == 'probe' else 'full fine-tuning'
                 if method == 'full' else f'LoRA r={method[4:]}')
        curve, secs = fit(model, data, lr, args, label)
        test = evaluate(model, data['test'], args)
        row = {'method': method, 'label': label, 'lr': lr,
               'trainable': count_parameters(model, trainable_only=True),
               'total': count_parameters(model), 'seconds': secs,
               'best_val_macro_recall': max(p['macro_recall'] for p in curve),
               'test_acc': test['acc'], 'test_macro_recall': test['macro_recall'],
               'curve': curve}
        print(f'{label:18s} trainable {row["trainable"]:>11,} | test acc '
              f'{test["acc"]:.3f} | test macro-recall {test["macro_recall"]:.3f}'
              f' | {secs:.0f}s', flush=True)
        if method == f'lora{args.merge_rank}':
            report['merge'] = check_merge(model, data, args)
        report['rows'].append(row)
        del model
        if args.device.type == 'cuda':
            torch.cuda.empty_cache()

    zero_shot = None
    if os.path.exists(args.zero_shot_results):
        with open(args.zero_shot_results) as f:
            zero_shot = json.load(f)['zero_shot_test']
    print('\n| method | trainable params | test accuracy | test macro-recall |')
    print('|---|---|---|---|')
    if zero_shot:
        print(f'| zero-shot prompt (version5) | 0 | {zero_shot["acc"]:.3f} | '
              f'{zero_shot["macro_recall"]:.3f} |')
    for r in report['rows']:
        print(f'| {r["label"]} | {r["trainable"]:,} '
              f'({100 * r["trainable"] / r["total"]:.3f}%) | '
              f'{r["test_acc"]:.3f} | {r["test_macro_recall"]:.3f} |')
    os.makedirs(os.path.dirname(args.results) or '.', exist_ok=True)
    with open(args.results, 'w') as f:
        json.dump(report, f, indent=1)
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        plot(report['rows'], zero_shot,
             os.path.join(args.plot_dir, 'lora_val_curves.png'),
             os.path.join(args.plot_dir, 'lora_params_vs_accuracy.png'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='LoRA fine-tuning of GPT-2')
    p.add_argument('--model', default='gpt2')
    p.add_argument('--tweet_dir', default='data/tweet_sentiment')
    p.add_argument('--max_len', type=int, default=64, help='tokens per tweet')
    p.add_argument('--pool', default='last', choices=['last', 'mean'])
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--eval_every', type=int, default=200)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--lr_probe', type=float, default=3e-3)
    p.add_argument('--lr_lora', type=float, default=5e-4)
    p.add_argument('--lr_full', type=float, default=3e-5)
    p.add_argument('--ranks', type=int, nargs='+', default=[1, 4, 16])
    p.add_argument('--alpha_over_rank', type=float, default=2.0,
                   help='alpha = this x rank, i.e. dW = 2 B A by default')
    p.add_argument('--merge_rank', type=int, default=4)
    p.add_argument('--amp', type=int, default=1,
                   help='bfloat16 autocast on CUDA (1) or float32 (0)')
    p.add_argument('--zero_shot_results', default='checkpoints/zero_shot.json')
    p.add_argument('--results', default='checkpoints/lora_results.json')
    p.add_argument('--plot_dir', default='static')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default=None)
    main(finish_args(p.parse_args()))
