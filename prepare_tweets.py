"""
Download the TweetEval "sentiment" task and write it as
data/tweet_sentiment/{train,val,test}.tsv, one "label<TAB>tweet" per line
(0 = negative, 1 = neutral, 2 = positive).

Source: TweetEval (Barbieri et al., 2020), https://github.com/cardiffnlp/tweeteval
(datasets/sentiment), which repackages SemEval-2017 Task 4, subtask A
(Rosenthal et al., 2017). User names are already anonymised as "@user".

The repository ships the result. TweetEval's full training set has 45,615
tweets (about 5 MB of text), so to keep the repository small the shipped
train.tsv is a random subset (seed 0, 30,000 tweets); validation (2,000) and
test (12,284) are complete. To train on everything:

    python prepare_tweets.py --n_train 0
"""

import argparse
import os
import random
import urllib.request

URL = ('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/'
       'datasets/sentiment/{split}_{kind}.txt')


def download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for kind in ['text', 'labels']:
            path = os.path.join(raw_dir, f'{split}_{kind}.txt')
            if not os.path.exists(path):
                url = URL.format(split=split, kind=kind)
                print(f'downloading {url}')
                urllib.request.urlretrieve(url, path)


def read(raw_dir, split):
    with open(os.path.join(raw_dir, f'{split}_text.txt'), encoding='utf-8') as f:
        texts = [t.rstrip('\n') for t in f]
    with open(os.path.join(raw_dir, f'{split}_labels.txt')) as f:
        labels = [int(l) for l in f if l.strip()]
    assert len(texts) == len(labels), split
    # collapse whitespace (a few tweets contain tabs, which would break the TSV)
    return [(l, ' '.join(t.split())) for t, l in zip(texts, labels)]


def main(args):
    download(args.raw_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        rows = read(args.raw_dir, split)
        n_all = len(rows)
        if split == 'train' and 0 < args.n_train < n_all:
            rows = random.Random(args.seed).sample(rows, args.n_train)
        path = os.path.join(args.out_dir, f'{split}.tsv')
        with open(path, 'w', encoding='utf-8') as f:
            for label, text in rows:
                f.write(f'{label}\t{text}\n')
        counts = [sum(1 for l, _ in rows if l == k) for k in range(3)]
        print(f'{path}: {len(rows):,} of {n_all:,} tweets, class counts '
              f'{counts}, {os.path.getsize(path) / 1e6:.2f} MB')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', default='data/tweeteval_raw')
    p.add_argument('--out_dir', default='data/tweet_sentiment')
    p.add_argument('--n_train', type=int, default=30000,
                   help='random subset of the training tweets (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
