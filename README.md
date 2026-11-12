# GPT from Scratch

Code for a decoder-only transformer (GPT) language model written from scratch in PyTorch: a character-level model, then a byte-pair-encoding (BPE) tokenizer, then a key-value cache for fast generation.

---

## Theory

Lecture notes for this topic are coming soon on https://parag1604.github.io/notes

## Python Environment Setup

Use python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- PyTorch - models and training (install the hardware accelerated build if possible; the code picks CUDA, then Apple MPS, then CPU)
- Numpy - the data pipeline and the BPE tokenizer
- tqdm - progress bar while learning the BPE merges

## Data

All scripts train on tinyshakespeare (about 1.1M characters of Shakespeare), from Andrej Karpathy's char-rnn repository. Download it into `./data`:

```bash
mkdir -p data
curl -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Other text files work too (`--data_path`). `version1.py` caches the learned BPE merges and the tokenized corpus in `./preprocessed/*.pkl`; delete that folder after changing the data or `--vocab_size`. Model weights are written to `./models/`.

## Running

Every script accepts the model and training sizes as arguments (`python versionX.py --help`). A script trains when there is no checkpoint yet (or when `--train` is passed), and then prints a sample that starts with `--prompt`. Quick check that everything works (well under a minute on a CPU):

```bash
python version0.py --max_iters 50 --eval_interval 25 --eval_iters 5 --n_layers 2 --n_embed 64 --n_head 4 --block_size 64
```

The default settings are the ones Parag used; they need a GPU (version0: 6 layers, 384 dimensions, 4000 iterations; version1: 8 layers, 512 dimensions, 10000 iterations).

### 0. Character-level GPT

```bash
python version0.py
```

Every character is a token (65 of them in tinyshakespeare). The file builds the whole model from scratch: token and position embeddings, single heads of causal self-attention (a lower-triangular mask stops a position from looking at the future), multi-head attention as a list of heads, the feed-forward network and pre-LayerNorm residual blocks. Training is next-character prediction with cross-entropy; generation samples one character at a time and feeds it back in. Train and validation losses are printed every `--eval_interval` iterations; weights go to `models/gpt_char.pth`.

### 1. BPE tokenizer and batched multi-head attention

```bash
python version1.py
```

New: a byte-pair-encoding tokenizer written from scratch. Starting from characters, it repeatedly merges the most frequent adjacent pair of tokens into a new token until the vocabulary has `--vocab_size` (2048) tokens; it prints the last tokens it learned. The same text then needs roughly a third as many tokens, so the 128-token context covers much more Shakespeare. New text is encoded by replaying the merges in the order they were learned. Also new: all attention heads are computed in one batched matrix multiplication (heads folded into a batch dimension) instead of a Python loop over heads. Weights go to `models/gpt_bpe.pth`.

### 2. KV cache

```bash
python version2.py
```

New: a key-value cache. Causal attention never changes the keys and values of earlier positions, so instead of re-running the model on the whole context for every new token, the script runs the prompt once (prefill) and then pushes only the newest token through the network, attending to the cached keys and values (decode). Because the model uses learned absolute position embeddings, the cache is rebuilt from the last `block_size // 2` tokens when it is full. The script loads the weights trained by `version1.py`, checks that greedy decoding with and without the cache produces identical tokens, and prints both timings. With the default model size (random weights, Apple M-series GPU) 300 tokens took 36 s without the cache and 12 s with it.
