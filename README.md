[English](README.md) | [简体中文](README.zh.md)

# Transformer Machine Translation in PyTorch

An educational English-to-Chinese translation project with a handwritten Transformer encoder–decoder, SentencePiece tokenization, training utilities, and beam search inference.

The repository contains source code and tokenizer models. Parallel corpora and trained translation weights are not included. No reproducible benchmark result or automated test suite is provided.

## What the project covers

- Multi-head attention, sinusoidal positional encoding, residual connections, and feed-forward layers.
- Padding and causal masks, teacher-forced training, and token-level cross-entropy.
- Adam with a Noam learning-rate schedule and a CUDA-oriented parallel loss helper.
- Beam search decoding and Chinese-tokenized SacreBLEU validation.
- Interactive English sentence translation.

Defaults in `config.py`: 6 encoder/decoder layers, model dimension 512, 8 heads, feed-forward dimension 2048, and 32,000-token source and target vocabularies. These are configuration values, not a quality claim.

## Environment

The dependency file records Python 3.10 and historical PyTorch/CUDA combinations in comments; PyTorch itself is installed separately. Choose a PyTorch build suited to your hardware, then install the pinned supporting packages:

```bash
python -m venv .venv
# Activate .venv in your shell, then install PyTorch separately.
python -m pip install -r requirements.txt
```

Supporting dependencies are NumPy, SentencePiece, SacreBLEU, and tqdm. Historical hardware notes are not a tested compatibility matrix. Run application commands from the repository root.

## Prepare data and configuration

Create all three UTF-8 JSON files before training. `main.py` constructs the test dataset even though its test call is disabled:

```text
data/json/train.json
data/json/dev.json
data/json/test.json
```

Each file is a JSON array of `[English source, Chinese target]` pairs:

```json
[
  ["Hello world.", "你好，世界。"],
  ["How are you?", "你好吗？"]
]
```

Use corpora you are entitled to use. No corpus download procedure is supplied. `data/get_corpus.py` extracts line-based text from existing JSON splits; it does not create the splits. Corpus analysis and Unicode preview scripts are optional inspection tools.

The application loads `tokenizer/eng.model` and `tokenizer/chn.model`. Retraining tokenizers requires consistent vocabulary sizes and PAD/BOS/EOS IDs in `config.py`, and retraining translation weights with those tokenizers. `tokenizer/tokenize.py` expects `data/corpus.en` and `data/corpus.ch` and uses paths relative to `tokenizer/`.

Review device settings before running. Defaults select CUDA device 0 (`gpu_id = '0'`, `device_id = [0]`), batch size 32, and 3 epochs. `main.py` and `translate.py` also set `CUDA_VISIBLE_DEVICES` to `0`. Training uses `MultiGPULossCompute`, whose replication/scatter operations require CUDA; setting `gpu_id` to an empty string alone does not provide CPU training. CPU inference also needs compatible checkpoint loading and device configuration.

## Train and translate

After preparing data, dependencies, and devices:

```bash
python main.py
```

Training computes development loss and BLEU each epoch. Best and final state dictionaries are saved under `run/train/exp/weights/`, or `exp1`, `exp2`, etc. when earlier experiment directories exist.

Set `config.test_model_path` to an actual checkpoint matching the architecture and tokenizers, then run:

```bash
python translate.py
```

Enter English sentences at the prompt; interrupt the terminal to stop. The default `best_bleu_26.30.pth` path is an example filename, not a bundled checkpoint or independently verified score. The original documentation reports a roughly 376 MB translation checkpoint, excluded from Git.

## Repository guide

```text
main.py                Training and development evaluation
translate.py           Interactive translation
config.py              Architecture, paths, checkpoints, devices
beam_decoder.py        Beam search
model/tf_model.py      Transformer implementation
model/train_utils.py   Parallel loss and Noam optimizer
tools/                 Dataset, masks, tokenizers, experiment folders
tokenizer/             SentencePiece models and vocabularies
data/                  Corpus tools and local JSON splits
weights/               Local checkpoint placeholder
```

## Evaluation limits and provenance

Development evaluation uses `sacrebleu.corpus_bleu(..., tokenize='zh')`. The disabled `test()` helper calls `evaluate()` with an extra argument and needs correction before test-set use. There is no supported test CLI or automated test suite. Training and inference were not executed for this documentation update.

Original training data and teaching materials are excluded; their rights remain with their respective authors/providers. No license file is declared in this repository, so it provides no explicit open-source license grant.
