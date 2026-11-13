---
name: seq2seq-bidirectional-rnns
description: "Bidirectional RNNs and sequence-to-sequence (encoder-decoder) models with runnable Keras and PyTorch code — reading forward and backward and concatenating to T x 2M, when bidirectional is allowed (never for forecasting), a BiLSTM reading an image's rows and columns (digits 0.88 vs 0.96 vs a 0.13 baseline), the encoder's final state as a \"thought vector\", teacher forcing (decoder input shifted right by one), separate training and one-step inference models, greedy vs sampled decoding loops, a seq2seq that learns to reverse sequences (vs a copy baseline), word-level translation on the manythings.org Spanish-English pairs, poetry generation as a decoder without an encoder, exposure bias. Study guide for Deep Learning: Advanced NLP and RNNs, sections 3–4."
created: 2026-11-13
updated: 2026-11-13
tags: [nlp, seq2seq, encoder-decoder, bidirectional-rnn, teacher-forcing, machine-translation, lstm, keras, pytorch, study-guide, advanced-nlp-rnn]
---
# Bidirectional RNNs and Seq2Seq

> **How to read this note.** Study guide for *Deep Learning: Advanced NLP and RNNs*, section 3 (bidirectional RNNs) and section 4 (sequence-to-sequence: translation and poetry generation). Read [text-classification-embeddings](text-classification-embeddings.md) first. The course's translation data (Spanish–English sentence pairs from manythings.org) and GloVe vectors need downloads, so the seq2seq snippet uses a toy task with a known answer — reversing a sequence — which exercises exactly the same machinery. The course's settings, checked against its public repo, are listed alongside. Next: [rnn-attention](rnn-attention.md).

## Why This Exists

**Problem.** Classification maps a sequence to a label. Translation, summarisation and chat map a sequence to **another sequence of a different length**, and the network has to produce it one token at a time — which needs two different models (one for training, one for generating) and a trick to train them at all.

**Key insight.** Split the job: an **encoder** reads the whole input into a state, and a **decoder** — a language model started from that state — writes the output one token at a time, each step fed the previous token. During training, feed it the **true** previous token (teacher forcing); at inference, feed its own guesses in a loop.

**Reach for this when:**
- Working through sections 3–4 of the course, or any classic seq2seq tutorial.
- You need to understand what "encoder", "decoder", "teacher forcing" and "autoregressive" mean before reading transformer code.
- Building a small model for a structured sequence task (reversal, transliteration, date formats).

**Don't reach for this when:**
- Real translation or chat — pretrained transformers (or an LLM API) are far better.
- Inputs are long — a single thought vector forgets; use attention ([rnn-attention](rnn-attention.md)).

## Diagrams

![Seq2seq: an encoder LSTM reads "how are you" into a thought vector (final h and c); a decoder LSTM starts from it and predicts "cómo estás <eos>"; during training its inputs are the true previous words (teacher forcing), at inference its own predictions fed back in a loop; a bidirectional encoder reads both directions](assets/seq2seq.svg)

## 1. Bidirectional RNNs

**The question.** In "the bank of the river", knowing "river" helps interpret "bank" — but a left-to-right RNN reaches "bank" before it sees "river". What if it could read both ways?

Run one RNN forward and another backward over the same sequence and concatenate their states at each step: T × M becomes T × 2M. In Keras, wrap the layer: `Bidirectional(LSTM(M, return_sequences=True))`. For a many-to-one task, the summary is either [last forward state; first backward state] or — the course's preference — a max pool over all steps.

**Only when the whole input is available.** Fine for a sentence you're classifying or translating. Never for forecasting (tomorrow's price, the next word): the backward pass would read the future.

**An image as a sequence** (the course's MNIST exercise): treat each row of pixels as one time step. One BiLSTM reads rows; a second reads columns (transpose the image first); pool each, concatenate, classify. Here on scikit-learn's 8×8 digits:

```python
import os; os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np, keras
from sklearn.datasets import load_digits                  # 1,797 8x8 handwritten digits, bundled with sklearn
keras.utils.set_random_seed(0)
X, y = load_digits(return_X_y=True); X = (X / 16).reshape(-1, 8, 8).astype("float32")
idx = np.random.default_rng(0).permutation(len(X)); tr, te = idx[:1400], idx[1400:]
def build(both):
    inp = keras.Input((8, 8))                              # an image as a sequence: 8 rows of 8 pixels
    rows = keras.layers.GlobalMaxPooling1D()(keras.layers.Bidirectional(keras.layers.LSTM(15, return_sequences=True))(inp))
    feats = rows
    if both:                                               # second branch reads columns: transpose the image first
        cols_in = keras.layers.Permute((2, 1))(inp)
        cols = keras.layers.GlobalMaxPooling1D()(keras.layers.Bidirectional(keras.layers.LSTM(15, return_sequences=True))(cols_in))
        feats = keras.layers.Concatenate()([rows, cols])
    m = keras.Model(inp, keras.layers.Dense(10, activation="softmax")(feats))
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(X[tr], y[tr], epochs=20, batch_size=32, verbose=0)
    return m.evaluate(X[te], y[te], verbose=0)[1]
print("BiLSTM over rows:          ", round(build(False), 3))
print("BiLSTM over rows + columns:", round(build(True), 3))
print("most frequent class:       ", round(np.bincount(y[te]).max() / len(te), 3))
# Expected output:
# BiLSTM over rows:           0.879
# BiLSTM over rows + columns: 0.962
# most frequent class:        0.128
```

Reading only rows gets 88%; adding the column branch gets 96% (most-frequent-class baseline: 13%). Two directions of reading give two different views of the same input. (The course: 28×28 MNIST, 15 units per direction, 10 epochs, sparse categorical cross-entropy, Adam.)

## 2. Encoder, decoder, thought vector

**Encoder:** an LSTM reads the input; its per-step outputs are thrown away and only the final $(h, c)$ is kept — the **thought vector**, the whole input squeezed into a fixed size. **Decoder:** a second LSTM whose initial state *is* that vector, trained as a language model of the output: given the previous token, predict the next. A softmax over the output vocabulary at each step.

**Teacher forcing.** Training inputs to the decoder are the target shifted right by one:

| | step 1 | step 2 | step 3 |
|---|---|---|---|
| decoder input | `<sos>` | cómo | estás |
| decoder target | cómo | estás | `<eos>` |

Without the shift the model would learn to copy its input. Feeding the **true** previous word (not the model's guess) makes training fast and stable — like a teacher correcting each mistake immediately.

**Inference is a loop.** There's no target to feed, so: start with `<sos>` and the encoder's state; predict a token; feed it back as the next input; stop at `<eos>` or a maximum length. In Keras this needs a **separate one-step decoder model** sharing the trained layers, with the states as explicit inputs and outputs. Pick each token by **argmax** (greedy — translation) or by **sampling** (variety — poetry).

```python
import torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(1)
V, L, H, SOS = 12, 6, 64, 0                              # 11 symbols + start token; task: reverse a 6-symbol sequence
def batch(n): x = torch.randint(1, V, (n, L)); return x, x.flip(1)
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 16); self.enc = nn.GRU(16, H, batch_first=True)
        self.dec = nn.GRU(16, H, batch_first=True); self.out = nn.Linear(H, V)
    def forward(self, x, y=None):
        _, h = self.enc(self.emb(x))                        # h = the "thought vector": the whole input in H numbers
        if y is not None:                                   # training: teacher forcing, feed the true previous token
            prev = torch.cat([torch.full((len(x), 1), SOS), y[:, :-1]], 1)
            return self.out(self.dec(self.emb(prev), h)[0])
        tok, outs = torch.full((len(x), 1), SOS), []        # inference: feed back our own prediction, one step at a time
        for _ in range(L):
            o, h = self.dec(self.emb(tok), h); tok = self.out(o).argmax(-1); outs.append(tok)
        return torch.cat(outs, 1)
m = Seq2Seq(); opt = torch.optim.Adam(m.parameters(), 3e-3)
for step in range(1500):
    x, y = batch(64); loss = nn.functional.cross_entropy(m(x, y).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
x, y = batch(1000); pred = m(x)
print("whole sequence right:", round((pred == y).all(1).float().mean().item(), 3), "  tokens right:", round((pred == y).float().mean().item(), 3))
print("baseline (copy the input unreversed), tokens right:", round((x == y).float().mean().item(), 3))
print("input", x[0].tolist(), "-> output", pred[0].tolist())
# Expected output:
# whole sequence right: 0.941   tokens right: 0.985
# baseline (copy the input unreversed), tokens right: 0.09
# input [2, 7, 11, 4, 4, 9] -> output [9, 4, 4, 11, 7, 2]
```

94% of 6-symbol sequences come out exactly reversed (98.5% of tokens); copying the input gets 9% of tokens right. The same code is a translator once the vocabularies are words and the pairs are sentences.

**Exposure bias.** Teacher forcing means the decoder never practises recovering from its own mistakes; at inference one wrong token can derail the rest. Remedies: scheduled sampling (mix in its own predictions during training), beam search at inference.

## 3. The course's translation and poetry models

**Translation (`wseq2seq.py`).** Tab-separated English–Spanish pairs from manythings.org/anki, keeping the first 10,000 lines (the shortest sentences). Two tokenizers (input and output), GloVe 100-d on the encoder side only, LSTM latent size 256, batch 64, 40 epochs, 20% validation, greedy decoding. It overfits (dropout didn't help), and the course judges outputs by eye — no BLEU score.

**Poetry (`poetry.py`).** A language model of Robert Frost's poems: the decoder alone, with zero initial states as explicit inputs so the same model can be reused for sampling. Vocabulary 3,000, 50-d embeddings, LSTM 25 units, Adam 0.01, 2,000 epochs; next-word accuracy around 50%. Generation samples each word (not argmax) so poems vary. This is the same autoregressive loop LLMs use ([llm-foundations](../llm-engineering/llm-foundations.md)).

**Two Keras workarounds that are now unnecessary:** the course one-hot encodes the target sequences because it believed sparse cross-entropy couldn't handle a sequence of targets (it can, in current Keras), and wraps slicing in `Lambda` layers to satisfy TF1-era graph rules.

## Trade-offs

- **Bidirectional vs unidirectional.** Context from both sides vs twice the compute and unusable for online prediction.
- **Teacher forcing.** Fast, stable training vs exposure bias at inference.
- **Greedy vs sampling vs beam search.** Deterministic and cheap vs varied vs better sequences at k× the cost.
- **One thought vector.** Simple vs an information bottleneck that grows with input length (the reason for attention).

## Common Pitfalls

- **Bidirectional models for forecasting.** Leaks the future; scores look great and production fails.
- **Forgetting the shift.** Decoder inputs must be targets shifted right with `<sos>`; otherwise it learns to copy.
- **Reusing the training model for generation.** It expects the whole target; build a one-step decoder that carries states.
- **Post-padding decoder targets without masking.** Padding tokens get trained on and dominate accuracy; mask them in the loss.
- **Judging translation by a few examples.** Use held-out pairs and a metric (BLEU, chrF) or human rating.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Classify/tag a complete sentence | Bidirectional LSTM/GRU | Forward-only RNN | Both-side context |
| Predict the future | Forward-only model | Bidirectional | No future data at prediction time |
| Output is a sequence of new length | Encoder-decoder | Many-to-one RNN | Different lengths |
| Inputs longer than ~20 tokens | Add attention | Bigger thought vector | Bottleneck |
| Creative generation | Sampling (temperature) | Argmax | Variety |
| Real translation | Pretrained transformer | Training seq2seq from scratch | Quality, data |

## References

- [Sutskever et al. — Sequence to Sequence Learning with Neural Networks (2014)](https://arxiv.org/abs/1409.3215) · [Cho et al. — RNN Encoder–Decoder (2014)](https://arxiv.org/abs/1406.1078)
- Schuster & Paliwal — Bidirectional Recurrent Neural Networks, *IEEE Trans. Signal Processing* 45(11), 1997.
- [Bengio et al. — Scheduled Sampling (2015)](https://arxiv.org/abs/1506.03099)
- [manythings.org — Tab-delimited bilingual sentence pairs](https://www.manythings.org/anki/)
- [Keras — Bidirectional layer](https://keras.io/api/layers/recurrent_layers/bidirectional/)

## See Also

- [text-classification-embeddings](text-classification-embeddings.md) — embeddings, LSTM return shapes, task shapes.
- [rnn-attention](rnn-attention.md) — section 5: removing the thought-vector bottleneck.
- [recurrent-neural-networks](recurrent-neural-networks.md) — LSTM cell and vanishing gradients.
- [transformer-forward-pass](../math-for-ml/transformer-forward-pass.md) — the decoder that replaced RNN seq2seq.
