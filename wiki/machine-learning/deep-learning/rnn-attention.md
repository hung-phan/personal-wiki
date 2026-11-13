---
name: rnn-attention
description: "Attention for RNN encoder-decoders (Bahdanau-style) with runnable PyTorch code — why one thought vector bottlenecks long inputs, a bidirectional encoder keeping one vector per input position, alignment scores from a tiny neural network over [decoder state s_(t−1); encoder state h_j], softmax over time (not features), the context vector Σ α_j h_j concatenated with the previous word, teacher forcing with attention, seq2seq with vs without attention on 16-symbol reversal (0.52 vs 0.995 token accuracy), reading attention maps and why they're not reliable explanations, the course's Keras implementation (softmax_over_time, RepeatVector, Concatenate, Dot, shared global layers, a manual loop over output steps) and its Twitter chatbot variant, how this leads to Transformers. Study guide for Deep Learning: Advanced NLP and RNNs, section 5."
created: 2026-11-13
updated: 2026-11-13
tags: [nlp, attention, bahdanau-attention, seq2seq, encoder-decoder, machine-translation, pytorch, keras, study-guide, advanced-nlp-rnn]
---
# Attention for Seq2Seq

> **How to read this note.** Study guide for *Deep Learning: Advanced NLP and RNNs*, section 5 (lectures 35–43): attention theory, the Keras implementation for translation, visualising the weights, and a chatbot built from the same code. Read [seq2seq-bidirectional-rnns](seq2seq-bidirectional-rnns.md) first. The snippet uses the same toy reversal task, made long enough (16 symbols) that a single thought vector struggles, and trains an encoder-decoder with and without attention side by side (about 30 seconds on a CPU). Transformers take this idea further: [transformer-forward-pass](../math-for-ml/transformer-forward-pass.md).

## Why This Exists

**Problem.** Plain seq2seq squeezes the entire input into one fixed-size vector before decoding starts. For a short sentence that's fine; for a long one, early words have faded by the time the encoder finishes, and the decoder has no way to look back.

**Key insight.** Keep the encoder's vector for **every** input position, and at each output step let the decoder compute a weighted average of them — weights chosen by how relevant each position is right now. The decoder "looks at" different input words as it writes different output words.

**Reach for this when:**
- An encoder-decoder's quality drops with input length.
- You want the conceptual bridge from RNN seq2seq to Transformers.
- Reading papers or code that mention alignment, context vectors or additive attention.

**Don't reach for this when:**
- Building a production translator or chatbot — use a pretrained transformer.
- You want to *explain* a model's decision from its attention weights — they're suggestive at best (see Pitfalls).

## Diagrams

![Attention: a bidirectional encoder gives one vector per input word; a tiny network scores each against the previous decoder state; softmax over input positions gives weights α; the context vector Σ α_j h_j feeds the decoder step along with the previous word](assets/attention.svg)

## 1. The mechanism

At output step $t$, with previous decoder state $s_{t-1}$ and encoder states $h_1, \dots, h_{T_x}$:

1. **Score** every input position with a small network shared across positions (the course: a 10-unit tanh layer, then one output):
   $$e_{t,j} = v^\top \tanh\big(W [s_{t-1}; h_j]\big).$$
2. **Normalise over time**: $\alpha_{t,j} = \text{softmax}_j(e_{t,j})$ — the weights sum to 1 across input positions. (Keras's default softmax works on the last axis, the features; the course writes a custom `softmax_over_time` for axis 1.)
3. **Context**: $c_t = \sum_j \alpha_{t,j}\, h_j$.
4. **Decode**: the decoder LSTM's input is $[\text{previous word embedding}; c_t]$; with teacher forcing the previous word is the true one.

The encoder is bidirectional, so each $h_j$ describes word $j$ *in context* (size $2M$). The decoder starts from zeros instead of the encoder's final state — the context vectors carry the information. Cost: $T_x$ scores at each of $T_y$ steps.

## 2. Does it help? Seq2seq with and without attention

```python
import torch, torch.nn as nn
torch.set_num_threads(1)
V, L, H, SOS = 12, 16, 64, 0                              # reverse a 16-symbol sequence: too much for one thought vector?
def batch(n): x = torch.randint(1, V, (n, L)); return x, x.flip(1)
class Model(nn.Module):
    def __init__(self, attention):
        super().__init__(); self.att = attention
        self.emb = nn.Embedding(V, 16); self.enc = nn.GRU(16, H, batch_first=True, bidirectional=True)
        self.dec = nn.GRUCell(16 + 2 * H, H); self.out = nn.Linear(H, V)
        self.score = nn.Sequential(nn.Linear(H + 2 * H, 10), nn.Tanh(), nn.Linear(10, 1))   # the course's tiny scoring net
    def forward(self, x, y=None):
        hs, _ = self.enc(self.emb(x))                         # N x L x 2H: one vector per input position
        s, prev, logits, alphas = torch.zeros(len(x), H), torch.full((len(x),), SOS), [], []
        for t in range(L):
            if self.att:                                      # score every input position against the decoder state
                e = self.score(torch.cat([s.unsqueeze(1).expand(-1, L, -1), hs], -1)).squeeze(-1)
                a = e.softmax(1); ctx = (a.unsqueeze(-1) * hs).sum(1); alphas.append(a)   # softmax over time -> context
            else:
                ctx = hs[:, -1]                               # no attention: the same summary at every step
            s = self.dec(torch.cat([self.emb(prev), ctx], -1), s); logits.append(self.out(s))
            prev = y[:, t] if y is not None else logits[-1].argmax(-1)   # teacher forcing vs own predictions
        return torch.stack(logits, 1), (torch.stack(alphas, 1) if alphas else None)
for att in (False, True):
    torch.manual_seed(0); m = Model(att); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for step in range(1500):
        x, y = batch(64); loss = nn.functional.cross_entropy(m(x, y)[0].reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    x, y = batch(500); logits, A = m(x); acc = (logits.argmax(-1) == y).float().mean().item()
    print(f"attention={att!s:5}  tokens right: {acc:.3f}")
print("output step t looks mostly at input position:", A[0].argmax(1).tolist())
# Expected output:
# attention=False  tokens right: 0.516
# attention=True   tokens right: 0.995
# output step t looks mostly at input position: [15, 15, 13, 12, 11, 11, 10, 9, 8, 7, 5, 5, 4, 3, 2, 0]
```

Same encoder, decoder, data and training budget: the model without attention (which sees only the encoder's last state at every step) gets about half the tokens right; with attention, 99.5%. The last line shows where the attention points for one example: output step 1 looks at input position 15, step 2 at 15, then 13, 12, 11 … 0 — roughly the reversal. Not exactly, because a bidirectional encoder state at position $j$ also knows about its neighbours, so attending to $j \pm 1$ can work just as well. That's also why attention maps are a hint, not an explanation.

## 3. The course's Keras implementation

The course reuses the translation pipeline from section 4 (manythings.org Spanish–English, 20,000 pairs, vocabulary 20,000, GloVe 100-d, encoder and decoder latent size 400, batch 64, 30 epochs, 20% validation). Keras has no loop over decoder steps that also recomputes attention, so it unrolls one by hand:

- **Layers defined once, globally**, then called at every step — calling the same layer object shares its weights.
- **`RepeatVector(Tx)`** copies $s_{t-1}$ to every input position; **`Concatenate`** joins it with the encoder outputs; two `Dense` layers score; **`Dot(axes=1)`** forms the weighted sum.
- At step $t$, a fresh `Lambda` slices out the teacher-forcing word $t$; it's concatenated with the context and fed to `LSTM(400, return_state=True)` with `initial_state=[s, c]`.
- The $T_y$ outputs are stacked and transposed back to batch × time × vocabulary.
- **Inference**: a separate one-step model takes the encoder outputs, the previous word and $(s, c)$, starting from zeros.

Results are described only as slightly better than plain seq2seq and still overfitting. The attention heatmap ($T_y \times T_x$, mostly diagonal for translation) is shown, with the plotting code left as an exercise. Swapping in Twitter question/answer pairs (`convert_twitter.py`) turns the same code into a chatbot — fluent-looking replies with little sense, which is what to expect from 20,000 pairs.

**Since then.** Keras ships attention layers (`Attention`, `AdditiveAttention`, `MultiHeadAttention`); in PyTorch it's ten lines, as above. **Transformers** (2017) keep the weighted-average idea, drop the RNN entirely, and let every position attend to every other — scaled dot-product scores instead of a small network, many heads in parallel. The course doesn't mention them.

## Trade-offs

- **Attention vs a single thought vector.** Handles long inputs, gives alignments vs $T_x \times T_y$ extra computation and more code.
- **Additive (small-network) vs dot-product scores.** Works with different encoder/decoder sizes vs faster, the Transformer choice (needs scaling).
- **RNN + attention vs Transformer.** Smaller, fine for short sequences vs parallel training and far better at scale.

## Common Pitfalls

- **Softmax over the wrong axis.** Weights must sum to 1 over input positions, not over features; check the shape.
- **Creating layers inside the step loop.** Each call then has its own weights; define layers once and reuse.
- **Reading attention maps as explanations.** Contextual encoder states blur positions; different weights can give the same output.
- **Training-inference mismatch.** Teacher forcing still applies; the one-step inference model must recompute attention every step.
- **Expecting a chatbot from 20k pairs.** Fluent nonsense; real assistants are pretrained LLMs.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Encoder-decoder, inputs > ~20 tokens | Attention | Bigger thought vector | Bottleneck |
| Learning how attention works | RNN + additive attention (§2) | Reading Transformer code first | Smaller pieces |
| Anything production | Pretrained transformer | RNN + attention from scratch | Quality |
| Want an alignment picture | Plot α as Ty × Tx | Trusting it as proof | Weights are hints |

## References

- [Bahdanau, Cho & Bengio — Neural Machine Translation by Jointly Learning to Align and Translate (2014)](https://arxiv.org/abs/1409.0473)
- [Luong et al. — Effective Approaches to Attention-based NMT (2015)](https://arxiv.org/abs/1508.04025)
- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [Jain & Wallace — Attention is not Explanation (2019)](https://arxiv.org/abs/1902.10186)
- [Keras — AdditiveAttention](https://keras.io/api/layers/attention_layers/additive_attention/)

## See Also

- [seq2seq-bidirectional-rnns](seq2seq-bidirectional-rnns.md) — the encoder-decoder this extends.
- [memory-networks](memory-networks.md) — section 6: the same weighted-average idea over sentences of a story.
- [transformer-forward-pass](../math-for-ml/transformer-forward-pass.md) — self-attention, step by step.
- [huggingface-open-models](../llm-engineering/huggingface-open-models.md) — using pretrained transformers.
