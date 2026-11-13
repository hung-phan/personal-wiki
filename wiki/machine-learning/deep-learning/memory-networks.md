---
name: memory-networks
description: "End-to-end memory networks for question answering with runnable PyTorch code — the bAbI tasks (story, question, one-word answer; single and two supporting facts), encoding each sentence as the sum of its word embeddings, attention over sentences by dot product with the question and softmax, the answer as a softmax over the vocabulary, separate embeddings for addressing and content, two hops where hop 2 is queried by hop 1's output, why bag-of-words sentences lose order and a learned time encoding (or the course's line-number token) fixes it (0.69 vs 1.00 accuracy vs a 0.53 heuristic baseline), the course's Keras settings (embedding 15 or 30, RMSprop, ELU dense layer). Study guide for Deep Learning: Advanced NLP and RNNs, section 6."
created: 2026-11-13
updated: 2026-11-13
tags: [nlp, memory-networks, question-answering, babi, attention, pytorch, keras, study-guide, advanced-nlp-rnn]
---
# Memory Networks: Question Answering over a Story

> **How to read this note.** Study guide for *Deep Learning: Advanced NLP and RNNs*, section 6 (lectures 44–49): the bAbI tasks, memory-network theory, and Keras models for one and two supporting facts. Read [rnn-attention](rnn-attention.md) first — a memory network is attention over sentences instead of words. The course downloads bAbI; the snippet generates bAbI-style stories in code (same format, controllable), trains in about 4 seconds, and shows why the course quietly numbers its sentences. Needs `torch`.

## Why This Exists

**Problem.** Given a short story and a question — "Mary moved to the bathroom. John went to the hallway. Mary travelled to the office. *Where is Mary?*" — the answer is in one sentence, but which one depends on the question. A single RNN state over the whole story has to remember everything for every possible question.

**Key insight.** Store each sentence as its own vector (the **memory**), turn the question into a vector, and let the question **attend** over the memories: weights from dot products and a softmax, an answer from the weighted sum. For questions that need two facts, attend twice, using the first result to form the second query.

**Reach for this when:**
- Working through section 6, or reading about bAbI, end-to-end memory networks or "hops".
- You want the conceptual ancestor of retrieval-augmented generation: look things up, then answer.

**Don't reach for this when:**
- Real question answering — use retrieval plus an LLM ([retrieval-augmented-generation](../llm-engineering/retrieval-augmented-generation.md)).
- Stories need word order or reasoning beyond a few lookups — bag-of-words sentences can't represent it.

## Diagrams

![Memory network: each story sentence becomes a sum of word vectors; the question vector u dots with each sentence and a softmax gives weights (0.9 on "Mary travelled to the office"); the weighted sum gives the answer; two hops for two supporting facts; the missing order information](assets/memory-network.svg)

## 1. The bAbI tasks

Facebook's bAbI set has 20 synthetic reasoning tasks; the course uses two (10,000-example versions). Each example is a story of short sentences, a question and a one-word answer. **Task 1, single supporting fact:** "Where is Mary?" — the answer comes from the last sentence about Mary. **Task 2, two supporting facts:** "Where is the football?" — first find who has the football, then where that person went. The files number sentences within a story; the course's parser rebuilds each story up to every question.

## 2. One hop

- **Sentence vectors:** embed each word and **sum** the vectors of a sentence — a bag of words, cheap and order-free.
- **Question vector** $u$: the same, with its own embedding.
- **Weights:** $p_i = \text{softmax}_i(m_i \cdot u)$ over the story's sentences.
- **Answer:** $\text{softmax}_{\text{vocab}}\big(W(\sum_i p_i c_i + u)\big)$, where $c_i$ is a second embedding of sentence $i$ (one embedding to *find* a memory, another for what it *contributes*).

Everything is trained end to end with cross-entropy on the answer word; no one tells the model which sentence is the supporting fact.

```python
import torch, torch.nn as nn, random
random.seed(0); torch.manual_seed(0); torch.set_num_threads(1)
people, places = ["mary", "john", "sandra", "daniel"], ["kitchen", "garden", "office", "bathroom", "hallway", "bedroom"]
vocab = {w: i + 1 for i, w in enumerate(people + places + ["moved", "to", "the", "where", "is"])}   # 0 = padding
S, W = 8, 4                                               # up to 8 sentences of 4 words each
def example():                                            # bAbI-style "single supporting fact"
    story = [[random.choice(people), "moved", "to", random.choice(places)] for _ in range(random.randint(2, S))]
    who = random.choice([s[0] for s in story])
    answer = [s[3] for s in story if s[0] == who][-1]     # the LATEST place this person moved to
    pad = [[0] * W] * (S - len(story))
    return pad + [[vocab[w] for w in s] for s in story], [vocab["where"], vocab["is"], vocab[who]], vocab[answer], vocab[story[-1][3]]
def data(n): ex = [example() for _ in range(n)]; return [torch.tensor([e[i] for e in ex]) for i in range(4)]
class MemNet(nn.Module):
    def __init__(self, time):
        super().__init__(); D = 32; self.time = time
        self.A, self.C, self.B = (nn.Embedding(len(vocab) + 1, D, padding_idx=0) for _ in range(3))
        self.T = nn.Parameter(torch.zeros(S, D)); self.out = nn.Linear(D, len(vocab) + 1)
    def forward(self, story, q):
        u = self.B(q).sum(1)                              # question vector = sum of its word vectors
        m, c = self.A(story).sum(2), self.C(story).sum(2) # each sentence = sum of word vectors (no word order)
        if self.time: m = m + self.T                      # learned "how recent is this sentence" vector per slot
        p = (m @ u.unsqueeze(-1)).squeeze(-1).softmax(1)  # attention over sentences
        return self.out((p.unsqueeze(-1) * c).sum(1) + u)
Xs, Xq, Y, last = data(5000); Ts, Tq, TY, Tlast = data(1000)
print("baseline 'last place mentioned in the story':", round((Tlast == TY).float().mean().item(), 3))
for time in (False, True):
    m = MemNet(time); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for ep in range(30):
        for i in range(0, 5000, 100):
            loss = nn.functional.cross_entropy(m(Xs[i:i+100], Xq[i:i+100]), Y[i:i+100]); opt.zero_grad(); loss.backward(); opt.step()
    print(f"memory network, time encoding={time!s:5}: test accuracy {(m(Ts, Tq).argmax(1) == TY).float().mean().item():.3f}")
# Expected output:
# baseline 'last place mentioned in the story': 0.528
# memory network, time encoding=False: test accuracy 0.692
# memory network, time encoding=True : test accuracy 1.000
```

The heuristic "answer with the last place mentioned" is right only 53% of the time (when the person asked about moved last). The memory network without position information reaches only 69%: summed word vectors can't say which of Mary's two sentences came **later**. Adding a learned vector per sentence slot (a **time encoding**, as in the 2015 end-to-end memory network paper) gives 100%. The course gets the same effect more cheaply: its parser **prepends each sentence's line number as an extra word** ("3 mary travelled to the office"), so the summed vector includes a token that says *when* — and it reports 100% on task 1 within 4 epochs. Its comments note this trick only in passing; it's what makes the model work.

## 3. Two hops

For two supporting facts, the output of hop 1 becomes the query for hop 2: $u_2 = u_1 + o_1$ (or a dense layer of it). Hop 1 finds "John picked up the football"; hop 2, now asking about John, finds "John went to the garden". The course's tricks: separate story embeddings per hop, with hop 1's content embedding reused as hop 2's addressing embedding, and a dense layer after each hop — with ELU, because ReLU overfit. The number of hops is chosen by hand to match the task.

The course's settings: task 1 — embedding size 15, RMSprop 0.01, batch 32, 4 epochs; task 2 — embedding size 30, RMSprop 0.005, 30 epochs, results described only as "pretty decent". Its last lecture points to **dynamic memory networks**, which encode sentences with RNNs to keep word order.

## Trade-offs

- **Bag-of-words sentences.** Fast, few parameters vs no word order ("dog bites man" = "man bites dog").
- **Fixed number of hops.** Simple vs must match the task's reasoning depth.
- **Synthetic benchmarks.** Clean, controllable vs near-perfect scores that don't transfer to real text.

## Common Pitfalls

- **No time encoding.** The model can't tell old facts from new; add slot vectors or a line-number token.
- **One embedding for everything.** Separate addressing and content embeddings work better.
- **Too many hops.** Extra hops add noise when the task needs fewer.
- **Reading 100% on bAbI as understanding.** Generated sentences use a tiny vocabulary and fixed templates.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Answer is one fact in a story | One hop + time encoding | Single RNN state | Direct lookup |
| Chain of two facts | Two hops | One hop | Second query depends on the first |
| Word order matters | RNN/transformer sentence encoder | Summed word vectors | Bag of words loses order |
| Real documents and questions | Retrieval + LLM (RAG) | Memory network from scratch | Scale and language |

## References

- [Weston et al. — Memory Networks (2014)](https://arxiv.org/abs/1410.3916) · [Sukhbaatar et al. — End-To-End Memory Networks (2015)](https://arxiv.org/abs/1503.08895)
- [Weston et al. — Towards AI-Complete Question Answering: the bAbI tasks (2015)](https://arxiv.org/abs/1502.05698)
- [Kumar et al. — Dynamic Memory Networks (2015)](https://arxiv.org/abs/1506.07285)

## See Also

- [rnn-attention](rnn-attention.md) — attention over words, the same weighted average.
- [retrieval-augmented-generation](../llm-engineering/retrieval-augmented-generation.md) — the modern way to answer from documents.
- [text-classification-embeddings](text-classification-embeddings.md) — embeddings and the course's earlier sections.
