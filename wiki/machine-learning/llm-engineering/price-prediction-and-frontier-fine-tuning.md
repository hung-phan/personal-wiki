---
name: price-prediction-and-frontier-fine-tuning
description: "The \"Price is Right\" capstone, part 1 — predicting a product's price from its description, with runnable code on a synthetic catalogue: generalization vs overfitting, training-time vs inference-time improvement, data sources and curation (Amazon Reviews 2023 metadata, filters, dedup to avoid leakage, weighted sampling by price), LLM rewriting of messy descriptions with batch APIs, train/validation/test splits, mean absolute error and R² as business-readable metrics, the baseline ladder (random, constant mean, linear regression on hand features, bag of words, random forest, gradient boosting, log-price target), neural nets on hashed text features, zero-shot frontier models, the evaluation harness (green/yellow/red scatter, running error with confidence band), and supervised fine-tuning of a frontier model with JSONL chat examples — and why it made results worse. Study guide for the LLM Engineering course, Week 6."
created: 2026-11-13
updated: 2026-11-13
tags: [llm, llm-engineering, fine-tuning, baselines, regression, data-curation, evaluation, xgboost, study-guide, llm-engineering-course]
---
# Price Prediction: Baselines, Data, and Fine-Tuning a Frontier Model

> **How to read this note.** Study guide for Week 6 of *AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents* (lectures 138–164), the first of three capstone weeks. The task: given a product's description, predict its price. The course curates ~820k products from the Amazon Reviews 2023 metadata, climbs a ladder of baselines, tries frontier models zero-shot, and fine-tunes one. Here a synthetic catalogue with the same ingredients (category, weight, descriptive words, skewed prices) makes every step runnable in seconds. Install: `VIRTUAL_ENV=/Users/phahng/Desktop/workarea/.venv uv pip install scikit-learn torch matplotlib`. Week 7 continues in [qlora-fine-tuning](qlora-fine-tuning.md), Week 8 in [agentic-ai-systems](agentic-ai-systems.md).

## Why This Exists

**Problem.** "Use an LLM" is tempting for any prediction task. Without baselines you can't tell whether a frontier model at $45 error is impressive or worse than a random forest; without careful data curation and splits, the numbers lie; and "fine-tune the frontier model" can make things worse.

**Key insight.** Treat an LLM as one more model on a **leaderboard of baselines**, all scored on the same held-out items with a metric the business understands. Most of the gain comes from the data; fine-tuning a frontier model only helps when the task needs something a prompt can't express.

**Reach for this when:**
- Deciding whether an LLM is worth it for a prediction task.
- Building a dataset for fine-tuning or evaluation.
- A fine-tuned model scores worse than the base model and you want to know why.

**Don't reach for this when:**
- You're fine-tuning an open-weight model with LoRA — [qlora-fine-tuning](qlora-fine-tuning.md).
- Your data is a clean table with a few numeric columns — gradient boosting, done; see [artificial-neural-networks](../deep-learning/artificial-neural-networks.md) §5 for the classical toolkit.

## Diagrams

![Price-prediction leaderboard from the course: mean absolute error from random guess $382 and constant $106 through classical ML ($68–102), neural nets ($64, $46), zero-shot frontier models ($63 down to GPT-5.1 at $45), the fine-tuned frontier model ($76, worse than zero-shot), the QLoRA Llama 3B ($40) and RAG plus ensemble ($30)](assets/price-leaderboard.svg)

## 1. The project and the central idea

**Generalization** — doing well on data the model has never seen — is the whole game; **overfitting** (memorising the training set) is the enemy. Weeks 1–5 improved models at **inference time** (prompts, tools, RAG); Weeks 6–7 improve them at **training time** (changing weights). They combine — Week 8 does both.

The business problem: estimate a fair price from a product description. Metric: **mean absolute error in dollars** — "on average we're off by $X" is something anyone understands, unlike an MSE of 10,000. Also report **R²** (1 = perfect, 0 = no better than always guessing the mean, negative = worse than that).

Ways to follow along in the course: watch only; a "lite" dataset (20k items, a few dollars of API cost); or the full dataset (800k items, ~$30 mostly for preprocessing).

## 2. Data: sources, curation, leakage

**Where data comes from:** your own proprietary data (the usual moat), Kaggle, the Hugging Face Hub, synthetic data from an LLM, or data vendors. The course uses product **metadata** (title, description, details, price) from McAuley Lab's *Amazon Reviews 2023* — not the reviews.

**Filters** turn raw scrape into usable items: price between $0.50 and $999.49; description at least 600 characters; strip long part numbers; convert weight to pounds; truncate text to ~4,000 characters. About 2.9M items survive, ~2.89M after **de-duplicating** on title and description. Duplicates matter: an item in both train and test makes validation look great and real performance worse (**leakage**).

**Curation by weighted sampling.** The raw catalogue is dominated by cheap items and by one category (automotive). The course samples 820k items with probability proportional to price² and down-weights automotive, giving a more balanced, more interesting price range. The weights are trial and error, judged by how well models train on the result. Same idea on the synthetic catalogue:

```python
import numpy as np
rng = np.random.default_rng(0)                            # synthetic Amazon-style catalogue
CATS = {"automotive": 40, "electronics": 90, "appliances": 120, "toys": 25, "tools": 60}   # typical price per category
WORDS = {"professional": 1.8, "stainless": 1.4, "wireless": 1.5, "kit": 1.3, "replacement": 0.6, "mini": 0.7,
         "heavy-duty": 1.6, "smart": 1.7, "pack": 0.9, "premium": 1.5}                      # price multipliers
FILLER = "durable easy install black compact design fits most models quality".split()
def make(n):
    cat = rng.choice(list(CATS), n); weight = rng.gamma(2.0, 2.0, n)                         # pounds
    rows, prices = [], []
    for c, w in zip(cat, weight):
        kw = rng.choice(list(WORDS), rng.integers(0, 4), replace=False)
        p = CATS[c] * np.prod([WORDS[k] for k in kw]) * (1 + 0.08 * w) * rng.lognormal(0, 0.35)
        text = " ".join(list(kw) + list(rng.choice(FILLER, 6)) + [c])
        rows.append((text, w)); prices.append(min(p, 999))
    return rows, np.array(prices)
rows, price = make(50_000)
cat = np.array([r[0].split()[-1] for r in rows])
print(f"raw catalogue: mean ${price.mean():.0f}, median ${np.median(price):.0f}, automotive share {np.mean(cat == 'automotive'):.0%}")
w = price ** 2 * np.where(cat == "automotive", 0.2, 1.0)     # favour pricier items, damp an over-represented category
idx = rng.choice(len(rows), 20_000, replace=False, p=w / w.sum())   # weighted sample WITHOUT replacement
print(f"curated sample: mean ${price[idx].mean():.0f}, median ${np.median(price[idx]):.0f}, automotive share {np.mean(cat[idx] == 'automotive'):.0%}")
# Expected output:
# raw catalogue: mean $143, median $103, automotive share 20%
# curated sample: mean $238, median $199, automotive share 4%
```

The instructor's emphasis: dataset work beat all his hyperparameter tuning.

**LLM preprocessing.** Scraped descriptions are messy. The course has a small open model (gpt-oss-20b on Groq) rewrite each one into a tight summary — title, category, brand, description, details — raising signal-to-noise for every later model. For 800k items it uses a **batch API**: upload a JSONL file of requests (each with a `custom_id`, because results come back out of order), wait up to 24 hours, pay about half price.

**Splits.** Classical ML used ~80/10/10. With lots of data, put almost everything in training and keep validation and test just big enough for a stable metric: 800k / 10k / 10k (lite: 20k / 1k / 1k). Use validation as often as you like; touch test once.

## 3. The baseline ladder

**The question.** Before any LLM, how good can simple models get?

Each rung adds one idea. Always start at the bottom — if the relationship were linear, no LLM would beat linear regression.

```python
import numpy as np
rng = np.random.default_rng(0)                            # synthetic Amazon-style catalogue
CATS = {"automotive": 40, "electronics": 90, "appliances": 120, "toys": 25, "tools": 60}   # typical price per category
WORDS = {"professional": 1.8, "stainless": 1.4, "wireless": 1.5, "kit": 1.3, "replacement": 0.6, "mini": 0.7,
         "heavy-duty": 1.6, "smart": 1.7, "pack": 0.9, "premium": 1.5}                      # price multipliers
FILLER = "durable easy install black compact design fits most models quality".split()
def make(n):
    cat = rng.choice(list(CATS), n); weight = rng.gamma(2.0, 2.0, n)                         # pounds
    rows, prices = [], []
    for c, w in zip(cat, weight):
        kw = rng.choice(list(WORDS), rng.integers(0, 4), replace=False)
        p = CATS[c] * np.prod([WORDS[k] for k in kw]) * (1 + 0.08 * w) * rng.lognormal(0, 0.35)
        text = " ".join(list(kw) + list(rng.choice(FILLER, 6)) + [c])
        rows.append((text, w)); prices.append(min(p, 999))
    return rows, np.array(prices)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from scipy.sparse import hstack, csr_matrix
(train_rows, y_tr), (test_rows, y_te) = make(8000), make(500)
mae = lambda p: np.abs(p - y_te).mean()
r2 = lambda p: 1 - ((p - y_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
feats = lambda rs: np.array([[w, len(t)] for t, w in rs])         # hand-made features: weight, text length
bow = CountVectorizer().fit([t for t, _ in train_rows])            # bag of words over the descriptions
Xb_tr, Xb_te = (hstack([bow.transform([t for t, _ in rs]), csr_matrix(feats(rs))]).tocsr() for rs in (train_rows, test_rows))
results = {
    "random $1-999": rng.uniform(1, 999, len(y_te)),
    "constant (train mean)": np.full(len(y_te), y_tr.mean()),
    "linear: weight + length": LinearRegression().fit(feats(train_rows), y_tr).predict(feats(test_rows)),
    "bag of words + linear": LinearRegression().fit(Xb_tr, y_tr).predict(Xb_te),
    "random forest": RandomForestRegressor(100, random_state=0, n_jobs=-1).fit(Xb_tr, y_tr).predict(Xb_te),
    "gradient boosting, log target": np.exp(HistGradientBoostingRegressor(random_state=0)
                                      .fit(Xb_tr.toarray(), np.log(y_tr)).predict(Xb_te.toarray())),
}
for name, p in results.items(): print(f"{name:32s} MAE ${mae(p):6.2f}   R² {r2(p):5.2f}")
# Expected output:
# random $1-999                    MAE $407.27   R² -10.40
# constant (train mean)            MAE $ 95.13   R² -0.00
# linear: weight + length          MAE $ 82.74   R²  0.24
# bag of words + linear            MAE $ 55.70   R²  0.64
# random forest                    MAE $ 45.18   R²  0.75
# gradient boosting, log target    MAE $ 41.11   R²  0.76
```

Read the ladder:
- **Random** guesses are wildly wrong (negative R²). **Constant** (the training mean) defines R² = 0.
- **Hand-made features** (weight, text length) help a little — **feature engineering** was the core craft of classical data science.
- **Bag of words** — count each word, regress on the counts — is the big jump: the words carry the price signal.
- **Random forest** and **gradient boosting** capture interactions ("premium" *and* "electronics"). Training on **log price** helps because prices are skewed and errors are proportional; the course trains on raw prices.

The course's ladder on real data had the same shape: random $382, constant $106, linear on weight and length $102, bag of words $77, random forest $72, XGBoost $68. The instructor priced 100 items by hand: $88.

## 4. Neural networks and the evaluation harness

The course's "vanilla" network: 8 layers on 5,000 hashed binary word features (~669k parameters), MSE loss, 2 epochs → **$64**, better than every classical model. Later, a deep **residual** network (10 blocks of Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm with skip connections, 289M parameters, 5 epochs, ~4 hours on a laptop GPU) reached **$46** — nearly matching the best frontier model.

The course's evaluator takes any function from item to price and reports MAE, MSE and R² on the first 200 test items, plus two charts: predicted vs true, coloured green / orange / red by how far off each point is, and the running mean error with a 95% confidence band as items accumulate. A small version:

```python
import numpy as np
rng = np.random.default_rng(0)                            # synthetic Amazon-style catalogue
CATS = {"automotive": 40, "electronics": 90, "appliances": 120, "toys": 25, "tools": 60}   # typical price per category
WORDS = {"professional": 1.8, "stainless": 1.4, "wireless": 1.5, "kit": 1.3, "replacement": 0.6, "mini": 0.7,
         "heavy-duty": 1.6, "smart": 1.7, "pack": 0.9, "premium": 1.5}                      # price multipliers
FILLER = "durable easy install black compact design fits most models quality".split()
def make(n):
    cat = rng.choice(list(CATS), n); weight = rng.gamma(2.0, 2.0, n)                         # pounds
    rows, prices = [], []
    for c, w in zip(cat, weight):
        kw = rng.choice(list(WORDS), rng.integers(0, 4), replace=False)
        p = CATS[c] * np.prod([WORDS[k] for k in kw]) * (1 + 0.08 * w) * rng.lognormal(0, 0.35)
        text = " ".join(list(kw) + list(rng.choice(FILLER, 6)) + [c])
        rows.append((text, w)); prices.append(min(p, 999))
    return rows, np.array(prices)
import torch, torch.nn as nn, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
torch.manual_seed(0)
(train_rows, y_tr), (test_rows, y_te) = make(8000), make(500)
hv = HashingVectorizer(n_features=2048, binary=True)        # words -> fixed-size 0/1 vector, no vocabulary to store
X = lambda rs: torch.tensor(np.hstack([hv.transform([t for t, _ in rs]).toarray(), [[w / 10] for _, w in rs]]), dtype=torch.float32)
Xtr, Xte = X(train_rows), X(test_rows)
ytr = torch.tensor(np.log(y_tr), dtype=torch.float32)     # train on log-price: errors become relative
net = nn.Sequential(nn.Linear(2049, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for epoch in range(15):
    for idx in torch.randperm(len(Xtr)).split(64):
        loss = ((net(Xtr[idx]).squeeze(1) - ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): p = net(Xte).squeeze(1).exp().numpy()
err = np.abs(p - y_te)
print(f"neural net MAE ${err.mean():.2f}   R² {1 - ((p - y_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum():.2f}")
print(f"within $40 or 20%: {np.mean((err < 40) | (err / y_te < 0.2)):.0%}")
color = np.where((err < 40) | (err / y_te < 0.2), "green", np.where((err < 80) | (err / y_te < 0.4), "orange", "red"))
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
a.scatter(y_te, p, c=color, s=8); a.plot([0, 999], [0, 999], "k--"); a.set(xlabel="true price", ylabel="predicted")
run = np.cumsum(err) / np.arange(1, len(err) + 1)          # running mean error as test items accumulate
ci = 1.96 * np.array([err[:k].std() / np.sqrt(k) for k in range(1, len(err) + 1)])
b.plot(run); b.fill_between(range(len(err)), run - ci, run + ci, alpha=0.3); b.set(xlabel="test items", ylabel="mean abs error ($)")
plt.show()
# Expected output:
# neural net MAE $40.35   R² 0.78
# within $40 or 20%: 73%
```

`HashingVectorizer` maps each word to one of 2,048 slots by a hash, so there's no vocabulary to store. The running-error chart is the important habit: with 200 items the band is still several dollars wide, so small leaderboard differences are noise.

## 5. Frontier models, zero-shot

Prompt: estimate the price, reply with the price only, then the product summary. No training. Course results on 200 items (Gemini 3 Pro on 50): GPT-4.1-nano $63, Gemini 2.5 Flash-Lite $59, Grok 4.1 Fast $58, Gemini 3 Pro $51, Claude Opus 4.5 $47, **GPT-5.1 $45** (similar with or without reasoning). World knowledge does the work: frontier models "know" what a cordless drill costs. A caveat the course doesn't stress: these are public Amazon prices, which may be in the models' training data — part of the score may be memory.

## 6. Fine-tuning a frontier model — and why it got worse

**The flow.** (1) Write training examples as **JSONL**: each line a chat — system prompt, user message with the product, assistant reply with the price. (2) Upload and start a supervised fine-tuning (SFT) job; monitor training and validation loss. (3) Evaluate the resulting model like any other.

```python
import json, io
items = [("Cordless drill kit, 20V, two batteries", 129.0), ("Replacement wiper blades, 2-pack", 18.5)]
def record(text, price):                                  # one supervised fine-tuning example in chat format
    return {"messages": [
        {"role": "system", "content": "Estimate the price of the product. Reply with the price only."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": f"${price:.2f}"}]}   # the label the model learns to produce
buf = io.StringIO()
for t, p in items: buf.write(json.dumps(record(t, p)) + "\n")   # JSONL: one JSON object per line
print(buf.getvalue().splitlines()[0])
test_prompt = record(*items[0])["messages"]               # BUG to avoid: this still contains the assistant answer
print("leaks the answer into the test prompt:", test_prompt[-1]["role"] == "assistant")
print("correct test prompt roles:", [m["role"] for m in test_prompt[:-1]])
# Expected output:
# {"messages": [{"role": "system", "content": "Estimate the price of the product. Reply with the price only."}, {"role": "user", "content": "Cordless drill kit, 20V, two batteries"}, {"role": "assistant", "content": "$129.00"}]}
# leaks the answer into the test prompt: True
# correct test prompt roles: ['system', 'user']
```

The last lines show a real bug from the course: building test prompts from the training records leaves the assistant's answer in the prompt, and the model scores a perfect, meaningless zero error.

**The run.** gpt-4.1-nano, 20,000 examples ($3.42), 1 epoch, batch size 16, validation on 50 items. Training loss fell fast — the model quickly learned to answer "$…". Validation loss bounced around. Test error: **$76 — worse than the same model zero-shot ($63)**. Earlier runs with 200–20,000 examples ranged from $68 to $96, sometimes with absurd outliers ($10,000).

**Why it failed.** The task is already easy to state in a prompt, and the model already has the world knowledge. A few thousand price labels mostly add noise to a model that's huge relative to the data. Fine-tuning a frontier model is for:
- **tone and style** (sound like your brand);
- **format reliability** (always emit this structure);
- **fixing specific edge-case failures** with targeted examples;
- tasks that are **hard to describe** in a prompt but easy to show.

Teaching a model a **new specialised skill from lots of data** suits an open model you fully control — which is Week 7, where a 3B Llama reaches $40. OpenAI's guidance is to start with 50–100 examples; the course suggests students do that. Other offerings exist (preference fine-tuning such as DPO, reinforcement fine-tuning), with model support that changes over time.

## Trade-offs

- **Classical ML vs LLM.** Cheap, fast, explainable, needs features vs world knowledge out of the box, costs per call.
- **More data vs better data.** Curation (filters, dedup, rebalancing, LLM rewriting) often beats more volume.
- **Frontier fine-tuning vs better prompting.** Locked-in style/format vs a quick, reversible change; for knowledge tasks, prompting or RAG usually wins.
- **Raw vs log target.** Matches the dollar metric directly vs handles skew and proportional errors better.
- **Big test set vs cost.** Tighter error bars vs more API calls per evaluation.

## Common Pitfalls

- **No baselines.** A $45 error means nothing until you know the mean predictor scores $106.
- **Duplicates across splits.** Leakage inflates validation and wastes tuning.
- **Tuning on the test set.** Validation for decisions, test once at the end.
- **The answer in the test prompt.** Strip the assistant turn when building evaluation prompts.
- **Comparing runs on different item counts.** 50 vs 200 vs 100 items aren't comparable; report confidence intervals.
- **Fine-tuning a frontier model for knowledge.** It's for behaviour; use RAG or an open-model fine-tune for skills.
- **Trusting one fine-tuning run.** Results vary a lot run to run; repeat before concluding.
- **Ignoring memorization.** Public benchmark-like data may be in the model's training set.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| First pass at a prediction task | Baseline ladder: constant → linear → BoW → boosting | Jumping to an LLM | Know what "good" means |
| Text describes the item | Bag of words / hashed features + boosting or NN | Hand features only | The words carry the signal |
| Skewed numeric target | Log target, MAE reported in original units | Raw MSE | Proportional errors, readable metric |
| Need world knowledge, small data | Frontier model zero-shot (+ RAG) | Training from scratch | Knowledge comes for free |
| Want a fixed tone/format | Frontier fine-tune with 50–100 examples | Thousands of noisy labels | Behaviour, not knowledge |
| Want a new specialised skill | Open-model QLoRA on lots of data ([qlora-fine-tuning](qlora-fine-tuning.md)) | Frontier SFT | Full control, beats frontier here |
| Preprocess 800k items with an LLM | Batch API with `custom_id` | One call at a time | Half price, async |

## References

- [Course repository: ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering)
- [Amazon Reviews 2023 (McAuley Lab) on Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) · [project page](https://amazon-reviews-2023.github.io/) · [Hou et al. — the dataset paper (2024)](https://arxiv.org/abs/2403.03952)
- [Groq batch API](https://console.groq.com/docs/batch)
- [scikit-learn HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) · [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) · [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [He et al. — Deep Residual Learning (ResNet, 2015)](https://arxiv.org/abs/1512.03385) — the skip connections in the 289M-parameter network.
- OpenAI's fine-tuning guide (platform.openai.com/docs/guides/fine-tuning) — formats, supported models, recommended example counts.

## See Also

- [qlora-fine-tuning](qlora-fine-tuning.md) — Week 7: fine-tuning an open 3B model to beat the frontier on this task.
- [agentic-ai-systems](agentic-ai-systems.md) — Week 8: RAG pricer, ensemble, and agents built on these models.
- [choosing-llms](choosing-llms.md) — model vs business metrics, error bars.
- [artificial-neural-networks](../deep-learning/artificial-neural-networks.md) — the training loop, baselines and leakage-free preprocessing.
