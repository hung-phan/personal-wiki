---
name: pytorch-cheatsheet
description: Task-indexed PyTorch API reference — make a tensor, pick a dtype, from_numpy vs torch.tensor, view vs reshape, requires_grad and backward, why .grad is None after zero_grad, model.train vs model.eval, no_grad for evaluation, DataLoader and TensorDataset, nn.Linear/Conv2d/MaxPool2d/LSTM argument order, conv output size with floor and padding='same', batch_first hidden-state layout, state_dict save and reload, and the translation table from pre-0.4 course code (Variable, volatile, loss.data[0], .cuda()) to modern PyTorch 2.x.
created: 2026-11-13
updated: 2026-11-13
tags: [pytorch, deep-learning, api-reference, cheatsheet, autograd, training-loop, tensors]
---

# PyTorch Cheatsheet

> **How to read this note.** Unlike its neighbours this is a *reference*, not a walkthrough: it is indexed by the task you are stuck on, not by the order a course teaches. It distils **"Practical Deep Learning with PyTorch"** (Ritchie Ng / Deep Learning Wizard, 58 lectures, 11 sections), and because the course spends six of those sections building the *same* training loop around a bigger model each time, the material collapses onto far fewer headings than the curriculum has. Where each section went:
>
> | Course section | Lands in |
> |----------------|----------|
> | 1 Introduction, 2 Software Requirements, 11 What's Next? | §Setup, Devices, Reproducibility — plus the course's `conda install pytorch` line, now outdated |
> | 3 PyTorch Fundamentals: Matrices (7 lectures) | §Tensors: Creation, Dtypes, Shapes |
> | 4 Variables and Gradients (3) | §Autograd — including the course's wrong claim that `requires_grad` defaults to `True` |
> | 5 Linear Regression (4), 6 Logistic Regression (6), 7 Feedforward NN (6) | §Building Models, §Losses and Optimizers, §The Canonical Training Loop. These three sections are one loop with `nn.Linear` → `nn.Linear`+`CrossEntropyLoss` → a hidden layer added; the *models* belong to [regression-models](../classical-ml/regression-models.md) and [classification-models](../classical-ml/classification-models.md), the loop is here |
> | 8 CNN with PyTorch (14) | §Conv and Pool Output Shapes — the course's `28→14→7` chain is worked through for calibration |
> | 9 RNN (5), 10 LSTM (6) | §Recurrent Layer Shapes — `batch_first`, hidden-state layout, and why the course's manual zero-init of `h0`/`c0` is unnecessary |
> | The "From CPU to GPU in PyTorch" lecture repeated in sections 8, 9 and 10 | §Setup, Devices, Reproducibility (once — `.to(device)`, not the course's per-tensor `.cuda()` branching) |
>
> **Know this before you start.** The course is **PyTorch 0.1.12-era** and nearly every snippet in it needs translating — `Variable`, `volatile=True`, `loss.data[0]`, `.cuda()` branches. §Course API → Modern Equivalent is that translation table, and §What the Course Predates lists the features (`torch.compile`, AMP, MPS) that did not exist when it was recorded. The API facts here were checked against the PyTorch 2.14 docs, not taken from the lectures.

## Why This Exists

**Problem.** Deep-learning theory and PyTorch spelling are two different skills. You know what a convolution does and still lose twenty minutes to `view` vs `reshape`, to a `.grad` that is `None` instead of zero, or to an `h0` whose batch dimension sits in a different place than the input's. Worse: most PyTorch material you learned from — including "Practical Deep Learning with PyTorch" (Ritchie Ng), which this note distils — was recorded against PyTorch 0.1.x/0.3, where half the idioms are now dead API.

**Key insight.** Modern PyTorch is a small surface: tensors with an optional `requires_grad` flag, modules that are callables with parameters, and one five-line loop (`zero_grad → forward → loss → backward → step`). Everything else is argument order, which is what a cheatsheet is for.

**Reach for this when:**
- You are writing training code right now and want the argument order of `nn.Conv2d` / `nn.LSTM` / `CrossEntropyLoss` without opening ten doc tabs.
- You are porting code from an older tutorial, book or course and hit `Variable`, `volatile=True`, `loss.data[0]` or `.cuda()` branching.
- A shape error: conv output size, flatten before a linear layer, hidden-state layout, logits vs probabilities.
- You need to know which call makes gradients appear, accumulate, or vanish.

**Don't reach for this when:**
- You want to understand *why* backprop, convolution or an LSTM gate works — read [artificial-neural-networks](artificial-neural-networks.md), [convolutional-neural-networks](convolutional-neural-networks.md), [recurrent-neural-networks](recurrent-neural-networks.md) instead. This note deliberately explains no theory.
- You want linear/logistic regression *as models* — [regression-models](../classical-ml/regression-models.md) and [classification-models](../classical-ml/classification-models.md) own those.
- You want distributed training, quantisation, custom autograd functions, or torchvision/torchtext pipelines. Not covered.

## Diagrams

![Anatomy of one PyTorch training step: which call creates, accumulates, clears and consumes gradients](assets/pytorch-training-step.svg)
<!-- DIAGRAM SPEC: name=pytorch-training-step | one horizontal strip, 5 numbered boxes left to right: (1) "opt.zero_grad()" (2) "out = model(xb)" (3) "loss = lossf(out, yb)" (4) "loss.backward()" (5) "opt.step()". Below the strip a thin second lane labelled "p.grad for each parameter p" showing its state under each box: under (1) "None", under (2) "None", under (3) "None", under (4) "filled (ADDS to whatever was there)", under (5) "unchanged - step only reads it". Above boxes (2) and (3) a bracket labelled "graph is built here; torch.no_grad() suppresses it". Above box (4) a small note "graph is freed after backward (retain_graph=True to keep)". Right of box (5) a dashed box labelled "model.eval() + torch.no_grad() for evaluation: boxes 1, 4, 5 are skipped". labels ASCII only, use "->" for arrows. The one idea it must land: gradients are ADDITIVE state living on the parameters, so zero_grad() is not bookkeeping politeness - skip it and step 4 sums this batch's gradient onto the last batch's. -->

## Setup, Devices, Reproducibility

Install with `pip install torch torchvision` (the course's `conda install pytorch torchvision -c soumith` is dead — the channel no longer exists and 0.1.12 is unobtainable).

| Task | Modern call |
|---|---|
| Pick a device once, portably | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — add an `mps` branch on Apple silicon |
| Move a tensor / model | `x = x.to(device)` (returns a new tensor), `model.to(device)` (in place for modules) |
| Check availability | `torch.cuda.is_available()`, `torch.backends.mps.is_available()` |
| Seed everything | `torch.manual_seed(0)` (also seeds CUDA), `np.random.seed(0)`, `random.seed(0)` |
| Seed a single sampler | `torch.Generator().manual_seed(0)`, passed to `randperm` / `DataLoader(generator=...)` |
| Force deterministic kernels | `torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG=:16:8` (cuDNN RNNs are non-deterministic otherwise) |
| One-line speedup | `model = torch.compile(model)` |
| Mixed precision | `with torch.autocast(device_type='cuda', dtype=torch.bfloat16):` around the forward, plus `torch.amp.GradScaler` for fp16 |

Threads and determinism matter more than they look: without a seed, two runs of the same model differ by enough to invert a model comparison (see Common Pitfalls).

## Tensors: Creation, Dtypes, Shapes

The three creation routes that behave differently, and the two reshapes that are not the same function:

| Want | Call | Note |
|---|---|---|
| Tensor from Python data | `torch.tensor([[1,2],[3,4]])` | dtype **inferred** (ints → `int64`); pass `dtype=torch.float32` to force |
| Tensor from a numpy array, no copy | `torch.from_numpy(arr)` | **shares storage** — editing the array edits the tensor |
| Tensor from a numpy array, copy | `torch.tensor(arr)` | independent memory |
| Avoid a copy when possible | `torch.as_tensor(x)` | keeps autograd history, no copy when dtype/device already match |
| Copy an existing tensor | `t.detach().clone()` | not `torch.tensor(t)`, which warns |
| Back to numpy | `t.numpy()` | also shares storage; needs CPU and `detach()` first if it requires grad |
| Zeros / ones / range / random | `torch.zeros(2,3)`, `torch.ones_like(t)`, `torch.arange(6.)`, `torch.randn(2,3)` | all take `dtype=`, `device=`, `requires_grad=` |
| Shape | `t.shape` or `t.size()`; `t.size(0)` for one axis | `t.numel()` for element count |
| Reshape (contiguous only, no copy) | `t.view(2, 3)` | errors on a non-contiguous tensor, e.g. after `.transpose()` |
| Reshape (always works) | `t.reshape(3, 2)` | copies if it must; the safe default |
| Flatten all but batch | `nn.Flatten()` as a layer, or `t.flatten(1)` | replaces the course's `out.view(out.size(0), -1)` |
| In-place anything | trailing underscore: `a.add_(b)`, `t.clamp_(0)` | mutates `a`; breaks autograd if `a` is needed for backward |
| Read one number out | `t.item()` | only on a 1-element tensor |

Dtype pairings worth memorising: images and weights are `float32`; class labels for `CrossEntropyLoss` must be `int64` (`torch.long`); masks are `bool`. A `RuntimeError: expected scalar type Long but found Float` is almost always a label-dtype problem.

This snippet answers: which creation call copies, which shares, and does `view` differ from `reshape` in practice?

```python
import numpy as np, torch
a = torch.tensor([[1, 2], [3, 4]])                      # from data; dtype inferred (int64 here)
print(a.dtype, a.shape, a.sum().item())
print("float version:", torch.tensor([[1, 2]], dtype=torch.float32).dtype)
n = np.array([1.0, 2.0])
shared = torch.from_numpy(n)                            # SHARES memory with the numpy array
copied = torch.tensor(n)                                # copies
n[0] = 99
print("from_numpy sees the edit:", shared[0].item(), "  torch.tensor does not:", copied[0].item())
print("back to numpy (also shared):", shared.numpy()[0])
x = torch.arange(6.)
print("view (needs contiguous):", x.view(2, 3).shape, " reshape (always works):", x.reshape(3, 2).shape)
print("flatten all but batch:", torch.nn.Flatten()(torch.zeros(8, 3, 4, 4)).shape)
print("add_ modifies in place:", torch.ones(2).add_(5).tolist())
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("portable device placement:", torch.zeros(2).to(dev).device)   # .to(dev) instead of .cuda()
# Expected output:
# torch.int64 torch.Size([2, 2]) 10
# float version: torch.float32
# from_numpy sees the edit: 99.0   torch.tensor does not: 1.0
# back to numpy (also shared): 99.0
# view (needs contiguous): torch.Size([2, 3])  reshape (always works): torch.Size([3, 2])
# flatten all but batch: torch.Size([8, 48])
# add_ modifies in place: [6.0, 6.0]
# portable device placement: mps:0
```

The interesting line is the shared-memory one: after `n[0] = 99` the `from_numpy` tensor reports 99 while the `torch.tensor` copy still reports 1.0 — same source array, two different answers, and a silent bug if you assumed a copy. `nn.Flatten()` turns `(8, 3, 4, 4)` into `(8, 48)`: batch dim preserved, everything else collapsed. The device line prints whatever this machine has; on this Mac it resolves to `mps:0`, which is exactly the point of writing `.to(device)` instead of `.cuda()` — the same file runs on CUDA, MPS and CPU untouched.

## Autograd

| Task | Call |
|---|---|
| Make a tensor trainable | `torch.tensor([2.0], requires_grad=True)` — **the default is `False`** |
| Flip it later | `t.requires_grad_(True)` (leaf tensors only) |
| Compute gradients | `loss.backward()` on a scalar; `y.backward(gradient=v)` for a non-scalar |
| Read a gradient | `p.grad` — same shape as `p`, or `None` |
| Clear gradients | `opt.zero_grad()` or `model.zero_grad()` |
| Turn off graph building | `with torch.no_grad():` block, or `@torch.no_grad()` decorator |
| Cut one tensor out of the graph | `t.detach()` |
| Keep the graph for a second backward | `loss.backward(retain_graph=True)` |
| Gradient of an output w.r.t. arbitrary inputs | `torch.autograd.grad(out, inputs)` |
| Clip exploding gradients | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after `backward()`, before `step()` |

Three facts that cause most autograd confusion, all demonstrated below: `requires_grad` defaults to `False`; `.grad` **accumulates** rather than overwrites; and since PyTorch 2.x `zero_grad()` defaults to `set_to_none=True`, so after clearing, `.grad` is `None`, not a zero tensor.

```python
import torch
w = torch.tensor([2.0], requires_grad=True)             # no Variable wrapper needed since PyTorch 0.4
print("requires_grad defaults to False:", torch.tensor([2.0]).requires_grad)
y = (3 * w ** 2).sum(); y.backward()
print("dy/dw at w=2 (expect 12):", w.grad.item())
y2 = (3 * w ** 2).sum(); y2.backward()
print("gradients ACCUMULATE, so now 24:", w.grad.item(), "-> call opt.zero_grad() every step")
opt = torch.optim.SGD([w], lr=0.1)
opt.zero_grad()                                          # set_to_none=True is the default in PyTorch 2.x
print("after zero_grad, .grad is", w.grad, "(None, not a zero tensor)")
with torch.no_grad():                                    # replaces the old volatile=True
    print("inside no_grad, results need no graph:", (w * 2).requires_grad)
print("detach() cuts one tensor out of the graph:", (w * 2).detach().requires_grad)
print("read a scalar with .item():", (w * 2).item(), "- the old loss.data[0] no longer works")
# Expected output:
# requires_grad defaults to False: False
# dy/dw at w=2 (expect 12): 12.0
# gradients ACCUMULATE, so now 24: 24.0 -> call opt.zero_grad() every step
# after zero_grad, .grad is None (None, not a zero tensor)
# inside no_grad, results need no graph: False
# detach() cuts one tensor out of the graph: False
# read a scalar with .item(): 4.0 - the old loss.data[0] no longer works
```

For `y = 3w²` at `w = 2` the gradient is `6w = 12`, which is what the first `backward()` writes. Running the *same* forward and backward again prints **24**, not 12 — the second gradient was added to the first. That accumulation is the whole reason `zero_grad()` exists, and it is also the feature that lets you emulate a large batch by summing gradients over several small ones. After `opt.zero_grad()` the printout is literally `None`: any code that does `p.grad.norm()` or `p.grad.zero_()` right after a `zero_grad()` call will raise `AttributeError` on `None` unless you pass `set_to_none=False`. The last three lines show the three escape hatches from the graph — a `no_grad()` block (results carry `requires_grad=False`), `detach()` on a single tensor, and `.item()` to pull a Python float out for logging.

The course's claim that `requires_grad` defaults to `True` is wrong, and was wrong when it was recorded; the snippet's second line is the disproof.

## Building Models

A module is a class with `__init__` (declare layers) and `forward` (use them); calling `model(x)` dispatches to `forward` plus hooks — never call `model.forward(x)` yourself.

**Layers, by job:**

| Job | Layer | Argument order / note |
|---|---|---|
| Fully connected | `nn.Linear(in_features, out_features, bias=True)` | input `(N, ..., in_features)` |
| 2-D convolution | `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)` | `out_channels` = number of kernels; `padding='same'` legal at stride 1 |
| 1-D convolution | `nn.Conv1d(...)` same signature | input `(N, C, L)` |
| Transposed conv | `nn.ConvTranspose2d(...)` | upsampling |
| Max / average pooling | `nn.MaxPool2d(kernel_size, stride=None, padding=0)`, `nn.AvgPool2d(...)` | **`stride` defaults to `kernel_size`** |
| Pool to a fixed size | `nn.AdaptiveAvgPool2d((1,1))` | the modern "global average pool" head |
| Flatten before a linear head | `nn.Flatten()` | `start_dim=1` by default |
| Normalisation | `nn.BatchNorm2d(num_features)`, `nn.BatchNorm1d`, `nn.LayerNorm(shape)` | behaviour differs between `train()` and `eval()` |
| Regularisation | `nn.Dropout(p=0.5)`, `nn.Dropout2d` | **off** in `eval()` |
| Activations (as layers) | `nn.ReLU()`, `nn.LeakyReLU()`, `nn.GELU()`, `nn.Tanh()`, `nn.Sigmoid()`, `nn.Softmax(dim=1)` | functional twins in `torch.nn.functional` |
| Recurrent | `nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', batch_first=False, dropout=0, bidirectional=False)` | `'relu'` is the other `nonlinearity` |
| LSTM / GRU | `nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=False, proj_size=0)`, `nn.GRU(...)` | LSTM takes no activation argument |
| Lookup table | `nn.Embedding(num_embeddings, embedding_dim)` | input is `int64` indices |
| Straight-line stack | `nn.Sequential(nn.Linear(64,100), nn.ReLU(), nn.Linear(100,10))` | no custom class needed when `forward` is just a chain |
| List of layers | `nn.ModuleList([...])`, `nn.ModuleDict({...})` | a plain Python list hides parameters from the optimizer |

**Module introspection:**

| Task | Call |
|---|---|
| Print the architecture | `print(model)` |
| Count parameter tensors | `len(list(model.parameters()))` |
| Count scalars | `sum(p.numel() for p in model.parameters())` |
| Named parameters | `for name, p in model.named_parameters():` |
| Train vs eval mode | `model.train()` / `model.eval()` — toggles dropout and batch-norm statistics only |
| Freeze a submodule | `for p in sub.parameters(): p.requires_grad = False` |

## Losses and Optimizers

The single most common beginner error is feeding a loss the wrong kind of number. What each one expects:

| Loss | Model output it wants | Target it wants |
|---|---|---|
| `nn.CrossEntropyLoss()` | raw **logits** `(N, C)` — it applies log-softmax internally | class indices `(N,)` `int64`, or class probabilities |
| `nn.NLLLoss()` | **log**-probabilities (i.e. after `log_softmax`) | class indices |
| `nn.BCEWithLogitsLoss()` | raw logits, any shape | floats in [0,1], same shape |
| `nn.BCELoss()` | probabilities (after sigmoid) | floats in [0,1] |
| `nn.MSELoss()` | predictions | same shape as predictions |
| `nn.L1Loss()`, `nn.SmoothL1Loss()` | predictions | same shape |

So: never put a `Softmax` or `Sigmoid` layer in front of `CrossEntropyLoss` / `BCEWithLogitsLoss`. Do apply `softmax(dim=1)` yourself if you want probabilities to display.

| Optimizer / schedule | Call |
|---|---|
| Plain SGD | `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)` |
| Default modern choice | `torch.optim.Adam(model.parameters(), lr=1e-3)` |
| Adam with correct weight decay | `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)` |
| Per-group learning rates | `optim.SGD([{'params': head.parameters(), 'lr': 1e-2}, {'params': body.parameters(), 'lr': 1e-4}])` |
| Step / cosine / plateau / one-cycle schedules | `optim.lr_scheduler.StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `OneCycleLR` — call `sched.step()` once per epoch (per batch for `OneCycleLR`) |
| Current learning rate | `opt.param_groups[0]['lr']` |

The course uses SGD at `lr=0.01` and nothing else — no momentum, no scheduler, no other optimizer. That is a teaching simplification, not a recommendation.

## Data Pipeline

| Task | Call |
|---|---|
| Wrap tensors already in memory | `TensorDataset(X, y)` from `torch.utils.data` |
| Custom dataset | subclass `Dataset`, implement `__len__` and `__getitem__` |
| Batch, shuffle, parallel-load | `DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)` |
| Train/val split | `torch.utils.data.random_split(ds, [0.8, 0.2])` |
| Built-in vision datasets | `torchvision.datasets.MNIST(root='./data', train=True, transform=..., download=True)` |
| Dataset tensors / labels | `ds.data`, `ds.targets` (the old `train_data` / `train_labels` names were removed) |
| Standard image transform | `transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])` |

`shuffle=True` on train, `False` on val/test. Epoch arithmetic, if you prefer to think in iterations (the course's framing): `num_epochs = n_iters // (len(train_dataset) // batch_size)` — with 60 000 samples and `batch_size=100`, 3000 iterations is 5 epochs. Note the integer division; the course's `/` returns a float in Python 3.

## The Canonical Training Loop

Everything above assembles into one shape that does not change between architectures. This snippet is the whole thing end to end — dataset, loader, train/eval modes, `no_grad()` evaluation, a baseline to compare against, save and reload — on scikit-learn's 8x8 digits so it runs in seconds.

```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
torch.manual_seed(0); torch.set_num_threads(1)
d = load_digits()
perm = torch.randperm(len(d.target), generator=torch.Generator().manual_seed(0))   # digits ship in subject order
X = torch.tensor(d.data / 16.0, dtype=torch.float32)[perm]; y = torch.tensor(d.target)[perm]
ntr = 1400
train = DataLoader(TensorDataset(X[:ntr], y[:ntr]), batch_size=64, shuffle=True)
model = nn.Sequential(nn.Linear(64, 100), nn.ReLU(), nn.Linear(100, 10))
opt, lossf = torch.optim.Adam(model.parameters(), 1e-3), nn.CrossEntropyLoss()   # CE takes LOGITS, no softmax
for epoch in range(8):
    model.train()                                        # matters once the model has dropout or batch norm
    for xb, yb in train:
        opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
model.eval()
with torch.no_grad():
    acc = (model(X[ntr:]).argmax(1) == y[ntr:]).float().mean().item()
print(f"test accuracy {acc:.3f}   (majority-class baseline {y[ntr:].bincount().max().item() / len(y[ntr:]):.3f})")
torch.save(model.state_dict(), "/tmp/m.pt")              # save the weights, not the pickled object
clone = nn.Sequential(nn.Linear(64, 100), nn.ReLU(), nn.Linear(100, 10))
print("reload:", clone.load_state_dict(torch.load("/tmp/m.pt")))
# Expected output:
# test accuracy 0.924   (majority-class baseline 0.116)
# reload: <All keys matched successfully>
```

Read it as five blocks. (1) `TensorDataset` + `DataLoader` turn two tensors into shuffled batches. (2) `nn.Sequential` replaces a custom class because the forward pass is a plain chain. (3) The inner loop is the only part you must know by heart: `opt.zero_grad()`, forward, loss, `backward()`, `opt.step()`. (4) `model.train()` / `model.eval()` bracket the two phases, and the evaluation sits inside `torch.no_grad()` — the course never calls either, which happens to be harmless for its models (no dropout, no batch norm) and silently wrong the moment you add one. (5) `state_dict` persistence: `torch.save(model.state_dict(), path)`, then `clone.load_state_dict(torch.load(path))`, whose return value prints `<All keys matched successfully>` when the architectures agree.

The numbers: test accuracy **0.924** against a majority-class baseline of **0.116**. Always print that second number. An accuracy quoted with no baseline is uninterpretable — on a balanced 10-class problem chance is ~0.10, so 0.924 is real learning; on a 95%-negative problem the same loop printing 0.95 would mean the model learned nothing.

Evaluation that needs gradients off but *stats* on (e.g. you want per-class accuracy) still needs `model.eval()`: `no_grad()` only stops graph building, it does not disable dropout.

## Conv and Pool Output Shapes

The shape chain is the one piece of CNN arithmetic you cannot avoid, because the first `nn.Linear` after the convolutions needs its `in_features` spelled out. The real formula per spatial dimension:

```text
O = floor( (W + 2P - D*(K-1) - 1) / S ) + 1
```

with `W` input size, `K` kernel, `P` padding, `S` stride, `D` dilation. At `D=1` this reduces to the course's `O = (W - K + 2P)/S + 1` — **with a floor the course omits**. This snippet checks the formula against PyTorch over five configurations, including the stride-3 case where the floor actually bites:

```python
import warnings; warnings.filterwarnings("ignore")   # padding="same" with an even kernel warns about a padded copy
import torch, torch.nn as nn
def out_size(W, K, P=0, S=1, D=1): return (W + 2 * P - D * (K - 1) - 1) // S + 1   # floor, not plain division
print("formula vs PyTorch, (W,K,P,S):")
for W, K, P, S in [(28, 5, 2, 1), (28, 5, 0, 1), (28, 3, 1, 2), (28, 4, 0, 3), (28, 2, 0, 2)]:
    real = nn.Conv2d(1, 1, K, stride=S, padding=P)(torch.zeros(1, 1, W, W)).shape[-1]
    print(f"  ({W},{K},{P},{S}) -> formula {out_size(W, K, P, S):2d}  PyTorch {real:2d}  {'ok' if real == out_size(W,K,P,S) else 'MISMATCH'}")
print('\n"same" padding P=(K-1)/2 keeps the size only for ODD K and stride 1:')
for K in (3, 5, 4):
    P = (K - 1) // 2
    print(f"  K={K}, P={P}: 28 -> {out_size(28, K, P, 1)}" + ("" if K % 2 else "   <- even kernel cannot stay 'same'"))
print("\nPyTorch accepts padding='same' (stride 1 only) and does the arithmetic for you:")
print("  ", nn.Conv2d(1, 1, 4, padding="same")(torch.zeros(1, 1, 28, 28)).shape[-1])
print("\nMaxPool2d defaults stride=kernel_size, which is why 'output = W/K' works for pooling but not for conv:")
print("  pool K=2:", nn.MaxPool2d(2)(torch.zeros(1, 1, 28, 28)).shape[-1], " conv K=2, S=1:", out_size(28, 2, 0, 1))
# Expected output:
# formula vs PyTorch, (W,K,P,S):
#   (28,5,2,1) -> formula 28  PyTorch 28  ok
#   (28,5,0,1) -> formula 24  PyTorch 24  ok
#   (28,3,1,2) -> formula 14  PyTorch 14  ok
#   (28,4,0,3) -> formula  9  PyTorch  9  ok
#   (28,2,0,2) -> formula 14  PyTorch 14  ok
# 
# "same" padding P=(K-1)/2 keeps the size only for ODD K and stride 1:
#   K=3, P=1: 28 -> 28
#   K=5, P=2: 28 -> 28
# ... trimmed
```

What it establishes. The floored formula matches `nn.Conv2d` for all five `(W,K,P,S)` rows, including `(28,4,0,3)` where the unfloored expression gives a fraction. The "same padding" block shows `P=(K-1)/2` returning 28 for `K=3` and `K=5` but **not** for `K=4` — with an even kernel there is no symmetric integer padding that preserves size, so the course's `P=(K-1)/2` is a rule for odd kernels at stride 1 only. PyTorch will do the asymmetric padding for you if you write `padding='same'` (stride 1 only), which the course predates. The last line is the subtlety behind the course's separate pooling shortcut `O = W/K`: `nn.MaxPool2d(2)` halves 28 to 14 because pooling **defaults `stride=kernel_size`**, whereas `nn.Conv2d` defaults `stride=1`, so a `K=2` conv at stride 1 gives 27, not 14. Same window size, different default stride — the shortcut is a property of the default, not of pooling.

Worked chain for the course's CNN, for calibration: `28 →(K=5,P=2) 28 →pool 14 →(K=5,P=2) 14 →pool 7`, so the head is `nn.Linear(32*7*7, 10)`. With `padding=0` instead: `28 → 24 → 12 → 8 → 4`, so `nn.Linear(32*4*4, 10)`. Change a kernel and you must change that number too — or use `nn.AdaptiveAvgPool2d((1,1))` and stop caring.

## Recurrent Layer Shapes

`nn.RNN`, `nn.LSTM` and `nn.GRU` share one layout convention and one trap.

| Tensor | Default layout | With `batch_first=True` |
|---|---|---|
| input | `(L, N, H_in)` | `(N, L, H_in)` |
| output | `(L, N, D*H_out)` | `(N, L, D*H_out)` |
| `h0` / `hn` (and `c0`/`cn`) | `(D*num_layers, N, H_out)` | **unchanged** |

with `L` timesteps, `N` batch, `D = 2` if `bidirectional` else 1. Forward signatures: `out, hn = rnn(x, h0)` and `out, (hn, cn) = lstm(x, (h0, c0))`; the state arguments are optional and default to zeros.

```python
import torch, torch.nn as nn
torch.manual_seed(0)
N, L, D, H, layers = 4, 28, 28, 16, 2                    # batch, timesteps, features, hidden, stacked layers
lstm = nn.LSTM(D, H, layers, batch_first=True)
x = torch.randn(N, L, D)
out, (h, c) = lstm(x)                                    # h0/c0 default to zeros - no need to build them by hand
print("batch_first=True input", tuple(x.shape), "-> out", tuple(out.shape))
print("but the states stay LAYERS-first:", tuple(h.shape), " <- the asymmetry that bites people")
print("last timestep == final layer's h:", torch.allclose(out[:, -1, :], h[-1]))
bi = nn.LSTM(D, H, layers, batch_first=True, bidirectional=True)
out2, (h2, _) = bi(x)
print("bidirectional doubles the feature axis:", tuple(out2.shape), " states:", tuple(h2.shape))
print("for a bi-LSTM out[:, -1] is NOT h[-1]:", torch.allclose(out2[:, -1, :H], h2[-2]),
      "(forward half matches h2[-2], the reverse half lives at out2[:, 0, H:])")
# Expected output:
# batch_first=True input (4, 28, 28) -> out (4, 28, 16)
# but the states stay LAYERS-first: (2, 4, 16)  <- the asymmetry that bites people
# last timestep == final layer's h: True
# bidirectional doubles the feature axis: (4, 28, 32)  states: (4, 4, 16)
# for a bi-LSTM out[:, -1] is NOT h[-1]: True (forward half matches h2[-2], the reverse half lives at out2[:, 0, H:])
```

Three things it proves. First, with `batch_first=True` the input and output are batch-first `(4, 28, 16)` while `h` stays layers-first `(2, 4, 16)` — that asymmetry is the most common `nn.LSTM` shape bug, and the course uses `batch_first=True` without ever flagging it. Second, the states were never passed: `h0`/`c0` default to zeros, so the course's hand-built `torch.zeros(layer_dim, x.size(0), hidden_dim)` (and the `.cuda()` branch it needs) is code you can delete. Third, for the unidirectional stack `out[:, -1, :]` and `h[-1]` are the same tensor — but flip `bidirectional=True` and they diverge: `out` grows to `(4, 28, 32)`, and `out[:, -1, :H]` matches `h2[-2]` (the final *forward* state) while the reverse direction's final state lives at `out[:, 0, H:]`. So `out[:, -1]` on a bi-directional model silently feeds your classifier the reverse direction's *first* step. Use `torch.cat([h[-2], h[-1]], dim=1)`, or pool over time.

Related: when you carry hidden state across batches, detach it (`h = h.detach()`) or backward will try to walk the entire history; and `clip_grad_norm_` is the standard guard for exploding gradients, which matters most for `nn.RNN(nonlinearity='relu')` unrolled over many steps. The course covers neither.

## Save, Load, Resume

| Task | Call |
|---|---|
| Save weights | `torch.save(model.state_dict(), 'net.pt')` |
| Load weights | `model.load_state_dict(torch.load('net.pt', weights_only=True))` → `<All keys matched successfully>` |
| Save a resumable checkpoint | `torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': e}, 'ckpt.pt')` |
| Load onto a different device | `torch.load(path, map_location='cpu')` |
| Partial / renamed load | `model.load_state_dict(sd, strict=False)` — returns missing and unexpected keys |

Save the `state_dict`, not the module object: pickling the model ties the file to your class definition and import paths. `weights_only=True` is the safe default for untrusted checkpoints. Use `.pt`/`.pth`, not the course's `.pkl`.

## Course API → Modern Equivalent

The course is PyTorch 0.1.12-era. Nearly every snippet in it needs translation; this is the table.

| Course spelling | Modern equivalent | Why it changed |
|---|---|---|
| `from torch.autograd import Variable`; `Variable(t, requires_grad=True)` | `torch.tensor(..., requires_grad=True)` — drop the wrapper | `Variable` and `Tensor` merged in 0.4; `Variable(t)` just returns a tensor now. Deprecated. |
| "`requires_grad` defaults to `True`" (lecture claim) | It defaults to **`False`** | Never true. Only the leaves you mark participate in autograd — see the autograd snippet above. |
| `torch.Tensor(arr)` | `torch.tensor(arr, dtype=..., device=...)` | The legacy constructor always produces float (even from ints) and takes no `dtype`/`device`. |
| `Variable(x, volatile=True)` for inference | `with torch.no_grad():` | `volatile` removed in 0.4; `no_grad()` is a context manager (or decorator) and composes. |
| `loss.data[0]` | `loss.item()` | 0-dim tensors became real in 0.4; indexing one now raises. |
| `t.data` to escape autograd | `t.detach()`, or a `no_grad()` block | `.data` bypasses autograd's correctness tracking silently; `detach()` is tracked. |
| `if torch.cuda.is_available(): x = Variable(x.cuda()) else: ...` per tensor | `device = torch.device(...)` once, then `x = x.to(device)` | `.to()` also covers `mps` / `xpu` / `cpu` and removes every if/else the course writes by hand. |
| `correct += (predicted.cpu() == labels.cpu()).sum()` | `correct += (pred == labels).sum().item()` | The `.cpu()` calls were needed only because the two operands were on different devices. The course's reason — "`.sum()` is a Python function" / "doesn't work on GPU" — is wrong; reductions run on GPU, and the round-trip forces a needless sync. |
| `accuracy = 100 * correct / total` | `... / total` then `.item()`, or accumulate floats | `(a == b).sum()` is a tensor, so the whole expression stays a tensor. |
| `train_dataset.train_data`, `.train_labels`, `.test_data`, `.test_labels` | `ds.data`, `ds.targets` | Renamed in torchvision 0.2.2; old names removed. |
| no `model.train()` / `model.eval()` anywhere | bracket training and evaluation with them | Harmless for the course's models; wrong the moment you add dropout or batch norm. |
| evaluation loop with no `torch.no_grad()` | wrap it | Otherwise you build a graph you never use — slower, and needless memory. |
| `out.view(out.size(0), -1)` to flatten | `nn.Flatten()` as a layer, or `t.flatten(1)` | Added later; `view` also fails on non-contiguous input where `reshape` would not. |
| `padding=2` computed via `P=(K-1)/2` | `padding='same'` (stride 1) | String padding arrived in PyTorch 1.9 and handles even kernels. |
| `h0 = Variable(torch.zeros(layers, x.size(0), hidden))` | omit it — `rnn(x)` | `h0`/`c0` default to zeros. Deletes one line plus its `.cuda()` branch. |
| `torch.max(outputs.data, 1)` | `outputs.argmax(dim=1)` | Clearer when you only want the index. |
| `isinstance(train_loader, collections.Iterable)` | delete it | `collections.Iterable` was removed in Python 3.10 (`collections.abc.Iterable`); it was pedagogical anyway. |
| `torch.save(model.state_dict(), 'x.pkl')` | same call, `.pt` + `weights_only=True` on load | Practice unchanged; conventions and safety flags moved on. |
| `conda install pytorch torchvision -c soumith` | `pip install torch torchvision` | The `soumith` channel is gone; the CUDA 7.5/8.0, cuDNN 6.0, g2.2xlarge and macOS-source-build instructions are all obsolete. |

## What the Course Predates

Five things a 2026 reader gets for free that are absent from the course:

- **`torch.compile(model)`** — one-line graph compilation and speedup. [docs](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- **Mixed precision** — `torch.autocast` plus `torch.amp.GradScaler`, roughly halving memory on modern accelerators. [docs](https://docs.pytorch.org/docs/stable/amp.html)
- **`nn.Flatten`** — a layer, so a plain `nn.Sequential` can go from conv stack to linear head. [docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.flatten.Flatten.html)
- **`padding='same'` / `'valid'`** as string arguments to `nn.Conv2d`. [docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)
- **The MPS backend** — Apple-silicon GPU acceleration, which is why `.to(device)` beats `.cuda()` and why the course's "compile PyTorch from source for macOS GPU" lecture is dead. [docs](https://docs.pytorch.org/docs/stable/notes/mps.html)

Also never covered, and all one-liners you will want: `nn.Dropout`, `nn.BatchNorm*`, `bidirectional=True`, `nn.Sequential`, `torch.nn.functional`, `clip_grad_norm_`, learning-rate schedulers, any optimizer besides SGD, weight initialisation, a validation split, and `transforms.Normalize`.

## Trade-offs

| Choice | What you gain | What you give up |
|---|---|---|
| `reshape` everywhere instead of `view` | never debug a contiguity error | a silent copy you did not budget for in a hot loop |
| `nn.Sequential` instead of a `Module` subclass | three lines instead of fifteen | any non-chain forward pass: skip connections, multiple inputs, intermediate returns |
| `padding='same'` instead of a computed integer | kernel size becomes a free parameter | the shape chain is no longer visible in the code; stride 1 only |
| `AdaptiveAvgPool2d((1,1))` head instead of `Linear(C*H*W, ...)` | input size stops mattering; no shape arithmetic | you discard spatial layout the dense head could have used |
| `set_to_none=True` (the default) for `zero_grad` | less memory; unused params are provably `None` | any code reading `p.grad` must handle `None` |
| `torch.compile` | real speedup, one line | long first-call compile, harder tracebacks, occasional graph breaks |
| `from_numpy` instead of `torch.tensor` | zero-copy | aliasing bugs: mutate either side and both change |
| `num_workers > 0` | the loader stops being the bottleneck | process-startup cost, trickier debugging, platform quirks |
| Deterministic algorithms + seeds | reproducible runs and trustworthy comparisons | some kernels get slower or raise where no deterministic version exists |

## Common Pitfalls

**Autograd and the loop**
- **Forgetting `opt.zero_grad()`** adds this batch's gradient to the previous one, so the effective learning rate creeps upward. The snippet above prints 12 then 24 for the identical computation — that is the bug in miniature.
- **Reading `p.grad` right after `zero_grad()`** gives `None` in PyTorch 2.x, not zeros, because `set_to_none=True` is the default. Guard for `None` or pass `set_to_none=False`.
- **Evaluating without `torch.no_grad()`** builds a graph you discard: slower and memory-hungry. **Evaluating without `model.eval()`** leaves dropout active and batch-norm updating its running statistics, so your reported accuracy is wrong *and* your model has been mutated by the evaluation.
- **Accumulating `loss` instead of `loss.item()`** keeps every batch's graph alive and leaks memory until the epoch ends.
- **Appending a `Softmax` layer before `CrossEntropyLoss`** applies softmax twice, flattening the gradients. `CrossEntropyLoss` wants logits.

**Shapes**
- **The course's output-size formula `O = (W - K + 2P)/S + 1` has no floor.** With stride > 1 and non-divisible sizes it returns a fraction; PyTorch floors. Use `floor((W + 2P - D*(K-1) - 1)/S) + 1`.
- **`P = (K-1)/2` is not a universal same-padding rule.** It assumes an **odd** kernel and **stride 1**; for even `K` no symmetric integer padding preserves size, and at stride > 1 the idea does not apply. Prefer `padding='same'`.
- **`O = W/K` is a pooling-only shortcut** and only because `nn.MaxPool2d`/`AvgPool2d` default `stride=kernel_size`. `nn.Conv2d` defaults `stride=1`, so the same reasoning gives the wrong answer for convolution. The course states the two formulas side by side without explaining the asymmetry.
- **`batch_first=True` does not move the hidden state's batch axis.** Inputs and outputs become `(N, L, H)`; `h0`/`c0` remain `(num_layers*D, N, H)`. The course relies on this and never says so.
- **`out[:, -1]` is not `h[-1]` for a bidirectional RNN.** It is the forward direction's last step concatenated with the *reverse direction's first* step. Concatenate `h[-2]` and `h[-1]` instead.
- **Forgetting to flatten before `nn.Linear`** raises a matmul shape error; `nn.Flatten()` in the stack is the fix.

**Reading the course's numbers**
- **Every accuracy in the course is a test-set accuracy that was also used to choose between models.** There is no validation split anywhere, so the reported figures are optimistic and the "model A beats model B" conclusions are selection on the test set.
- **The activation ranking is a single unseeded run.** Sigmoid 91.95%, tanh 95.02%, ReLU 95.91% on the same 1-hidden-layer MNIST net. The ordering matches conventional wisdom, but with no seed, no repeats and no variance estimate these margins do not establish it — reshuffling alone moves numbers by that much.
- **"Max pooling beats average pooling" rests on one comparison too** (97.04% vs 93%). Do not generalise: modern architectures routinely use global *average* pooling as the final reduction. The 93% figure is doubly suspect — see the next item.
- **Do not copy the average-pooling class as shown.** In that lecture `__init__` is renamed to `avgpool1`/`avgpool2` while `forward()` still calls `self.maxpool1`/`self.maxpool2`; run as displayed it raises `AttributeError`. (It is left as a deliberate fix-it exercise, but it means the 93% came from code the viewer never sees.)
- **"Beyond two or three RNN layers performance degrades, I don't know why"** is an acknowledged gap in the course (LSTM: 96.05% at 1 layer, 95.5% at 2, 91.2% at 3). Part of the answer is gradients through depth *and* time with nothing counteracting them — and the standard remedy, gradient clipping, is never mentioned.
- **82% for logistic regression on MNIST is not that model's ceiling** — it reflects the deliberately small learning rate and 3000-iteration budget, not a limit of linear softmax regression.
- **"LSTM for any recurrent structure"** was fair advice in 2017. GRU is a cheaper near-equal, and for sequence modelling at scale Transformers displaced recurrent nets; the course mentions neither attention nor GRU.

## Decision Table

| Scenario | Use this | Not that | Why |
|---|---|---|---|
| Tensor from a numpy array you will keep mutating | `torch.tensor(arr)` | `torch.from_numpy(arr)` | `from_numpy` shares storage; later edits silently change your tensor |
| Zero-copy hand-off of a large array | `torch.from_numpy(arr)` | `torch.tensor(arr)` | avoids duplicating the buffer |
| Reshaping output of a transpose/permute | `reshape` | `view` | `view` requires contiguous memory and will raise |
| Flattening conv output for a linear head | `nn.Flatten()` | `out.view(out.size(0), -1)` | a layer, so it composes in `nn.Sequential`, and no contiguity trap |
| Multiclass classification head | `nn.Linear(...)` + `CrossEntropyLoss` | `Softmax` + `NLLLoss` hand-rolled | one fused, numerically stable op |
| Single-label binary classification | `BCEWithLogitsLoss` | `Sigmoid` + `BCELoss` | fused log-sum-exp avoids overflow |
| Running inference / evaluating | `model.eval()` **and** `torch.no_grad()` | either alone | `eval()` fixes dropout/batch norm; `no_grad()` stops graph building — different jobs |
| Choosing between architectures | a held-out validation split, reported with a baseline | the test set (what the course does) | reusing test data for selection inflates the number you report |
| Conv where you want the size preserved | `padding='same'` | computing `(K-1)/2` | correct for even kernels too, and the intent is readable |
| Classifier head independent of input size | `nn.AdaptiveAvgPool2d((1,1))` | hand-computed `Linear(C*H*W, ...)` | no shape chain to recompute when a kernel changes |
| Picking an optimizer with no time to tune | `AdamW(lr=1e-3)` | `SGD(lr=0.01)` (course default) | works at the default on most problems; SGD+momentum needs tuning to beat it |
| Last state of a bidirectional RNN | `torch.cat([h[-2], h[-1]], dim=1)` | `out[:, -1, :]` | the last timestep mixes the forward end with the reverse *start* |
| Moving work to an accelerator | `.to(device)` | `.cuda()` with an if/else | one code path across CUDA, MPS, XPU and CPU |
| Persisting a model | `state_dict` + `.pt` | pickling the module | a pickled module breaks when your class or import path moves |
| Exploding loss in a deep RNN | `clip_grad_norm_` + a smaller learning rate | more layers | depth does not fix an unstable gradient; clipping bounds it |

## References

- `torch.Tensor` and creation semantics — https://docs.pytorch.org/docs/2.14/tensors.html
- `torch.tensor` (copies, `dtype`, `requires_grad`) — https://docs.pytorch.org/docs/2.14/generated/torch.tensor.html
- Autograd mechanics; `Variable` is deprecated — https://docs.pytorch.org/docs/2.14/autograd.html
- `torch.no_grad` (replaces `volatile=True`) — https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html
- `Optimizer.zero_grad` and `set_to_none=True` — https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html
- CUDA semantics and device placement — https://docs.pytorch.org/docs/2.14/notes/cuda.html
- MPS (Apple silicon) backend — https://docs.pytorch.org/docs/stable/notes/mps.html
- `nn.Conv2d`, output-size formula and `padding='same'` — https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
- `nn.Flatten` — https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.flatten.Flatten.html
- `nn.LSTM` (shapes, defaults, bidirectional states) — https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
- `nn.RNN` — https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html
- `nn.CrossEntropyLoss` (expects logits) — https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- `nn.Dropout` (`train()` vs `eval()` behaviour) — https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- `nn.Sequential` — https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
- Optimizers and learning-rate schedulers — https://docs.pytorch.org/docs/stable/optim.html
- `clip_grad_norm_` — https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- `Dataset` / `DataLoader` — https://docs.pytorch.org/docs/stable/data.html
- Saving and loading modules (`state_dict`) — https://docs.pytorch.org/docs/2.14/notes/modules.html
- `torch.compile` — https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- Automatic mixed precision (`autocast`, `GradScaler`) — https://docs.pytorch.org/docs/stable/amp.html
- Reproducibility and deterministic algorithms — https://docs.pytorch.org/docs/stable/notes/randomness.html
- Course: "Practical Deep Learning with PyTorch", Ritchie Ng (Udemy) — the source this note translates; recorded against PyTorch 0.1.12.

## See Also

- [artificial-neural-networks](artificial-neural-networks.md) — neurons, activations, gradient descent, backprop and the training recipe as *theory*; this note is the API for it.
- [convolutional-neural-networks](convolutional-neural-networks.md) — convolution, pooling and output-size arithmetic from first principles.
- [recurrent-neural-networks](recurrent-neural-networks.md) — RNN/LSTM cells, vanishing gradients, the gate equations.
- [seq2seq-bidirectional-rnns](seq2seq-bidirectional-rnns.md) — what bidirectional actually buys you, beyond the shape rules here.
- [regression-models](../classical-ml/regression-models.md) and [classification-models](../classical-ml/classification-models.md) — linear and logistic regression as models.
