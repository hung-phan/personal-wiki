"""Diagrams for wiki/deep-learning notes.  Run: python3 scripts/deep_learning_diagrams.py wiki/deep-learning/assets [ae|rbm|ann ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_svg as ps
from paper_svg import T, R, A, P, C, header, matrix, colvec, note
ps.OUT = sys.argv[1]
svg = ps.svg
WHICH = sys.argv[2:] or ["all"]
def want(n): return "all" in WHICH or n in WHICH

# ---------------- AE 1: bottleneck + masked loss ----------------
if want("ae"):
    b = [header("AUTOENCODERS", "Squeeze, rebuild, grade only what you know", 960)]
    # left: funnel
    b.append(T(40, 96, "1. the network", "h2"))
    layers = [(70, 8, "pur", "input", "100 movies"), (170, 5, "blu", "", ""), (260, 3, "yel", "code", "16 numbers"),
              (350, 5, "blu", "", ""), (450, 8, "grn", "output", "100 guesses")]
    for x, n, c, lab, sub in layers:
        top = 120 + (8 - n) * 14
        for k in range(n):
            b.append(C(x, top + k * 28 + 14, 10, c))
        if lab:
            b.append(T(x, 360, lab, "h2", "middle")); b.append(T(x, 376, sub, "mut", "middle"))
    for (x1, n1, *_), (x2, n2, *_) in zip(layers, layers[1:]):
        for i in range(n1):
            for j in range(n2):
                y1 = 120 + (8 - n1) * 14 + i * 28 + 14; y2 = 120 + (8 - n2) * 14 + j * 28 + 14
                b.append(f'<line x1="{x1 + 10}" y1="{y1}" x2="{x2 - 10}" y2="{y2}" class="lnd" stroke-width="0.4"/>')
    b.append(T(120, 118, "encoder", "hand", "middle")); b.append(T(400, 118, "decoder", "hand", "middle"))
    b.append(P("M450,392 C 450,420 70,420 70,392", "lnr", "arr"))
    b.append(T(260, 436, "target = the input itself (self-supervised)", "hand", "middle"))
    # right: masked loss grid
    b.append(T(530, 96, "2. the loss — only rated cells count", "h2"))
    grid = [[5, 0, 3, 0, 0, 4], [0, 2, 0, 0, 5, 0], [4, 0, 0, 1, 0, 0], [0, 0, 5, 0, 3, 2]]
    pred = [[4.6, 3.1, 3.2, 2.8, 3.9, 4.1], [3.0, 2.4, 3.3, 3.5, 4.4, 2.9], [3.7, 3.0, 2.6, 1.6, 3.1, 2.7], [3.3, 3.8, 4.6, 2.2, 3.2, 2.5]]
    gx, gy, cw, ch = 590, 140, 52, 34
    for j in range(6): b.append(T(gx + j * cw + cw / 2, gy - 8, f"m{j + 1}", "mut", "middle"))
    for i in range(4):
        b.append(T(gx - 8, gy + i * ch + 22, f"user {i + 1}", "mut", "end"))
        for j in range(6):
            r = grid[i][j]
            b.append(R(gx + j * cw, gy + i * ch, cw, ch, "grn" if r else "grey"))
            b.append(T(gx + j * cw + cw / 2, gy + i * ch + 15, str(r) if r else "?", "numb" if r else "numf", "middle"))
            b.append(T(gx + j * cw + cw / 2, gy + i * ch + 29, f"→{pred[i][j]}", "num" if r else "numf", "middle"))
    b.append(T(530, 300, "green = rated: error counts in the loss", "handk"))
    b.append(T(530, 318, "grey = unrated (0): output ignored, no gradient", "handk"))
    b.append(T(530, 342, "the grey guesses ARE the recommendations", "hand"))
    b.append(note(530, 362, 400, 80, "Grade fairly", ["Test: feed what the user rated before,", "score only ratings you held back."], "handk"))
    b.append(note(30, 460, 900, 84, "Key idea",
                  ["A narrow middle forces the net to keep only patterns shared across many users (\"likes sci-fi\").",
                   "If the middle is as wide as the input, it can just copy — then it learns nothing (see variants)."]))
    svg("ae-bottleneck.svg", 960, 560, "".join(b))

# ---------------- AE 2: variants ----------------
if want("ae"):
    b = [header("AUTOENCODERS", "Four ways to stop the network from cheating", 960)]
    panels = [(30, "Undercomplete", "code narrower than input", [8, 3, 8], None, "the bottleneck itself"),
              (260, "Overcomplete", "code WIDER than input", [4, 7, 4], "copy", "can copy input → output"),
              (490, "Sparse", "wide code, few units ON", [4, 7, 4], "sparse", "penalty on activations"),
              (720, "Denoising", "corrupt input, clean target", [6, 3, 6], "noise", "learn to fill gaps")]
    for x, title, sub, sizes, mode, foot in panels:
        b.append(R(x, 92, 210, 300, "note", 8)); b.append(T(x + 105, 116, title, "h2", "middle"))
        b.append(T(x + 105, 134, sub, "mut", "middle"))
        cols = [x + 35, x + 105, x + 175]
        pos = []
        for ci, n in enumerate(sizes):
            top = 160 + (8 - n) * 13
            pos.append([(cols[ci], top + k * 26 + 13) for k in range(n)])
        for (a, bb) in [(0, 1), (1, 2)]:
            for (x1, y1) in pos[a]:
                for (x2, y2) in pos[bb]:
                    b.append(f'<line x1="{x1 + 8}" y1="{y1}" x2="{x2 - 8}" y2="{y2}" class="lnd" stroke-width="0.35"/>')
        for ci, col in enumerate(pos):
            for k, (cx, cy) in enumerate(col):
                c = ["pur", "yel", "grn"][ci]
                if mode == "sparse" and ci == 1 and k not in (1, 4): c = "grey"
                if mode == "noise" and ci == 0 and k in (1, 4): c = "grey"
                b.append(C(cx, cy, 8, c))
                if mode == "noise" and ci == 0 and k in (1, 4): b.append(T(cx, cy + 4, "×", "red", "middle"))
        if mode == "copy":
            for k in range(4):
                y1 = pos[0][k][1]; y2 = pos[1][k + 1][1]; y3 = pos[2][k][1]
                b.append(f'<line x1="{cols[0] + 8}" y1="{y1}" x2="{cols[1] - 8}" y2="{y2}" class="lnr"/>')
                b.append(f'<line x1="{cols[1] + 8}" y1="{y2}" x2="{cols[2] - 8}" y2="{y3}" class="lnr"/>')
        b.append(T(x + 105, 372, foot, "hand", "middle"))
    b.append(note(30, 410, 900, 134, "When to use which",
                  ["Undercomplete — the default; pick the code size like any hyperparameter.",
                   "Overcomplete — never on its own (red path = identity shortcut); pair it with sparse or denoising.",
                   "Sparse — L1 penalty on hidden activations; good when you want many specialised features.",
                   "Denoising — randomly hide inputs, grade against the full input; helps most when the model overfits.",
                   "Stacked = more layers, trained end to end. 'Deep AE' (2006) = RBMs pre-trained layer by layer, then unrolled."], "handk"))
    svg("ae-variants.svg", 960, 560, "".join(b))
print("ok")

# ---------------- RBM 1: full BM vs restricted ----------------
if want("rbm"):
    import math
    b = [header("BOLTZMANN MACHINES", "Every node talks vs. two layers talk", 960)]
    def graph(cx, cy, vis, hid, full):
        pts_v = [(cx - 120 + i * 80, cy + 80) for i in range(vis)]
        pts_h = [(cx - 80 + i * 80, cy - 60) for i in range(hid)]
        allp = pts_v + pts_h
        o = []
        for i, p in enumerate(allp):
            for q in allp[i + 1:]:
                same_v = p in pts_v and q in pts_v; same_h = p in pts_h and q in pts_h
                if (same_v or same_h) and not full: continue
                cls = "lnr" if (same_v or same_h) else "lnd"
                o.append(f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" class="{cls}" stroke-width="{1.2 if cls=="lnr" else 0.9}"/>')
        for (x, y) in pts_v: o.append(C(x, y, 16, "yel"))
        for (x, y) in pts_h: o.append(C(x, y, 16, "pur"))
        return "".join(o)
    b.append(R(30, 90, 440, 320, "note", 8)); b.append(T(250, 116, "Boltzmann machine (full)", "h2", "middle"))
    b.append(graph(250, 250, 4, 3, True))
    b.append(T(250, 380, "red = visible–visible / hidden–hidden links", "hand", "middle"))
    b.append(T(250, 398, "too many to sample → impractical", "mut", "middle"))
    b.append(R(490, 90, 440, 320, "note", 8)); b.append(T(710, 116, "Restricted (RBM)", "h2", "middle"))
    b.append(graph(710, 250, 4, 3, False))
    b.append(T(710, 380, "only visible ↔ hidden: a bipartite graph", "hand", "middle"))
    b.append(T(710, 398, "given v, every h is independent → fast", "mut", "middle"))
    for x in (130, 590): b.append(T(x - 70, 335, "visible v", "mut")); b.append(T(x - 70, 195, "hidden h", "mut"))
    b.append(note(30, 426, 900, 118, "Energy → probability",
                  ["E(v, h) = −b·v − a·h − hᵀWv   and   P(v, h) = e^(−E) / Z,  where Z sums over ALL states.",
                   "Low energy = likely. Training reshapes W so states that look like real data get low energy.",
                   "No arrows, no output layer: the model describes the whole system, it doesn't map input → output."], "handk"))
    svg("rbm-graphs.svg", 960, 560, "".join(b))

# ---------------- RBM 2: contrastive divergence ----------------
if want("rbm"):
    b = [header("BOLTZMANN MACHINES", "Contrastive divergence: compare data with a dream", 960)]
    b.append(T(40, 104, "1. the Gibbs chain (one user's row)", "h2"))
    xs = [80, 220, 360, 500]
    labs = [("v⁰", "yel", "the real ratings"), ("h⁰", "pur", "tastes it implies"), ("v¹", "yel", "reconstruction"), ("h¹", "pur", "tastes of that")]
    for x, (l, c, sub) in zip(xs, labs):
        b.append(R(x - 45, 130, 90, 60, c, 8)); b.append(T(x, 166, l, "h1", "middle"))
        b.append(T(x, 210, sub, "mut", "middle"))
    for x1, x2, lab in [(125, 175, "sample h|v"), (265, 315, "sample v|h"), (405, 455, "sample h|v")]:
        b.append(A(x1, 160, x2, 160)); b.append(T((x1 + x2) / 2, 150, lab, "mut", "middle"))
    b.append(T(40, 250, "ΔW ∝  h⁰·v⁰ᵀ   −   h¹·v¹ᵀ", "h2"))
    b.append(T(40, 272, "first term from the data, second from the model's dream", "mut"))
    b.append(T(40, 300, "CD-1 = stop after one round trip. CD-k = k round trips.", "handk"))
    b.append(T(40, 318, "Unrated movies are clamped out of v at every step.", "handk"))
    # energy curve
    b.append(T(600, 104, "2. what the update does to the energy", "h2"))
    import math
    def ey(t): return 215 + 40 * math.exp(-((t - 0.25) ** 2) / 0.006) + 75 * math.exp(-((t - 0.72) ** 2) / 0.004) - 12 * math.sin(t * 9)   # bigger y = drawn lower = lower energy
    pts = [f"{600 + k * 5.5:.1f},{ey(k / 60):.1f}" for k in range(61)]
    b.append(f'<polyline points="{" ".join(pts)}" class="ln" stroke-width="2"/>')
    td, tr = 0.25, 0.72                                   # data sits in a shallow dip, the dream in a deep wrong valley
    xd, yd, xr, yr = 600 + td * 330, ey(td), 600 + tr * 330, ey(tr)
    b.append(C(xd, yd - 9, 9, "grn")); b.append(T(xd, yd - 62, "real data", "hand", "middle"))
    b.append(A(xd, yd - 56, xd, yd - 22, "lnr", "arr")); b.append(T(xd + 12, yd - 34, "push energy DOWN", "handk"))
    b.append(C(xr, yr - 9, 9, "pnk")); b.append(T(xr, yr + 30, "model's dream", "hand", "middle"))
    b.append(A(xr - 30, yr - 4, xr - 30, yr - 44, "lnr", "arr")); b.append(T(xr - 38, yr - 30, "push UP", "handk", "end"))
    b.append(T(600, 360, "Repeat until the dream looks like the data —", "handk"))
    b.append(T(600, 378, "then the lowest valleys sit on real ratings.", "handk"))
    b.append(note(30, 400, 900, 144, "Why not backprop?",
                  ["There's no target to compare against — the model must match a whole probability distribution.",
                   "The exact gradient needs an average over every possible state (Z): impossible for 100+ units.",
                   "CD estimates it with a short sampling chain started at the data. Cheap, biased, works well enough.",
                   "Stack trained RBMs → a deep belief network (2006): the trick that first made deep nets trainable."], "handk"))
    svg("rbm-contrastive-divergence.svg", 960, 560, "".join(b))

# ---------------- ANN 1: neuron, activations, hidden features ----------------
if want("ann"):
    import math
    b = [header("ANN · PART 1", "One neuron, then a layer of them", 960)]
    b.append(T(40, 100, "1. one neuron = weighted sum, then a squash", "h2"))
    ins = [("age", 140), ("active?", 200), ("products", 260)]
    for (lab, y), w in zip(ins, ["w₁", "w₂", "w₃"]):
        b.append(C(90, y, 16, "yel")); b.append(T(64, y + 4, lab, "mut", "end"))
        b.append(A(106, y, 222, 200)); b.append(T(165, (y + 200) / 2 - 4, w, "hand", "middle"))
    b.append(C(250, 200, 28, "blu")); b.append(T(250, 196, "Σ + b", "h2", "middle")); b.append(T(250, 212, "z", "mut", "middle"))
    b.append(A(278, 200, 320, 200))
    b.append(R(322, 176, 70, 48, "pnk", 6)); b.append(T(357, 205, "φ(z)", "h2", "middle"))
    b.append(A(392, 200, 432, 200)); b.append(C(450, 200, 16, "grn")); b.append(T(450, 232, "output", "mut", "middle"))
    b.append(T(40, 300, "z = w₁·age + w₂·active + w₃·products + b,   output = φ(z)", "handk"))
    # activation plots
    b.append(T(530, 100, "2. the squash φ — four common choices", "h2"))
    fns = [("step", lambda z: 1.0 if z > 0 else 0.0, "yes/no"), ("sigmoid", lambda z: 1 / (1 + math.exp(-z)), "0–1, a probability"),
           ("tanh", math.tanh, "−1 to 1"), ("ReLU", lambda z: max(z, 0) / 3, "0 or z: hidden-layer default")]
    for k, (name, f, sub) in enumerate(fns):
        x0, y0 = 540 + (k % 2) * 200, 120 + (k // 2) * 100
        b.append(R(x0, y0, 180, 84, "cell", 6))
        b.append(f'<line x1="{x0 + 10}" y1="{y0 + 54}" x2="{x0 + 170}" y2="{y0 + 54}" class="lnd"/>')
        pts = " ".join(f"{x0 + 10 + i * 4:.0f},{y0 + 54 - 34 * f((i - 20) / 5):.1f}" for i in range(41))
        b.append(f'<polyline points="{pts}" class="lnr" stroke-width="2"/>')
        b.append(T(x0 + 10, y0 + 16, name, "h2")); b.append(T(x0 + 10, y0 + 78, sub, "mut"))
    # hidden features
    b.append(T(40, 350, "3. hidden neurons = learned combinations (churn example, illustrative)", "h2"))
    feats = [("older AND inactive", "→ likely to leave"), ("3+ products", "→ unhappy bundle"), ("Germany AND high balance", "→ shops around")]
    for k, (f1, f2) in enumerate(feats):
        x0 = 40 + k * 300
        b.append(R(x0, 366, 280, 54, "note", 6)); b.append(T(x0 + 14, 388, f1, "handk")); b.append(T(x0 + 14, 408, f2, "hand"))
    b.append(note(30, 436, 900, 108, "Key idea",
                  ["No hidden layer = one weighted sum = logistic regression. Hidden layers let the net build its own",
                   "features from combinations of inputs — nobody hand-codes \"older AND inactive\"; training finds what helps.",
                   "Recipe: ReLU in hidden layers; sigmoid output for yes/no, softmax for many classes, nothing for numbers."], "handk"))
    svg("ann-neuron.svg", 960, 560, "".join(b))

# ---------------- ANN 2: how it learns ----------------
if want("ann"):
    import math
    b = [header("ANN · PART 1", "How a network learns: guess, measure, nudge", 960)]
    b.append(T(40, 100, "1. one training step", "h2"))
    steps = [("forward", "rows → prediction ŷ", 140), ("loss", "how wrong? e.g. cross-entropy", 205),
             ("backward", "backprop: blame per weight", 270), ("update", "w ← w − lr · gradient", 335)]
    for name, sub, y in steps:
        b.append(R(40, y - 22, 300, 48, "cell", 8)); b.append(T(56, y, name, "h2")); b.append(T(56, y + 18, sub, "mut"))
    for y in (166, 231, 296): b.append(A(190, y + 2, 190, y + 16))
    b.append(P("M342,335 C 400,335 400,140 342,140", "lnr", "arr")); b.append(T(404, 240, "repeat", "hand"))
    # loss bowl with lr cases
    b.append(T(470, 100, "2. step size (learning rate) on a 1-D cost bowl", "h2"))
    cx, cy, sy = 700, 360, 0.006                          # bowl: bottom at the centre (bigger y = lower on screen)
    pts = " ".join(f"{cx + d:.0f},{cy - sy * d * d:.1f}" for d in range(-170, 171, 5))
    b.append(f'<polyline points="{pts}" class="ln" stroke-width="2"/>')
    def on(d): return (cx + d, cy - sy * d * d)
    for seq, cls, lab, ly in [([-150, -135, -122, -110], "pur", "too small: crawls", 120),
                              ([-150, -40, -8, -1], "grn", "good: settles at the bottom", 140),
                              ([-150, 160, -165, 168], "pnk", "too big: bounces out", 160)]:
        for d1, d2 in zip(seq, seq[1:]):
            (x1, y1), (x2, y2) = on(d1), on(d2)
            b.append(f'<path d="M{x1:.0f},{y1:.0f} Q{(x1 + x2) / 2:.0f},{min(y1, y2) - 30:.0f} {x2:.0f},{y2:.0f}" class="lnd" marker-end="url(#ar)"/>')
        for d in seq: x, y = on(d); b.append(C(x, y, 5, cls))
        b.append(C(478, ly - 4, 5, cls)); b.append(T(488, ly, lab, "mut"))
    # batch vs sgd
    b.append(T(40, 404, "3. rows per update", "h2"))
    for k, (name, size) in enumerate([("batch GD", 16), ("mini-batch", 4), ("stochastic", 1)]):
        y0 = 416 + k * 40
        b.append(T(40, y0 + 18, name, "ink"))
        for i in range(16):
            b.append(R(150 + i * 22, y0, 20, 24, "yel" if (i // size) % 2 == 0 else "blu", 2))
            if (i + 1) % size == 0: b.append(f'<line x1="{150 + i * 22 + 21}" y1="{y0 - 3}" x2="{150 + i * 22 + 21}" y2="{y0 + 27}" class="lnr"/>')
    b.append(T(520, 432, "red tick = one weight update", "hand"))
    b.append(T(520, 452, "batch: 1 smooth step per pass", "handk"))
    b.append(T(520, 472, "stochastic: 1 noisy step per row", "handk"))
    b.append(T(520, 492, "mini-batch (32–256): the usual compromise —", "handk"))
    b.append(T(520, 512, "GPU-friendly, still many steps per pass", "handk"))
    svg("ann-training.svg", 960, 560, "".join(b))
