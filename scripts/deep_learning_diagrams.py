"""Diagrams for wiki/machine-learning/deep-learning notes.  Run: python3 scripts/deep_learning_diagrams.py wiki/machine-learning/deep-learning/assets [ae|rbm|ann ...]"""
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

# ---------------- CNN 1: conv -> relu -> pool -> flatten, by the numbers ----------------
if want("cnn"):
    b = [header("CONVOLUTIONAL NETWORKS", "One filter, followed all the way to the dense layer", 960)]
    img = [[0] * 7 for _ in range(7)]
    for i in range(1, 6): img[i][i] = 1
    img[1][5] = img[5][1] = 1
    k = [[1 if i == j else -1 for j in range(3)] for i in range(3)]
    fm = [[sum(img[r + a][c + d] * k[a][d] for a in range(3) for d in range(3)) for c in range(5)] for r in range(5)]
    rl = [[max(v, 0) for v in row] for row in fm]
    pl = [[max(rl[r][c] for r in range(2 * i, min(2 * i + 2, 5)) for c in range(2 * j, min(2 * j + 2, 5))) for j in range(3)] for i in range(3)]
    y0 = 130
    b.append(T(40, 104, "1. image 7×7", "h2"))
    b.append(matrix(40, y0, img, 22, 22, lambda i, j: "ink" if img[i][j] else "cell", lambda i, j: "numf" if not img[i][j] else "num"))
    for i in range(7):  # dark pixels need light text: redraw as grey boxes with bold 1
        for j in range(7):
            if img[i][j]: b.append(R(40 + j * 22, y0 + i * 22, 22, 22, "pur")); b.append(T(51 + j * 22, y0 + i * 22 + 15, "1", "numb", "middle"))
    b.append(R(40 + 22, y0 + 22, 66, 66, "hi"))
    b.append(T(213, y0 + 86, "*", "h1", "middle"))
    b.append(T(232, 104, "filter", "h2"))
    b.append(matrix(232, y0 + 44, k, 22, 22, lambda i, j: "yel" if k[i][j] > 0 else "blu"))
    b.append(T(265, y0 + 130, "+1 on the \\ diagonal", "mut", "middle")); b.append(T(265, y0 + 144, "−1 elsewhere", "mut", "middle"))
    b.append(A(305, y0 + 77, 335, y0 + 77))
    b.append(T(345, 104, "2. feature map", "h2"))
    b.append(matrix(345, y0 + 10, fm, 24, 24, lambda i, j: "pnk" if fm[i][j] < 0 else ("grn" if fm[i][j] == 3 else "cell")))
    b.append(R(345 + 24, y0 + 10 + 24, 24, 24, "hi"))
    b.append(T(405, y0 + 152, "3 = perfect match", "hand", "middle"))
    b.append(A(470, y0 + 70, 500, y0 + 70)); b.append(T(485, y0 + 60, "ReLU", "mut", "middle"))
    b.append(T(510, 104, "3. ReLU", "h2"))
    b.append(matrix(510, y0 + 10, rl, 24, 24, lambda i, j: "grey" if rl[i][j] == 0 else "cell", lambda i, j: "numf" if rl[i][j] == 0 else "num"))
    for i in (0, 2, 4):
        for j in (0, 2, 4):
            b.append(R(510 + j * 24, y0 + 10 + i * 24, 24 * (2 if j < 4 else 1), 24 * (2 if i < 4 else 1), "lnd"))
    b.append(T(570, y0 + 152, "negatives become 0", "hand", "middle"))
    b.append(A(635, y0 + 70, 665, y0 + 70)); b.append(T(650, y0 + 60, "max", "mut", "middle"))
    b.append(T(675, 104, "4. pool 2×2", "h2"))
    b.append(matrix(675, y0 + 34, pl, 26, 26, lambda i, j: "grn" if pl[i][j] == 3 else "cell"))
    b.append(T(714, y0 + 130, "edge windows", "mut", "middle")); b.append(T(714, y0 + 144, "hang over", "mut", "middle"))
    b.append(A(760, y0 + 70, 790, y0 + 70))
    b.append(T(800, 104, "5. flatten", "h2"))
    flat = [v for row in pl for v in row]
    b.append(colvec(805, y0 - 4, flat, 26, 17, "cell"))
    for t, yy in ((0, 160), (1, 200), (2, 240)): b.append(C(880, yy, 10, "blu"))
    for i in range(9):
        for yy in (160, 200, 240): b.append(f'<line x1="831" y1="{y0 - 4 + i * 17 + 8}" x2="870" y2="{yy}" class="lnd" stroke-width="0.4"/>')
    b.append(C(925, 200, 11, "grn")); b.append(T(925, 230, "σ", "h2", "middle"))
    for yy in (160, 200, 240): b.append(f'<line x1="890" y1="{yy}" x2="914" y2="200" class="lnd" stroke-width="0.6"/>')
    b.append(T(880, 290, "6. dense", "h2", "middle"))
    # explanations
    b.append(note(30, 340, 290, 150, "Convolution = slide & score",
                  ["Lay the filter on a 3×3 patch,", "multiply cell by cell, add up.", "Big number = patch looks like", "the filter. Slide 1 pixel, repeat.", "7 − 3 + 1 = 5 cells per side."], "handk"))
    b.append(note(335, 340, 290, 150, "ReLU, then pooling",
                  ["ReLU keeps \"looks like it\" and", "drops \"looks unlike it\".", "Max pooling keeps the strongest", "score per 2×2 block: the stroke", "may shift a pixel, the 3 survives."], "handk"))
    b.append(note(640, 340, 290, 150, "Flatten + dense",
                  ["Stack every pooled map into one", "long vector. From here it is the", "ANN of Part 1: dense layers vote,", "sigmoid (2 classes) or softmax", "(many) turns votes into odds."], "handk"))
    b.append(note(30, 506, 900, 66, "Output size of a conv or pool layer",
                  ["out = floor((n + 2·padding − kernel) / stride) + 1      e.g. 64 | conv 3: 62 | pool 2: 31 | conv 3: 29 | pool 2: 14"], "num"))
    svg("cnn-steps.svg", 960, 590, "".join(b))

# ---------------- CNN 2: the course model, shapes and parameter counts ----------------
if want("cnn"):
    b = [header("CONVOLUTIONAL NETWORKS", "The course's cats-vs-dogs model, layer by layer", 960)]
    layers = [("input", "64×64×3", "0", "pur", 64, 3), ("Rescaling 1/255", "64×64×3", "0", "grey", 64, 3),
              ("Flip / Zoom", "train only", "0", "grey", 64, 3), ("Conv2D 32, 3×3", "62×62×32", "896", "yel", 62, 32),
              ("MaxPool 2", "31×31×32", "0", "blu", 31, 32), ("Conv2D 32, 3×3", "29×29×32", "9,248", "yel", 29, 32),
              ("MaxPool 2", "14×14×32", "0", "blu", 14, 32), ("Flatten", "6,272", "0", "cell", 0, 0),
              ("Dense 128 ReLU", "128", "802,944", "grn", 0, 0), ("Dense 1 sigmoid", "1", "129", "pnk", 0, 0)]
    x = 40
    for i, (name, shape, params, c, side, ch) in enumerate(layers):
        cx = 90 + i * 86
        if side:
            h = 30 + side * 1.6; d = min(ch, 32) * 0.9
            b.append(f'<path d="M{cx},{250 - h / 2} l{d},-{d * 0.6} l40,0 l0,{h} l-{d},{d * 0.6} z" class="{c}" opacity="0.6"/>')
            b.append(R(cx, 250 - h / 2, 40, h, c))
        else:
            h = {"6,272": 150, "128": 90, "1": 24}[shape]
            b.append(R(cx + 12, 250 - h / 2, 16, h, c, 3))
        b.append(T(cx + 20, 380, name.split(" ")[0], "h2", "middle"))
        b.append(T(cx + 20, 396, " ".join(name.split(" ")[1:]), "mut", "middle"))
        b.append(T(cx + 20, 416, shape, "num", "middle"))
        b.append(T(cx + 20, 434, params, "numb" if params != "0" else "numf", "middle"))
    b.append(T(70, 416, "shape", "mut", "end")); b.append(T(70, 434, "params", "mut", "end"))
    b.append(T(238, 118, "inside the model, so predict() can't skip them", "hand", "middle"))
    b.append(P("M196,126 L196,150 M282,126 L282,150", "lnr"))
    b.append(note(30, 456, 430, 110, "Where the weights are",
                  ["Total 813,217 parameters.", "Both conv layers together: 10,144 (1.2%).", "The first dense layer alone: 802,944 (98.7%).", "Pooling and flatten learn nothing."], "handk"))
    b.append(note(480, 456, 450, 110, "Why convolution is cheap",
                  ["A filter's 3×3×channels weights are reused at", "every position (weight sharing). A dense layer", "on the raw 64×64×3 image would need 12,288", "weights per neuron — and wouldn't know a shifted cat."], "handk"))
    svg("cnn-architecture.svg", 960, 580, "".join(b))

# ---------------- RNN 1: unrolled network + vanishing / exploding gradient ----------------
if want("rnn"):
    b = [header("RECURRENT NETWORKS", "One cell, reused at every step", 960)]
    b.append(T(40, 104, "1. folded", "h2"))
    b.append(C(110, 300, 16, "yel")); b.append(T(110, 304, "x", "ink", "middle"))
    b.append(C(110, 220, 18, "blu")); b.append(T(110, 224, "h", "ink", "middle"))
    b.append(C(110, 140, 16, "pnk")); b.append(T(110, 144, "y", "ink", "middle"))
    b.append(A(110, 284, 110, 240)); b.append(A(110, 202, 110, 158))
    b.append(P("M128,212 C 175,185 175,255 128,228", "lnr", "arr")); b.append(T(178, 224, "W_h", "hand"))
    b.append(T(90, 266, "W_x", "mut", "end")); b.append(T(90, 184, "W_y", "mut", "end"))
    b.append(T(110, 345, "memory loops back", "mut", "middle"))
    b.append(T(250, 230, "=", "h1", "middle"))
    b.append(T(300, 104, "2. unrolled over time", "h2"))
    xs = [340 + i * 120 for i in range(5)]
    for i, x in enumerate(xs):
        b.append(C(x, 300, 16, "yel")); b.append(T(x, 304, f"x{i + 1}", "num", "middle"))
        b.append(C(x, 220, 18, "blu")); b.append(T(x, 224, f"h{i + 1}", "num", "middle"))
        b.append(C(x, 140, 16, "pnk")); b.append(T(x, 144, f"y{i + 1}", "num", "middle"))
        b.append(A(x, 284, x, 240)); b.append(A(x, 202, x, 158))
        b.append(T(x + 6, 266, "W_x", "mut")); b.append(T(x + 6, 184, "W_y", "mut"))
        if i < 4: b.append(A(x + 18, 220, xs[i + 1] - 20, 220, "lnr", "arr")); b.append(T(x + 60, 212, "W_h", "hand", "middle"))
    b.append(T(xs[0] - 50, 224, "h0=0", "mut", "end")); b.append(A(xs[0] - 46, 220, xs[0] - 20, 220))
    b.append(T(580, 345, "same W_x, W_h, W_y in every column: 1 cell's weights, not 5", "hand", "middle"))
    b.append(T(580, 365, "h_t = tanh(W_x·x_t + W_h·h_(t-1) + b)", "num", "middle"))
    # gradient bars
    b.append(T(40, 405, "3. backprop through time: the error signal from step 10 reaching step 10 − k", "h2"))
    for r, (w, cls, lab) in enumerate([(0.5, "bar", "w_rec = 0.5  vanishes"), (1.5, "barr", "w_rec = 1.5  explodes")]):
        y0 = 470 + r * 70
        b.append(T(40, y0 - 18, lab, "handk"))
        for k in range(10):
            v = w ** k; h = min(40, 40 * v / (1 if w < 1 else 40))
            b.append(R(220 + k * 50, y0 + 10 - h, 30, max(h, 1), cls))
            b.append(T(235 + k * 50, y0 + 24, f"{v:.3g}", "num", "middle"))
    b.append(T(480, 450, "k = 0 (the step itself)  …  k = 9", "mut", "middle"))
    b.append(note(740, 420, 200, 136, "Fixes", ["LSTM / GRU gates", "gradient clipping", "  (for exploding)", "careful init", "shorter windows"], "handk"))
    svg("rnn-unrolled.svg", 960, 580, "".join(b))

# ---------------- RNN 2: the LSTM cell ----------------
if want("rnn"):
    b = [header("RECURRENT NETWORKS", "The LSTM cell: a memory line with three valves", 960)]
    b.append(R(130, 110, 470, 300, "note", 12))
    b.append(T(60, 154, "c(t−1)", "num", "middle")); b.append(T(670, 154, "c(t)", "numb", "middle"))
    b.append(P("M90,150 L640,150", "lnr", "arr"))
    b.append(T(365, 132, "cell state: the memory line, changed only by × and +", "hand", "middle"))
    for x, s in ((220, "×"), (360, "+")):
        b.append(C(x, 150, 13, "cell")); b.append(T(x, 155, s, "h2", "middle"))
    gates = [(220, "f", "σ", "forget gate", "pnk"), (310, "i", "σ", "input gate", "grn"), (410, "g", "tanh", "candidate", "yel"), (500, "o", "σ", "output gate", "blu")]
    b.append(P("M60,370 L560,370", "ln")); b.append(T(60, 392, "[h(t−1), x(t)]", "num"))
    for x, n, act, lab, c in gates:
        b.append(R(x - 28, 280, 56, 34, c, 4)); b.append(T(x, 302, f"{n} = {act}", "num", "middle"))
        b.append(P(f"M{x},370 L{x},316", "ln", "ar")); b.append(T(x, 390, lab, "mut", "middle"))
    b.append(P("M220,278 L220,165", "ln", "ar"))
    b.append(C(360, 220, 12, "cell")); b.append(T(360, 225, "×", "h2", "middle"))
    b.append(P("M310,278 L310,220 L346,220", "ln", "ar")); b.append(P("M410,278 L410,220 L374,220", "ln", "ar"))
    b.append(P("M360,208 L360,165", "ln", "ar"))
    b.append(P("M560,150 L560,200", "ln", "ar")); b.append(R(532, 200, 56, 26, "yel", 4)); b.append(T(560, 218, "tanh", "num", "middle"))
    b.append(C(560, 252, 12, "cell")); b.append(T(560, 257, "×", "h2", "middle"))
    b.append(P("M560,226 L560,238", "ln", "ar")); b.append(P("M500,278 L500,252 L546,252", "ln", "ar"))
    b.append(P("M572,252 L640,252 L640,340 L680,340", "ln", "ar")); b.append(T(690, 344, "h(t): output + next step", "numb"))
    b.append(T(130, 440, "σ outputs 0…1 = how far each valve is open;  tanh outputs −1…1 = the content", "handk"))
    eq = ["f = σ(W_f·[h(t−1), x(t)] + b_f)", "i = σ(W_i·[h(t−1), x(t)] + b_i)", "g = tanh(W_g·[h(t−1), x(t)] + b_g)",
          "o = σ(W_o·[h(t−1), x(t)] + b_o)", "c(t) = f * c(t−1) + i * g", "h(t) = o * tanh(c(t))"]
    b.append(note(30, 460, 440, 150, "The six equations", [], "num"))
    for k, e in enumerate(eq): b.append(T(44, 502 + k * 17, e, "num"))
    b.append(note(490, 460, 440, 150, "Why gradients survive",
                  ["Along the memory line, c(t) depends on c(t−1)", "through a multiply by f, not by a weight matrix", "and tanh. With f near 1, the error signal flows", "back many steps almost undamped.", "GRU: 2 gates, no separate c. Peephole: gates also see c."], "handk"))
    svg("lstm-cell.svg", 960, 630, "".join(b))

# ---------------- SOM 1: weights are positions; BMU pulls its map-neighbours ----------------
if want("som"):
    b = [header("SELF-ORGANIZING MAPS", "Nodes are points in data space that get pulled", 960)]
    # panel 1: the network view
    b.append(T(40, 104, "1. three inputs, nine nodes", "h2"))
    for k, lab in enumerate(["x1", "x2", "x3"]):
        b.append(C(70, 170 + k * 60, 14, "yel")); b.append(T(70, 174 + k * 60, lab, "num", "middle"))
    for r in range(3):
        for c in range(3):
            x, y = 200 + c * 40, 170 + r * 60
            b.append(C(x, y, 12, "pnk" if (r, c) == (1, 1) else "blu"))
    for k in range(3):
        b.append(f'<line x1="84" y1="{170 + k * 60}" x2="188" y2="230" class="lnd" stroke-width="0.8"/>')
    b.append(T(150, 318, "node 5 stores w = (0.2, 0.7, 0.4)", "num", "middle"))
    b.append(T(150, 338, "a weight vector is a POSITION, not a multiplier", "hand", "middle"))
    b.append(T(150, 356, "no activation function, no output value", "mut", "middle"))
    # panel 2: input space with map drawn in it
    b.append(T(340, 104, "2. one update, seen in data space", "h2"))
    b.append(R(340, 120, 300, 230, "cell", 4))
    pts = {}
    for r in range(4):
        for c in range(4):
            pts[(r, c)] = (380 + c * 60 + (r * 7) % 13, 150 + r * 52 + (c * 11) % 9)
    for (r, c), (x, y) in pts.items():
        if c < 3: b.append(f'<line x1="{x}" y1="{y}" x2="{pts[(r, c + 1)][0]}" y2="{pts[(r, c + 1)][1]}" class="ln" stroke-width="0.8"/>')
        if r < 3: b.append(f'<line x1="{x}" y1="{y}" x2="{pts[(r + 1, c)][0]}" y2="{pts[(r + 1, c)][1]}" class="ln" stroke-width="0.8"/>')
    sx, sy = 560, 300
    bmu = (2, 2)
    for (r, c), (x, y) in pts.items():
        dgrid = max(abs(r - bmu[0]), abs(c - bmu[1]))
        frac = {0: 0.5, 1: 0.25}.get(dgrid, 0)
        b.append(C(x, y, 7, "pnk" if dgrid == 0 else ("grn" if dgrid == 1 else "blu")))
        if frac: b.append(A(x, y, x + (sx - x) * frac, y + (sy - y) * frac, "lnr", "arr"))
    b.append(C(sx, sy, 6, "yel")); b.append(T(sx + 12, sy + 26, "row x", "handk"))
    b.append(T(500, 340, "red = BMU (closest node), moves most", "mut", "middle"))
    b.append(note(660, 96, 270, 254, "The update rule",
                  ["1. pick a row x", "2. BMU = node nearest x", "3. for every node j:", "   w_j += lr · h(j) · (x − w_j)", "h = 1 at the BMU, smaller for", "nodes farther away ON THE MAP", "(green: 1 step away, pulled less;", "blue: outside the radius)", "", "no target, no loss, no backprop"], "handk"))
    # panel 3: shrinking radius
    b.append(T(40, 392, "3. the radius shrinks: first unfold, then fine-tune", "h2"))
    for k, (rad, lab) in enumerate([(3, "early: big neighbourhood"), (2, "middle"), (1, "late: almost just the BMU")]):
        x0 = 60 + k * 300
        for r in range(7):
            for c in range(7):
                d = max(abs(r - 3), abs(c - 3))
                b.append(R(x0 + c * 18, 410 + r * 18, 16, 16, "pnk" if d == 0 else ("grn" if d <= rad - 1 or (rad == 1 and d == 0) else "cell"), 2))
        b.append(T(x0 + 63, 556, lab, "mut", "middle"))
        if k < 2: b.append(A(x0 + 150, 473, x0 + 250, 473))
    b.append(T(800, 440, "same picture on every pass:", "handk")); b.append(T(800, 460, "the map keeps neighbours", "handk"))
    b.append(T(800, 480, "near neighbours, so similar", "handk")); b.append(T(800, 500, "rows land in nearby cells", "handk"))
    b.append(T(800, 520, "(topology preserved)", "hand"))
    svg("som-learning.svg", 960, 580, "".join(b))

# ---------------- SOM 2: U-matrix and why the flags are pseudo-labels ----------------
if want("som"):
    U = [[0.94, 0.93, 0.54, 0.62, 0.68, 0.77, 0.62, 0.53, 0.54, 0.63], [0.8, 0.71, 0.6, 0.58, 1.0, 0.57, 0.56, 0.64, 0.52, 0.53], [0.71, 0.69, 0.92, 0.69, 0.53, 0.58, 0.55, 0.54, 0.56, 0.49], [0.76, 0.75, 0.87, 0.75, 0.75, 0.75, 0.63, 0.54, 0.46, 0.53], [0.72, 0.74, 0.74, 0.9, 0.74, 0.75, 0.62, 0.58, 0.56, 0.56], [0.57, 0.59, 0.74, 0.66, 0.63, 0.59, 0.64, 0.61, 0.64, 0.58], [0.57, 0.63, 0.69, 0.56, 0.59, 0.75, 0.56, 0.67, 0.6, 0.62], [0.6, 0.8, 0.63, 0.57, 0.57, 0.53, 0.61, 0.7, 0.67, 0.69], [0.64, 0.76, 0.75, 0.56, 0.57, 0.51, 0.62, 0.69, 0.8, 0.63], [0.6, 0.76, 0.8, 0.59, 0.49, 0.59, 0.59, 0.88, 0.72, 0.94]]
    N = [[2, 1, 8, 13, 4, 4, 8, 4, 8, 8], [7, 4, 8, 6, 1, 9, 7, 9, 12, 7], [8, 12, 1, 13, 7, 13, 12, 8, 9, 5], [4, 1, 5, 3, 4, 9, 5, 4, 9, 9], [5, 6, 3, 3, 5, 4, 2, 12, 9, 5], [7, 7, 6, 6, 4, 14, 12, 6, 10, 6], [7, 5, 5, 9, 6, 2, 11, 7, 14, 3], [4, 5, 11, 10, 8, 9, 8, 6, 12, 2], [9, 7, 3, 5, 6, 9, 9, 13, 8, 5], [10, 6, 2, 10, 10, 10, 5, 1, 14, 1]]
    Fr = [[1, 1, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]]
    flagged = {0, 1, 14, 22, 32, 43, 71, 97, 99}
    b = [header("SELF-ORGANIZING MAPS", "Reading the U-matrix: white = far from neighbours", 960)]
    b.append(T(40, 104, "the map from §4 (number = rows in the cell)", "h2"))
    x0, y0, cs = 40, 120, 40
    for r in range(10):
        for c in range(10):
            g = int(30 + 225 * (U[r][c] - 0.45) / 0.55)
            b.append(f'<rect x="{x0 + c * cs}" y="{y0 + r * cs}" width="{cs}" height="{cs}" fill="rgb({g},{g},{min(255, g + 12)})" stroke="#8a8174" stroke-width="0.5"/>')
            b.append(f'<text x="{x0 + c * cs + cs / 2}" y="{y0 + r * cs + cs / 2 + 4}" text-anchor="middle" font-size="11" font-family="ui-monospace, Menlo, monospace" fill="{"#222" if g > 140 else "#eee"}">{N[r][c]}</text>')
            if Fr[r][c]: b.append(f'<circle cx="{x0 + c * cs + cs - 7}" cy="{y0 + r * cs + 7}" r="3.5" fill="#d93025"/>')
            if r * 10 + c in flagged: b.append(R(x0 + c * cs + 2, y0 + r * cs + 2, cs - 4, cs - 4, "hi"))
    for k in range(6):
        g = int(30 + 225 * k / 5)
        b.append(f'<rect x="{x0 + 410}" y="{y0 + 300 - k * 60}" width="16" height="60" fill="rgb({g},{g},{min(255, g + 12)})"/>')
    b.append(T(x0 + 418, y0 + 2, "far", "mut", "middle")); b.append(T(x0 + 418, y0 + 376, "close", "mut", "middle"))
    b.append(note(510, 96, 420, 140, "How a cell's shade is computed",
                  ["Take the node's weight vector, measure the distance", "to each of its (up to 8) grid neighbours, average,", "divide by the map's largest average. Dense regions", "of data: nodes packed close, dark. Gaps between", "clusters, or lonely rows: nodes far apart, white."], "handk"))
    b.append(note(510, 252, 420, 120, "What the red marks show",
                  ["red outline = the 9 whitest cells, 20 rows flagged", "red dot = cell holds a planted fraud (the truth", "the course never has). Flags: 8 of 20 right (40%),", "vs 2.9% at random. Not bad — but not fraud labels."], "handk"))
    b.append(note(510, 388, 420, 130, "Then the course trains an ANN on the flags",
                  ["Its labels are the SOM's guesses, so its accuracy", "only measures agreement with the SOM (97.1% =", "the score of \"never fraud\"). Its ranking found 2", "true frauds in its top 20: worse than the flags."], "handk"))
    b.append(T(40, 548, "white ≠ fraud: corner and edge nodes have fewer neighbours, and border cells between two clusters are white too", "hand"))
    svg("som-umatrix.svg", 960, 570, "".join(b))

# ---------------- ANN annex: least squares + logistic regression ----------------
if want("annex"):
    import math
    b = [header("ML BASICS (ANNEX)", "A line for numbers, an S-curve for yes/no", 960)]
    # left: OLS residuals. hours 1..9 -> score
    pts = [(1, 47), (2, 49), (3, 58), (4, 57), (5, 68), (6, 66), (7, 78), (8, 76), (9, 88)]
    n = len(pts); mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
    b1 = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts); b0 = my - b1 * mx
    X = lambda h: 70 + h * 38; Y = lambda s: 420 - (s - 30) * 4.6
    b.append(T(40, 104, "1. linear regression: minimise the squared gaps", "h2"))
    b.append(P(f"M{X(0)},{Y(30)} L{X(10)},{Y(30)} M{X(0)},{Y(30)} L{X(0)},{Y(95)}", "ln"))
    b.append(T(X(10), Y(30) + 18, "hours studied", "mut", "end")); b.append(T(X(0) + 6, Y(95) + 4, "score", "mut"))
    b.append(P(f"M{X(0)},{Y(b0)} L{X(10)},{Y(b0 + 10 * b1)}", "lnr"))
    for x, y in pts:
        yh = b0 + b1 * x
        b.append(f'<line x1="{X(x)}" y1="{Y(y)}" x2="{X(x)}" y2="{Y(yh)}" class="ln" stroke="#5f8a3a" stroke-width="2"/>')
        b.append(C(X(x), Y(y), 4.5, "pnk")); b.append(C(X(x), Y(yh), 3, "grn"))
    xs, ys = pts[4]
    b.append(T(X(xs) - 10, Y(ys) - 4, "y_i (actual)", "handk", "end")); b.append(T(X(xs) + 10, Y(b0 + b1 * xs) + 18, "ŷ_i (on the line)", "handk"))
    b.append(T(X(0) + 8, Y(b0) - 8, f"b0 = {b0:.1f}: score at 0 hours", "hand"))
    b.append(T(X(5.6), Y(b0 + b1 * 9.6) - 8, f"b1 = {b1:.2f} points per hour", "hand", "middle"))
    b.append(note(40, 450, 420, 106, "Ordinary least squares",
                  ["Of all straight lines, keep the one with the", "smallest sum of (y_i − ŷ_i)². Closed form:", "b1 = cov(x, y) / var(x),  b0 = ȳ − b1·x̄"], "handk"))
    # right: logistic
    X2 = lambda d: 540 + d * 6.2; Y2 = lambda p: 380 - p * 240
    b.append(T(510, 104, "2. logistic regression: predict a probability", "h2"))
    b.append(P(f"M{X2(0)},{Y2(0)} L{X2(62)},{Y2(0)} M{X2(0)},{Y2(-0.15)} L{X2(0)},{Y2(1.1)}", "ln"))
    for v in (0, 0.5, 1): b.append(T(X2(0) - 6, Y2(v) + 4, str(v), "num", "end"))
    b.append(T(X2(62), Y2(0) + 18, "days since last order", "mut", "end"))
    b.append(P(f"M{X2(0)},{Y2(0.5)} L{X2(62)},{Y2(0.5)}", "lnd"))
    b.append(T(X2(62), Y2(0.5) - 6, "threshold 0.5", "mut", "end"))
    for d in (3, 7, 10, 14, 18, 22, 26, 29, 40): b.append(C(X2(d), Y2(0), 4, "blu"))
    for d in (24, 31, 34, 38, 43, 47, 51, 55, 58): b.append(C(X2(d), Y2(1), 4, "pnk"))
    b.append(P(f"M{X2(0)},{Y2(-0.12)} L{X2(60)},{Y2(1.12)}", "lnd"))
    b.append(T(X2(1), Y2(-0.12) + 4, "straight line goes below 0 (and above 1)", "mut"))
    sig = " ".join(f"{'M' if i == 0 else 'L'}{X2(d):.1f},{Y2(1 / (1 + math.exp(-(d - 31) / 5))):.1f}" for i, d in enumerate(range(0, 61)))
    b.append(P(sig, "lnr"))
    for d, p in ((5, 0.008), (25, 0.238), (35, 0.666), (55, 0.988)):
        yy = Y2(1 / (1 + math.exp(-(d - 31) / 5)))
        b.append(f'<line x1="{X2(d)}" y1="{Y2(0)}" x2="{X2(d)}" y2="{yy:.1f}" class="lnd"/>'); b.append(C(X2(d), yy, 4, "yel"))
        b.append(T(X2(d), yy - 10, f"{p:.0%}" if p >= 0.01 else "0.8%", "numb", "middle"))
    b.append(note(510, 450, 420, 106, "Two outputs from one curve",
                  ["p̂ = σ(b0 + b1·x) is between 0 and 1: a score you", "can rank customers by. ŷ = 1 if p̂ > threshold:", "0.5 by default, move it to trade misses for false alarms."], "handk"))
    svg("annex-regression.svg", 960, 580, "".join(b))

# ---------------- NLP 1: text classification pipeline ----------------
if want("textcls"):
    b = [header("TEXT CLASSIFICATION", "Ids, vectors, a CNN or RNN body, one sigmoid per label", 960)]
    stages = [("text", "\"you are an idiot\"", "cell"), ("ids", "[12, 5, 33, 871, 0, 0]", "grey"), ("embeddings", "T x D (lookup rows of V x D)", "yel"),
              ("body", "Conv1D or (Bi)LSTM", "blu"), ("global max pool", "strongest signal over time", "grn")]
    for k, (t1, t2, c) in enumerate(stages):
        x = 30 + k * 186
        b.append(R(x, 104, 166, 60, c, 6)); b.append(T(x + 83, 128, t1, "h2", "middle")); b.append(T(x + 83, 148, t2, "mut", "middle"))
        if k < 4: b.append(A(x + 166, 134, x + 186, 134))
    labels = ["toxic", "severe", "obscene", "threat", "insult", "identity"]
    for k, lab in enumerate(labels):
        x = 60 + k * 145
        b.append(R(x, 214, 120, 44, "pnk", 4)); b.append(T(x + 60, 234, lab, "numb", "middle")); b.append(T(x + 60, 250, "sigmoid: p(yes)", "mut", "middle"))
    b.append(P("M837,164 L837,190 L120,190 L120,212", "ln", "ar")); b.append(T(480, 186, "6 independent yes/no outputs (a comment can be several things at once)", "mut", "middle"))
    b.append(T(40, 296, "padding: 0 = no word; ids start at 1", "mut"))
    b.append(note(40, 316, 430, 150, "Loss and score", ["loss = mean binary cross-entropy over the 6 labels", "(softmax would force exactly one label)", "", "score = AUC per label, then average:", "accuracy is ~90% for 'always clean'"], "handk"))
    b.append(T(500, 326, "task shapes (input T x D)", "h2"))
    shapes = [("many-to-one", "sentiment, spam, toxicity", "T x D to K"), ("many-to-many, same T", "part-of-speech tags", "T x D to T x K"),
              ("many-to-many, different T", "translation (seq2seq)", "Tx x D to Ty x K"), ("one-to-many", "text / poetry generation", "1 to T x K")]
    for k, (a, e, s) in enumerate(shapes):
        y = 344 + k * 30
        b.append(T(500, y, a, "numb")); b.append(T(700, y, e, "mut")); b.append(T(930, y, s, "num", "end"))
    svg("text-classification.svg", 960, 490, "".join(b))

# ---------------- NLP 2: seq2seq ----------------
if want("seq2seq"):
    b = [header("SEQ2SEQ", "Encode the whole input into a state, decode one token at a time", 960)]
    src = ["how", "are", "you"]; tgt = ["<sos>", "cómo", "estás"]; outs = ["cómo", "estás", "<eos>"]
    for k, w in enumerate(src):
        x = 40 + k * 110
        b.append(R(x, 240, 90, 40, "yel", 4)); b.append(T(x + 45, 265, "LSTM", "numb", "middle"))
        b.append(A(x + 45, 316, x + 45, 282)); b.append(T(x + 45, 332, w, "num", "middle"))
        if k < 2: b.append(A(x + 90, 260, x + 110, 260))
    b.append(T(175, 220, "encoder (outputs thrown away)", "mut", "middle"))
    b.append(R(370, 236, 120, 48, "pnk", 6)); b.append(T(430, 256, "thought vector", "numb", "middle")); b.append(T(430, 274, "final h, c", "mut", "middle"))
    b.append(A(350, 260, 370, 260)); b.append(A(490, 260, 520, 260))
    for k in range(3):
        x = 520 + k * 140
        b.append(R(x, 240, 100, 40, "blu", 4)); b.append(T(x + 50, 265, "LSTM", "numb", "middle"))
        b.append(A(x + 50, 316, x + 50, 282)); b.append(T(x + 50, 332, tgt[k], "num", "middle"))
        b.append(A(x + 50, 240, x + 50, 200)); b.append(R(x, 160, 100, 40, "grn", 4)); b.append(T(x + 50, 185, outs[k], "numb", "middle"))
        if k < 2: b.append(A(x + 100, 260, x + 140, 260)); b.append(P(f"M{x + 100},180 C {x + 120},180 {x + 120},330 {x + 160},330", "lnd", "ar"))
    b.append(T(730, 140, "decoder: predicts the next word", "mut", "middle"))
    b.append(T(730, 360, "dashed: at inference, feed back the word just predicted", "hand", "middle"))
    b.append(note(40, 392, 430, 140, "Training: teacher forcing", ["decoder input  = <sos> cómo estás", "decoder target = cómo estás <eos>", "", "feed the TRUE previous word, not the guess:", "fast, stable, but never practises recovering"], "handk"))
    b.append(note(500, 392, 430, 140, "Inference: a loop", ["x = <sos>, state = encoder's final state", "repeat: predict, pick (argmax or sample),", "feed it back; stop at <eos> or max length", "", "needs its own one-step decoder model"], "handk"))
    b.append(note(40, 96, 300, 96, "Bidirectional encoder", ["read left-to-right AND right-to-left,", "concatenate: T x 2M", "only if the whole input is known"], "handk"))
    svg("seq2seq.svg", 960, 550, "".join(b))

# ---------------- NLP 3: attention ----------------
if want("attention"):
    b = [header("ATTENTION", "At every output step, look back at all input positions", 960)]
    words = ["how", "are", "you", "today"]; alph = [0.05, 0.1, 0.7, 0.15]
    for k, w in enumerate(words):
        x = 60 + k * 120
        b.append(R(x, 380, 90, 40, "yel", 4)); b.append(T(x + 45, 405, f"h{k + 1}", "numb", "middle")); b.append(T(x + 45, 440, w, "num", "middle"))
        b.append(R(x, 290, 90, 40, "grey", 4)); b.append(T(x + 45, 315, "score", "num", "middle")); b.append(A(x + 45, 380, x + 45, 332))
        b.append(R(x + 15, 210 - alph[k] * 80, 60, alph[k] * 80, "pnk", 2)); b.append(T(x + 45, 204 - alph[k] * 80, f"α={alph[k]}", "num", "middle")); b.append(A(x + 45, 290, x + 45, 214))
    b.append(T(60, 456, "bidirectional encoder: one vector per input word", "mut"))
    b.append(T(560, 322, "score = Dense(1)( tanh( Dense(10)( [s_(t−1); h_j] ) ) )", "num"))
    b.append(T(560, 340, "the same tiny network for every position j", "mut"))
    b.append(T(560, 196, "α = softmax over the input positions", "num")); b.append(T(560, 214, "(sums to 1 across time, not across features)", "mut"))
    b.append(R(560, 100, 170, 50, "grn", 6)); b.append(T(645, 122, "context_t", "numb", "middle")); b.append(T(645, 140, "Σ α_j · h_j", "num", "middle"))
    b.append(R(780, 100, 150, 50, "blu", 6)); b.append(T(855, 122, "decoder step t", "numb", "middle")); b.append(T(855, 140, "input: [prev word; context]", "mut", "middle"))
    b.append(A(730, 125, 780, 125)); b.append(P("M855,150 L855,290 L770,290", "lnd", "ar")); b.append(T(845, 230, "s_(t−1) goes into", "hand", "end")); b.append(T(845, 246, "every score", "hand", "end"))
    b.append(note(560, 380, 370, 110, "Why it helps", ["no single vector has to hold the whole input;", "cost: Tx scores at each of Ty steps.", "Transformers keep this idea and drop the RNN."], "handk"))
    svg("attention.svg", 960, 510, "".join(b))

# ---------------- NLP 4: memory network ----------------
if want("memnet"):
    b = [header("MEMORY NETWORKS", "Answer a question by attending over the sentences of a story", 960)]
    story = ["Mary moved to the bathroom.", "John went to the hallway.", "Mary travelled to the office."]
    ws = [0.08, 0.02, 0.9]
    for k, s in enumerate(story):
        y = 110 + k * 54
        b.append(R(40, y, 280, 40, "cell", 4)); b.append(T(52, y + 25, s, "num"))
        b.append(A(320, y + 20, 360, y + 20)); b.append(R(360, y, 140, 40, "yel", 4)); b.append(T(430, y + 25, "Σ word vectors", "num", "middle"))
        b.append(A(500, y + 20, 540, y + 20)); b.append(R(540, y + 4, ws[k] * 140 + 4, 32, "pnk", 2)); b.append(T(700, y + 25, f"weight {ws[k]}", "num"))
    b.append(R(40, 290, 280, 40, "blu", 4)); b.append(T(52, 315, "Q: Where is Mary?", "numb"))
    b.append(A(320, 310, 360, 310)); b.append(R(360, 290, 140, 40, "blu", 4)); b.append(T(430, 315, "question vector u", "num", "middle"))
    b.append(T(540, 300, "weight = softmax over sentences of (sentence · u)", "mut")); b.append(T(540, 318, "answer = softmax over vocab ( Σ weight · sentence )", "mut"))
    b.append(R(800, 222, 130, 40, "grn", 6)); b.append(T(865, 247, "answer: office", "numb", "middle")); b.append(A(760, 236, 800, 240))
    b.append(note(40, 356, 430, 150, "Two supporting facts: two hops", ["Q: Where is the football?", "hop 1 finds 'John picked up the football'", "hop 2 is queried by hop 1's output:", "finds 'John went to the garden'", "(separate embeddings per hop; ELU dense layer)"], "handk"))
    b.append(note(500, 356, 430, 150, "The catch: order", ["summing word vectors loses word order;", "nothing says which sentence is LATEST.", "Add a learned vector per sentence slot,", "or (the course) the line number as a word."], "handk"))
    svg("memory-network.svg", 960, 530, "".join(b))
