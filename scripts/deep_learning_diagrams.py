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

# ---------------- Neural style transfer ----------------
if want("style"):
    b = [header("NEURAL STYLE TRANSFER", "Freeze the network, make the image the variable", 960)]
    thumbs = [("content image x", "the photo", 110, "cell"), ("style image s", "the painting", 196, "yel"),
              ("generated x'", "THE VARIABLE", 282, "pnk")]
    for t1, t2, y, c in thumbs:
        b.append(R(30, y, 150, 62, c, 6)); b.append(T(105, y + 26, t1, "numb", "middle")); b.append(T(105, y + 46, t2, "mut", "middle"))
        b.append(A(180, y + 31, 236, 240))
    b.append(R(236, 150, 150, 200, "blu", 8))
    b.append(T(311, 174, "VGG16", "h2", "middle")); b.append(T(311, 194, "weights FROZEN", "mut", "middle"))
    b.append(T(311, 212, "avg pooling", "mut", "middle"))
    for k in range(5):
        b.append(R(252, 230 + k * 23, 118, 18, "cell", 2)); b.append(T(311, 243 + k * 23, f"block{k + 1}", "num", "middle"))
    b.append(A(386, 250, 440, 190)); b.append(T(410, 178, "one tap", "mut", "middle"))
    b.append(R(440, 160, 150, 56, "grn", 6)); b.append(T(515, 182, "MSE on activations", "num", "middle")); b.append(T(515, 202, "J_content", "numb", "middle"))
    b.append(T(600, 182, "deeper tap ->", "mut")); b.append(T(600, 198, "blurrier, more abstract", "mut"))
    for k in range(5):
        y = 236 + k * 23
        b.append(A(386, y + 9, 440, y + 9))
        b.append(R(440, y, 74, 18, "yel", 2)); b.append(T(477, y + 13, "Gram CxC", "num", "middle"))
        b.append(R(520, y, 50, 18, "cell", 2)); b.append(T(545, y + 13, "MSE", "num", "middle"))
        b.append(A(570, y + 9, 596, 300))
    b.append(T(600, 352, "spatial axis cancels: H x W x C -> C x C", "mut"))
    b.append(R(596, 282, 150, 40, "grn", 6)); b.append(T(671, 300, "sum w_L * J_style_L", "num", "middle")); b.append(T(671, 316, "J_style", "numb", "middle"))
    b.append(A(590, 190, 790, 215)); b.append(A(746, 300, 790, 250))
    b.append(R(790, 205, 140, 50, "pnk", 6)); b.append(T(860, 226, "J = J_content", "numb", "middle")); b.append(T(860, 244, "+ J_style", "numb", "middle"))
    b.append(P("M860,205 L860,104 L16,104 L16,313 L28,313", "lnr", "ar"))
    b.append(T(470, 96, "dJ/dx' : L-BFGS updates PIXELS, not weights", "hand", "middle"))
    b.append(note(30, 390, 900, 76, "The one idea",
                  ["The gradient arrow comes back to the image and never touches the network. Ordinary training is argmin over W with the data fixed;",
                   "here it is argmin over x' with the weights fixed. No dataset, no training run, no checkpoint - just one image walked downhill."], "handk"))
    svg("style-transfer-architecture.svg", 960, 480, "".join(b))

# ---------------- PyTorch training step ----------------
if want("ptstep"):
    b = [header("ONE PYTORCH TRAINING STEP", "Gradients are additive state living on the parameters", 960)]
    steps = ["opt.zero_grad()", "out = model(xb)", "loss = lossf(out, yb)", "loss.backward()", "opt.step()"]
    grads = ["None", "None", "None", "filled: ADDS to", "unchanged:"]
    grads2 = ["", "", "", "whatever was there", "step only reads it"]
    for k, (s, g, g2) in enumerate(zip(steps, grads, grads2)):
        x = 30 + k * 186
        b.append(R(x, 150, 170, 54, "blu", 6)); b.append(T(x + 85, 172, f"{k + 1}", "mut", "middle")); b.append(T(x + 85, 192, s, "numb", "middle"))
        if k < 4: b.append(A(x + 170, 177, x + 186, 177))
        b.append(R(x, 250, 170, 50, "yel" if k == 3 else "cell", 4))
        b.append(T(x + 85, 270, g, "num", "middle"))
        if g2: b.append(T(x + 85, 288, g2, "mut", "middle"))
    b.append(T(30, 240, "p.grad for each parameter p:", "mut"))
    b.append(P("M216,144 L216,128 L588,128 L588,144", "ln"))
    b.append(T(402, 120, "graph is built here; torch.no_grad() suppresses it", "hand", "middle"))
    b.append(T(675, 140, "graph is freed after backward", "mut", "middle"))
    b.append(T(675, 126, "(retain_graph=True keeps it)", "mut", "middle"))
    b.append(R(30, 322, 900, 44, "grey", 6))
    b.append(T(480, 342, "evaluation: model.eval() + torch.no_grad(), and steps 1, 4 and 5 are skipped entirely", "numb", "middle"))
    b.append(T(480, 358, "model.eval() only changes dropout and batch norm; no_grad() is what stops the graph being built", "mut", "middle"))
    b.append(note(30, 382, 900, 76, "Why zero_grad() is not politeness",
                  ["backward() ADDS this batch's gradient to whatever p.grad already holds. Skip step 1 and you train on the sum of every batch so far.",
                   "Since PyTorch 2.x, zero_grad() sets p.grad to None rather than to a zero tensor - cheaper, but code that reads p.grad must handle None."], "handk"))
    svg("pytorch-training-step.svg", 960, 472, "".join(b))
# ---------------- VAE forward pass ----------------
if want("vae"):
    b = [header("VARIATIONAL AUTOENCODERS", "The bottleneck holds parameters, not a code", 960)]
    DIG = ["..####..", ".##..##.", "......##", "...###..", "......##", ".##..##.", "..####..", "........"]

    def dtile(x, y, s, noise=0):
        o = []
        for i, row in enumerate(DIG):
            for j, ch in enumerate(row):
                on = ch == "#"
                if noise and (i * 7 + j * 5) % noise == 0:
                    on = not on
                o.append(R(x + j * s, y + i * s, s, s, "grey" if on else "cell"))
        return "".join(o)

    # --- Panel A: posterior predictive ---
    b.append(T(30, 108, "A. posterior predictive (reconstruction)", "h2"))
    b.append(dtile(30, 164, 8))
    b.append(T(62, 246, "x (D pixels)", "mut", "middle"))
    b.append(A(94, 196, 114, 196))
    b.append(P("M118,162 L188,180 L188,212 L118,230 Z", "cell"))
    b.append(T(153, 200, "encoder", "num", "middle"))
    b.append(T(153, 246, "encoder (ReLU)", "mut", "middle"))
    b.append(A(190, 196, 212, 196))
    # split head: one dense layer of width 2M, cut in half
    b.append(R(216, 160, 148, 36, "blu"))
    b.append(T(290, 176, "mu (M)", "numb", "middle"))
    b.append(T(290, 190, "no activation", "mut", "middle"))
    b.append(R(216, 196, 148, 36, "yel"))
    b.append(T(290, 212, "sigma (M)", "numb", "middle"))
    b.append(T(290, 226, "softplus(.) + 1e-6 > 0", "mut", "middle"))
    b.append(P("M216,152 L216,144 L364,144 L364,152", "ln"))
    b.append(T(290, 138, "width 2M, then split", "mut", "middle"))
    b.append(A(364, 178, 402, 190))
    b.append(A(364, 214, 402, 202))
    b.append(C(428, 196, 24, "pnk"))
    b.append(T(428, 201, "z", "numb", "middle"))
    b.append(T(428, 246, "z ~ q(z|x)", "mut", "middle"))
    b.append(T(428, 272, "z = mu + sigma * eps,  eps ~ N(0, I)", "hand", "middle"))
    b.append(A(452, 196, 478, 196))
    b.append(P("M482,180 L552,162 L552,230 L482,212 Z", "cell"))
    b.append(T(519, 200, "decoder", "num", "middle"))
    b.append(T(517, 246, "decoder (ReLU)", "mut", "middle"))
    b.append(A(554, 196, 578, 196))
    b.append(R(580, 168, 78, 56, "cell", 4))
    b.append(T(619, 200, "logits (D)", "numb", "middle"))
    b.append(A(658, 196, 682, 196))
    b.append(R(684, 164, 136, 64, "grn", 6))
    b.append(T(752, 188, "Bernoulli(p)", "numb", "middle"))
    b.append(T(752, 206, "p = sigmoid(logits)", "mut", "middle"))
    b.append(A(820, 186, 844, 178))
    b.append(A(820, 208, 844, 258))
    b.append(dtile(848, 150, 7))
    b.append(T(876, 220, "mean image = p", "mut", "middle"))
    b.append(dtile(848, 232, 7, noise=5))
    b.append(T(876, 302, "sample ~ Bern(p)", "mut", "middle"))

    # --- Panel B: prior predictive ---
    b.append(T(30, 338, "B. prior predictive (generation)", "h2"))
    b.append(R(30, 372, 120, 56, "note", 6))
    b.append(T(90, 396, "z ~ N(0, I)", "numb", "middle"))
    b.append(T(90, 414, "no encoder, no x", "mut", "middle"))
    b.append(A(150, 400, 178, 400))
    b.append(P("M182,384 L252,366 L252,434 L182,416 Z", "grey"))
    b.append(T(219, 404, "decoder", "num", "middle"))
    b.append(T(217, 450, "SAME weights, reused", "mut", "middle"))
    b.append(A(254, 400, 280, 400))
    b.append(R(284, 372, 136, 56, "grn", 6))
    b.append(T(352, 394, "Bernoulli(p)", "numb", "middle"))
    b.append(T(352, 412, "p = sigmoid(logits)", "mut", "middle"))
    b.append(A(420, 400, 450, 400))
    b.append(dtile(456, 372, 7))
    b.append(T(528, 394, "a digit, but you don't", "handk"))
    b.append(T(528, 414, "get to choose which", "handk"))
    b.append(note(30, 490, 900, 76, "The one idea",
                  ["The bottleneck holds no code: it holds the PARAMETERS (mu, sigma) of a distribution, and z is one draw from it.",
                   "Generation is the same decoder with the encoder deleted and the prior N(0, I) plugged in where q(z|x) sat."], "handk"))
    svg("vae-forward-pass.svg", 960, 590, "".join(b))

# ---------------- ELBO and the KL gap ----------------
if want("elbo"):
    b = [header("THE ELBO", "A computable bound, and the gap you never see", 960)]
    base, top, bar_t = 410, 140, 250
    b.append(A(120, base, 120, 120, "ln", "ar"))
    b.append(T(130, 104, "log-likelihood scale (higher = better fit)", "mut"))
    b.append(P(f"M120,{base} L860,{base}", "ln"))
    # the ceiling: log p(x), fixed
    b.append(P(f"M120,{top} L700,{top}", "lnd"))
    b.append(T(130, 132, "log p(x)   (fixed: does not depend on q)", "numb"))
    # the gap
    b.append(R(210, top + 2, 300, bar_t - top - 2, "note", 4))
    b.append(T(360, 176, "the gap must shrink", "hand", "middle"))
    b.append(A(250, 186, 250, 242, "lnr", "arr"))
    b.append(P("M520,142 L534,142 L534,248 L520,248", "ln"))
    b.append(T(546, 182, "KL(q(z|x) || p(z|x)) >= 0", "numb"))
    b.append(T(546, 200, "(not computable: needs p(z|x))", "mut"))
    b.append(T(546, 224, "-> q approximates the true posterior", "hand"))
    # the bar: the ELBO
    b.append(R(210, bar_t, 300, base - bar_t, "pur", 4))
    b.append(T(360, 290, "ELBO = E_q[log p(x|z)]", "numb", "middle"))
    b.append(T(360, 308, "- KL(q(z|x) || p(z))", "numb", "middle"))
    b.append(T(360, 334, "computable - this is the loss we code", "mut", "middle"))
    b.append(A(250, 394, 250, 352, "lnr", "arr"))
    b.append(T(360, 372, "training raises the ELBO", "hand", "middle"))
    b.append(T(120, base + 50,"log p(x) = ELBO + KL   ==>   log p(x) >= ELBO, hence 'evidence lower bound'", "numb"))
    b.append(note(30, 486, 900, 76, "The one idea",
                  ["log p(x) does not move when q moves, so every nat the ELBO gains is a nat the KL gap loses.",
                   "Maximising a bound you can compute is the same act as fitting the posterior you cannot compute."], "handk"))
    svg("elbo-gap.svg", 960, 586, "".join(b))

# ---------------- GAN: one discriminator, two losses ----------------
if want("gan"):
    b = [header("GENERATIVE ADVERSARIAL NETWORKS", "One discriminator, two losses", 960)]
    b.append(P("M510,96 L510,600", "lnd"))

    # --- LEFT: what is really happening ---
    b.append(T(30, 108, "What is really happening", "h2"))
    b.append(R(40, 132, 150, 44, "blu", 6))
    b.append(T(115, 154, "z ~ p(z)", "numb", "middle"))
    b.append(T(115, 170, "dim 100", "mut", "middle"))
    b.append(A(115, 176, 115, 194))
    b.append(R(40, 196, 150, 44, "cell", 6))
    b.append(T(115, 223, "Generator G", "h2", "middle"))
    b.append(A(115, 240, 115, 258))
    b.append(R(40, 260, 150, 44, "pnk", 6))
    b.append(T(115, 287, "x_hat = G(z)", "numb", "middle"))
    b.append(R(230, 260, 150, 44, "grn", 6))
    b.append(T(305, 281, "real x", "numb", "middle"))
    b.append(T(305, 297, "training batch", "mut", "middle"))
    b.append(A(115, 304, 160, 340))
    b.append(A(305, 304, 275, 340))
    b.append(R(120, 344, 200, 56, "yel", 6))
    b.append(T(220, 370, "Discriminator D", "h2", "middle"))
    b.append(T(220, 390, "same weights, called twice", "mut", "middle"))
    b.append(A(220, 400, 220, 418))
    b.append(C(220, 436, 16, "cell"))
    b.append(T(220, 441, "D", "numb", "middle"))
    b.append(T(244, 432, "D(.) = P(real)", "num"))
    b.append(T(244, 448, "one output node", "mut"))
    b.append(A(204, 450, 140, 474))
    b.append(A(236, 450, 340, 474))
    costs = [("J_D = -[log D(x) + log(1 - D(G(z)))]", "updates theta_D only", 478),
             ("J_G = -log D(G(z))", "updates theta_G only; gradient flows BACK through D", 544)]
    for eq, who, y in costs:
        b.append(R(40, y, 440, 48, "grey", 6))
        b.append(T(260, y + 22, eq, "numb", "middle"))
        b.append(T(260, y + 40, who, "mut", "middle"))
    # gradient path: out of J_G, up the left margin, into D, then on into G
    # (paper_svg has no dashed-red class; inline dasharray keeps the class's dark-mode colour)
    dash = ' style="stroke-dasharray:6 4"'
    b.append(f'<path d="M40,568 L34,568 L34,372 L112,372" class="lnr" marker-end="url(#arr)"{dash}/>')
    b.append(f'<path d="M34,372 L34,218 L36,218" class="lnr" marker-end="url(#arr)"{dash}/>')
    b.append(T(236, 160, "dashed red = gradient", "hand"))
    b.append(T(236, 180, "path: D is traversed", "hand"))
    b.append(T(236, 200, "but NOT updated", "hand"))

    # --- RIGHT: how Keras codes it ---
    b.append(T(530, 108, "How Keras codes it", "h2"))
    b.append(R(545, 130, 235, 272, "note", 8))
    b.append(T(662, 154, "combined_model", "h2", "middle"))
    stack = [("Input z", "", "cell", 172), ("generator", "trainable", "blu", 228),
             ("discriminator", "trainable = False", "grey", 284), ("output", "target = 1 (all real)", "grn", 340)]
    for name, sub, cls, y in stack:
        b.append(R(565, y, 195, 38, cls, 6))
        b.append(T(662, y + 18 if sub else y + 24, name, "numb", "middle"))
        if sub:
            b.append(T(662, y + 34, sub, "mut", "middle"))
        if y < 340:
            b.append(A(662, y + 38, 662, y + 54))
    b.append(P("M760,303 L796,303", "lnd", "ar"))
    b.append(T(864, 268, "the SAME object", "mut", "middle"))
    b.append(R(800, 284, 128, 38, "yel", 6))
    b.append(T(864, 302, "discriminator", "numb", "middle"))
    b.append(T(864, 318, "trainable = True", "mut", "middle"))
    for i, l in enumerate(["compiled EARLIER, so", "its own optimizer", "still updates it"]):
        b.append(T(864, 340 + i * 16, l, "mut", "middle"))
    b.append(R(530, 424, 400, 56, "grey", 6))
    b.append(T(730, 446, "compile D first, THEN set trainable = False,", "numb", "middle"))
    b.append(T(730, 464, "THEN build combined. Wrong order = broken GAN.", "numb", "middle"))
    b.append(note(30, 616, 900, 76, "The one idea",
                  ["There is only ONE discriminator. The two losses differ only in which parameters the optimizer may touch:",
                   "J_G backpropagates THROUGH D to reach G, so D stays in the graph while its own weights stay frozen."], "handk"))
    svg("gan-two-losses.svg", 960, 716, "".join(b))

# ---------------- GAN: stride 2 vs stride 1/2 ----------------
if want("gantconv"):
    b = [header("UPSAMPLING IN THE GENERATOR", "Stride 2 shrinks, stride 1/2 grows", 960)]
    b.append(P("M330,96 L330,400", "lnd"))
    b.append(P("M640,96 L640,400", "lnd"))

    def grid(x, y, n, s, cls="cell"):
        return "".join(R(x + j * s, y + i * s, s, s, cls) for i in range(n) for j in range(n))

    titles = [(30, "1. stride 2 (discriminator)", "ordinary strided convolution"),
              (340, "2. stride 1/2 (generator)", "fractionally strided convolution"),
              (650, "3. where the info lives", "depth-to-space tradeoff")]
    for x, t1, t2 in titles:
        b.append(T(x, 110, t1, "h2"))
        b.append(T(x, 128, t2, "mut"))

    # panel 1: 8x8 -> 4x4
    b.append(grid(40, 156, 8, 20, "grey"))
    b.append(R(40, 156, 100, 100, "hi"))
    b.append(T(120, 148, "input 8 x 8", "mut", "middle"))
    b.append(A(204, 236, 226, 236))
    b.append(grid(230, 196, 4, 20, "grey"))
    b.append(T(270, 188, "out 4 x 4", "mut", "middle"))
    b.append(T(120, 336, "5x5 kernel, hops 2 cells", "mut", "middle"))
    b.append(T(30, 358, "halves the spatial size", "numb"))

    # panel 2: 4x4 with inserted gaps (7x7 lattice) -> 8x8
    for i in range(7):
        for j in range(7):
            real = i % 2 == 0 and j % 2 == 0
            b.append(R(350 + j * 20, 156 + i * 20, 20, 20, "grey" if real else "note"))
    b.append(R(350, 156, 100, 100, "hi"))
    b.append(T(420, 148, "4 x 4 + inserted gaps", "mut", "middle"))
    b.append(A(494, 226, 516, 226))
    b.append(grid(520, 174, 8, 13, "grey"))
    b.append(T(572, 166, "out 8 x 8", "mut", "middle"))
    b.append(T(420, 316, "hollow = inserted gaps", "mut", "middle"))
    b.append(T(340, 358, "doubles the spatial size - still an", "numb"))
    b.append(T(340, 374, "ordinary convolution, NOT a deconv", "numb"))

    # panel 3: tall-thin -> wide-flat
    b.append(R(667, 170, 46, 150, "blu", 4))
    b.append(T(690, 148, "4 x 4 x 1024", "numb", "middle"))
    b.append(T(690, 336, "fat in channels", "mut", "middle"))
    b.append(A(720, 245, 794, 245))
    b.append(T(757, 235, "trade", "mut", "middle"))
    b.append(R(800, 218, 124, 52, "grn", 4))
    b.append(T(862, 248, "64 x 64 x 3", "numb", "middle"))
    b.append(T(862, 288, "wide in space", "mut", "middle"))
    b.append(T(650, 358, "each layer trades depth for extent;", "numb"))
    b.append(T(650, 374, "nothing is created from nothing", "numb"))
    b.append(note(30, 415, 900, 76, "The one idea",
                  ["Upsampling is not magic: a 'transposed' convolution is an ORDINARY convolution run over an input with",
                   "zeros inserted between its pixels - the generator unpacks information already packed into channels."], "handk"))
    svg("gan-transposed-conv.svg", 960, 515, "".join(b))
# ---------------- Transfer learning: surgery, modes, pipelines ----------------
if want("transfer"):
    b = [header("TRANSFER LEARNING", "Body plus head: what you chop, freeze and precompute", 960)]

    # ---- PANEL A: surgery -------------------------------------------------
    b.append(T(30, 104, "A. surgery", "h2"))
    b.append(R(45, 120, 120, 24, "cell", 4) + T(105, 136, "input image", "num", "middle"))
    b.append(A(105, 144, 105, 160))
    for k in range(5):
        y = 160 + k * 26
        b.append(R(45, y, 120, 22, "grey", 2) + T(105, y + 15, f"conv/pool {k + 1}", "num", "middle"))
        if k < 4:
            b.append(P(f"M105,{y + 22} L105,{y + 26}", "ln"))
    for i, s in enumerate(["pretrained on", "ImageNet;", "weights kept"]):
        b.append(T(175, 208 + i * 16, s, "mut"))
    # body output bypasses the discarded head, down the right margin
    b.append(P("M105,286 L105,296 L285,296 L285,436 L167,436", "ln", "ar"))
    b.append(T(175, 290, "features z", "mut"))
    b.append(P("M35,312 L275,312", "lnd"))
    b.append(T(35, 330, "chop here: include_top=False", "hand"))
    for i, s in enumerate(["Dense 4096", "Dense 4096", "Dense 1000"]):
        b.append(R(45, 344 + i * 22, 120, 18, "grey", 2) + T(105, 357 + i * 22, s, "num", "middle"))
    b.append(P("M45,344 L165,406", "lnr") + P("M165,344 L45,406", "lnr"))
    for i, s in enumerate(["ImageNet head:", "1000 classes -", "useless to you"]):
        b.append(T(175, 360 + i * 16, s, "mut"))
    b.append(R(45, 414, 120, 44, "grn", 4))
    b.append(T(105, 434, "your head", "num", "middle") + T(105, 450, "Dense(K, softmax)", "num", "middle"))
    b.append(T(30, 480, "a network = feature transformer", "mut") + T(30, 496, "+ a linear model", "mut"))

    # ---- PANEL B: two modes -----------------------------------------------
    b.append(T(300, 104, "B. two modes", "h2"))
    modes = [(140, "1. feature extractor (body FROZEN)", "yel", "FROZEN",
              "gradients stop here", "seconds to train; safe on small data"),
             (300, "2. fine-tune (body TRAINABLE)", "grn", "TRAINABLE",
              "gradients reach the body", "better ceiling; low LR, top blocks only")]
    for base, title, strip, stripw, glab, cap in modes:
        b.append(T(300, base, title, "numb"))
        for j in range(3):
            b.append(R(305 + j * 36, base + 12, 32, 34, "grey", 2)
                     + T(321 + j * 36, base + 33, f"b{j + 1}", "num", "middle"))
        b.append(A(411, base + 29, 423, base + 29))
        b.append(R(425, base + 12, 56, 34, "grn", 4) + T(453, base + 33, "head", "num", "middle"))
        b.append(A(483, base + 29, 495, base + 29))
        b.append(R(497, base + 12, 56, 34, "pnk", 4) + T(525, base + 33, "loss", "num", "middle"))
        b.append(R(305, base + 50, 104, 16, strip, 2) + T(357, base + 62, stripw, "num", "middle"))
        if strip == "yel":
            b.append(P(f"M525,{base + 46} L525,{base + 78} L459,{base + 78}", "lnr", "arr"))
            b.append(P(f"M451,{base + 68} L451,{base + 88}", "lnr"))
        else:
            b.append(P(f"M525,{base + 46} L525,{base + 78} L309,{base + 78} L309,{base + 68}", "lnr", "arr"))
        b.append(T(305, base + 100, glab, "hand"))
        b.append(T(300, base + 120, cap, "mut"))

    # ---- PANEL C: two pipelines -------------------------------------------
    b.append(T(600, 104, "C. two pipelines", "h2"))
    b.append(R(600, 130, 330, 185, "cell", 6) + T(614, 152, "augment in the loop", "h2"))
    loop = ["images (batch)", "random flip / zoom", "frozen body -> z", "head -> loss"]
    for k, s in enumerate(loop):
        y = 166 + k * 26
        b.append(R(660, y, 190, 22, "blu", 3) + T(755, y + 15, s, "num", "middle"))
        if k < 3:
            b.append(A(755, y + 22, 755, y + 26))
    b.append(P("M655,255 L625,255 L625,177 L655,177", "lnr", "arr"))
    b.append(T(614, 286, "the cycle runs every batch, every epoch", "mut"))
    b.append(T(614, 302, "slow: body re-run each epoch; augmentation works", "mut"))

    b.append(R(600, 335, 330, 200, "cell", 6) + T(614, 357, "precompute Z", "h2"))
    b.append(T(614, 375, "once, up front:", "mut"))
    b.append(R(614, 382, 96, 24, "grey", 3) + T(662, 398, "all images", "num", "middle"))
    b.append(A(712, 394, 726, 394))
    b.append(R(728, 382, 96, 24, "grey", 3) + T(776, 398, "frozen body", "num", "middle"))
    b.append(A(826, 394, 840, 394))
    b.append(R(846, 382, 60, 78, "blu", 3))
    for i in range(1, 4):
        b.append(P(f"M846,{382 + i * 19} L906,{382 + i * 19}", "lnd"))
    b.append(T(876, 474, "Z: N x D", "num", "middle"))
    b.append(T(614, 420, "body never touched again", "mut") + T(614, 436, "(fast), one pass only", "mut"))
    b.append(R(614, 470, 96, 24, "grn", 3) + T(662, 486, "Z (fixed)", "num", "middle"))
    b.append(A(712, 482, 726, 482))
    b.append(R(728, 470, 96, 24, "grn", 3) + T(776, 486, "logistic reg", "num", "middle"))
    b.append(P("M824,494 L832,494 L832,510 L618,510 L618,496", "lnr", "arr"))
    b.append(T(614, 526, "cycle touches only Z: no images to augment", "mut"))

    b.append(note(30, 552, 900, 76, "The one idea",
                  ["Freezing is a choice about GRADIENTS; precomputing Z is a choice about COMPUTE - two independent knobs.",
                   "Augmentation is what forces the body to stay inside the loop, which is why the fast path has to give it up."], "handk"))
    svg("transfer-learning.svg", 960, 645, "".join(b))

# ---------------- ResNet: identity block vs convolutional block ----------------

if want("resnet"):
    b = [header("RESNET BLOCKS", "Same main branch, two kinds of shortcut", 960)]
    rows_tail = ["BN", "ReLU", "Conv 3x3, f2", "BN", "ReLU", "Conv 1x1, f3", "BN"]

    def main_branch(cx, first):
        out = [C(cx, 120, 11, "pnk"), T(cx, 125, "x", "numb", "middle"), A(cx, 131, cx, 140)]
        for k, s in enumerate([first] + rows_tail):
            y = 142 + k * 24
            out.append(R(cx - 60, y, 120, 18, "cell", 2))
            out.append(T(cx, y + 13, s, "num", "middle"))
            if k < 7:
                out.append(P(f"M{cx},{y + 18} L{cx},{y + 24}", "ln"))
        out += [A(cx, 328, cx, 338), C(cx, 352, 13, "pnk"), T(cx, 357, "+", "numb", "middle"),
                A(cx, 365, cx, 375), R(cx - 60, 375, 120, 18, "grn", 2), T(cx, 388, "ReLU", "num", "middle"),
                A(cx, 393, cx, 403), C(cx, 414, 11, "grn"), T(cx, 419, "out", "num", "middle")]
        return "".join(out)

    # ---- PANEL A: identity block ----
    b.append(T(30, 104, "A. identity block", "h2"))
    b.append(main_branch(220, "Conv 1x1, f1"))
    b.append(P("M231,120 L300,120 L300,352 L235,352", "ln", "ar"))
    b.append(T(312, 228, "shortcut: a plain wire,", "mut") + T(312, 244, "0 parameters, 0 cost", "mut"))
    for i, s in enumerate(["no ReLU before the", "add - activation", "comes AFTER it"]):
        b.append(T(312, 300 + i * 18, s, "hand"))
    b.append(T(30, 344, "out = f(x) + x", "numb"))
    for i, s in enumerate(["f learns only the", "RESIDUAL; f(x)=0", "recovers identity"]):
        b.append(T(30, 362 + i * 16, s, "mut"))
    b.append(A(150, 352, 205, 352, "lnd", "ar"))

    b.append(P("M478,96 L478,440", "lnd"))

    # ---- PANEL B: convolutional block ----
    b.append(T(500, 104, "B. convolutional block", "h2"))
    b.append(main_branch(690, "Conv 1x1, str s"))
    b.append(P("M701,120 L855,120 L855,196", "ln", "ar"))
    b.append(R(800, 196, 110, 22, "yel", 2) + T(855, 211, "Conv 1x1, str s", "num", "middle"))
    b.append(A(855, 218, 855, 248))
    b.append(R(800, 248, 110, 22, "yel", 2) + T(855, 263, "BN", "num", "middle"))
    b.append(P("M855,270 L855,282 L775,282 L775,352 L705,352", "ln", "ar"))
    b.append(T(800, 302, "shortcut has a conv:", "mut") + T(800, 318, "it ADDS parameters", "mut"))
    for i, s in enumerate(["used when the shape", "changes (stride or", "channel count), so x", "CAN be added at all"]):
        b.append(T(500, 344 + i * 16, s, "mut"))
    b.append(A(620, 352, 675, 352, "lnd", "ar"))

    # ---- bottom strip: how the 50 layers stack ----
    b.append(T(30, 440, "C. how they stack", "h2"))
    stack = ["conv 7x7", "BN", "maxpool", "C I I", "C I I I", "C I I I I I", "C I I", "avg pool", "Dense 1000"]
    for k, s in enumerate(stack):
        x = 30 + k * 100
        b.append(R(x, 466, 88, 26, "grn" if 3 <= k <= 6 else "cell", 3) + T(x + 44, 483, s, "num", "middle"))
        if k < 8:
            b.append(A(x + 88, 479, x + 100, 479))
        if 3 <= k <= 6:
            b.append(T(x + 44, 460, f"stage{k - 1}", "mut", "middle"))
    b.append(T(30, 512, "C = convolutional block (shortcut has a conv) ; I = identity block (plain wire)", "mut"))
    b.append(T(30, 530, "16 blocks x 3 convs + the first conv + the final dense = 50 -> ResNet50", "numb"))

    b.append(note(30, 552, 900, 76, "The one idea",
                  ["The two block types have the SAME main branch; the only difference is what the shortcut carries - a bare",
                   "wire, or one conv + BN to match a changed shape. In both, the last BN has no ReLU: it comes after the add."], "handk"))
    svg("resnet-blocks.svg", 960, 645, "".join(b))

# ---------------- SSD: fixed tensor in, variable boxes out ----------------

if want("ssd"):
    b = [header("SSD: SINGLE SHOT DETECTOR", "A fixed grid of guesses, then a cleanup", 960)]

    def anchor_cell(x, y, s, shapes):
        out = [R(x, y, s, s, "cell", 2)]
        for w, h, cls in shapes:
            out.append(R(x + (s - w) / 2, y + (s - h) / 2, w, h, cls, 2))
        return "".join(out)

    cs = [70, 175, 280, 385, 490, 595]
    ks = [4, 6, 6, 6, 4, 4]
    ws = [68, 58, 48, 40, 32, 24]
    hs = [84, 70, 58, 46, 36, 24]
    sizes = ["38x38", "19x19", "10x10", "5x5", "3x3", "1x1"]
    rfs = ["sees ~7px", "~15px", "~30px", "~60px", "~100px", "~300px"]

    # (1) backbone + (2) heads, all feeding one bus up top
    b.append(P("M70,100 L760,100", "ln"))
    b.append(T(430, 92, "concatenate every head", "mut"))
    for k in range(6):
        cx, w, h = cs[k], ws[k], hs[k]
        b.append(R(cx - 43, 116, 86, 50, "blu", 3))
        b.append(T(cx, 132, f"k={ks[k]} anchors", "num", "middle"))
        b.append(T(cx, 147, "3x3 conv ->", "num", "middle"))
        b.append(T(cx, 162, "k*(4+21)", "num", "middle"))
        b.append(A(cx, 116, cx, 102))
        b.append(A(cx, 240 - h / 2, cx, 170, "lnd", "ar"))
        b.append(R(cx - w / 2, 240 - h / 2, w, h, "grn", 2))
        b.append(T(cx, 300, sizes[k], "numb", "middle") + T(cx, 316, rfs[k], "mut", "middle"))
        if k < 5:
            b.append(A(cx + w / 2 + 4, 240, cs[k + 1] - ws[k + 1] / 2 - 4, 240))
    b.append(T(125, 186, "conv + pool", "mut", "middle"))

    # (3) one fixed-size tensor
    for i, s in enumerate(["8732 rows", "one per anchor"]):
        b.append(T(655, 150 + i * 16, s, "numb" if i == 0 else "mut"))
    for i, s in enumerate(["each row:", "4 box offsets", "+ 21 class scores"]):
        b.append(T(655, 192 + i * 16, s, "mut"))
    b.append(T(655, 252, "fixed size,", "hand") + T(655, 270, "always", "hand"))
    b.append(P("M760,100 L820,100 L820,130", "ln", "ar"))
    b.append(R(770, 130, 100, 210, "pur", 4))
    counts = ["38x38x4=5776", "19x19x6=2166", "10x10x6= 600", "5x5x6  = 150", "3x3x4  =  36", "1x1x4  =   4"]
    for i, s in enumerate(counts):
        y = 130 + i * 35
        if i:
            b.append(P(f"M770,{y} L870,{y}", "lnd"))
        b.append(T(820, y + 22, s, "num", "middle"))

    # (4) the only variable-length part
    b.append(A(820, 340, 820, 404))
    b.append(T(810, 370, "variable length starts HERE, outside the net", "hand", "end"))
    b.append(R(745, 404, 150, 30, "grey", 4) + T(820, 424, "score > threshold", "num", "middle"))
    b.append(A(820, 434, 820, 452))
    b.append(R(745, 452, 150, 46, "grey", 4))
    b.append(T(820, 472, "per-class NMS:", "num", "middle") + T(820, 490, "IoU > 0.5 dropped", "num", "middle"))
    b.append(A(820, 498, 820, 516))
    b.append(R(720, 516, 200, 120, "blu", 4) + T(820, 534, "3 boxes survive", "num", "middle"))
    b.append(R(734, 546, 86, 40, "cell", 2) + T(777, 570, "dog 0.91", "num", "middle"))
    b.append(R(828, 542, 78, 32, "cell", 2) + T(867, 562, "car 0.88", "num", "middle"))
    b.append(R(760, 592, 110, 34, "cell", 2) + T(815, 613, "person 0.76", "num", "middle"))

    # the anchors inside one cell
    b.append(anchor_cell(40, 410, 96, [(88, 88, "pur"), (88, 40, "blu"), (40, 88, "grn"), (58, 58, "yel")]))
    b.append(T(88, 522, "k=4 (38x38 head)", "num", "middle"))
    b.append(anchor_cell(180, 410, 96, [(88, 88, "pur"), (88, 44, "blu"), (44, 88, "grn"),
                                        (88, 26, "pnk"), (26, 88, "grey"), (56, 56, "yel")]))
    b.append(T(228, 522, "k=6 (10x10 head)", "num", "middle"))
    b.append(P("M88,410 L88,356 L70,356 L70,332", "lnd", "ar"))
    b.append(P("M228,410 L228,356 L280,356 L280,332", "lnd", "ar"))
    b.append(T(300, 404, "the k default boxes inside ONE cell", "h2"))
    for i, s in enumerate(["4 shapes: small square, big square, tall, wide",
                           "6 adds two more aspect ratios (1:3, 3:1)",
                           "anchor shapes are FIXED; the head predicts",
                           "only an offset (dx, dy, dw, dh) per anchor"]):
        b.append(T(300, 428 + i * 16, s, "mut"))
    b.append(T(300, 504, "early/large maps carry SMALL objects,", "hand"))
    b.append(T(300, 522, "late/small maps carry the BIG ones", "hand"))

    b.append(note(30, 656, 900, 76, "The one idea",
                  ["The network always emits the same fixed-size tensor, 8732 x (4 offsets + 21 scores). The variable number",
                   "of objects is produced after it, by thresholding and NMS - and the map that fires tells you the object's size."], "handk"))
    svg("ssd-pipeline.svg", 960, 750, "".join(b))

# ---------------- Sliding window = one convolution ----------------

if want("slidewin"):
    b = [header("SLIDING WINDOW AS CONVOLUTION", "Drop the dense layer and the window slides itself", 960)]

    # the two inputs, same weights
    b.append(T(40, 100, "A. one window", "h2"))
    b.append(R(40, 112, 100, 100, "cell", 2) + T(90, 166, "one 32x32 crop", "num", "middle"))
    b.append(T(40, 258, "1 crop -> 1 forward pass", "mut"))
    b.append(T(560, 100, "B. the whole image", "h2"))
    b.append(R(758, 112, 134, 134, "grey", 2))
    for dx, dy, cls in [(0, 0, "blu"), (32, 0, "grn"), (0, 32, "yel"), (32, 32, "pur")]:
        b.append(R(760 + dx, 114 + dy, 94, 94, cls, 2))
    b.append(T(930, 258, "4 window positions, stride 4 px (drawn exaggerated)", "mut", "end"))
    for i, s in enumerate(["4 crops of the same image, one forward pass each",
                           "vs ONE pass over the whole image"]):
        b.append(T(200, 140 + i * 20, s, "mut"))
    b.append(T(200, 188, "same weights, same parameter count", "hand"))
    b.append(T(200, 208, "the Dense layer becomes a 5x5 conv", "hand"))
    b.append(T(200, 232, "colours below match the window positions above", "mut"))

    ops = ["input", "conv 5x5", "pool /2", "conv 5x5", "pool /2", "conv 5x5", "conv 1x1", "conv 1x1"]
    left = ["32x32x3", "28x28x32", "14x14x32", "10x10x64", "5x5x64", "1x1x128", "1x1x128", "1x1x3"]
    right = ["36x36x3", "32x32x32", "16x16x32", "12x12x64", "6x6x64", "2x2x128", "2x2x128", "2x2x3"]
    for x0, shapes in [(40, left), (560, right)]:
        for k, (op, sh) in enumerate(zip(ops, shapes)):
            y = 270 + k * 34
            cls = "yel" if k == 5 else ("grn" if k == 7 else "cell")
            b.append(R(x0, y, 230, 26, cls, 3))
            b.append(T(x0 + 10, y + 17, op, "num") + T(x0 + 220, y + 17, sh, "numb", "end"))
            if k < 7:
                b.append(A(x0 + 115, y + 26, x0 + 115, y + 34))
    b.append(T(280, 457, "was Dense 128", "hand"))
    b.append(T(280, 525, "one softmax", "hand"))
    b.append(T(800, 450, "same kernel,", "mut") + T(800, 466, "2x2 output", "mut"))
    for dx, dy, cls in [(0, 0, "blu"), (26, 0, "grn"), (0, 26, "yel"), (26, 26, "pur")]:
        b.append(R(810 + dx, 500 + dy, 24, 24, cls, 2))
    b.append(A(792, 521, 806, 521))
    b.append(T(835, 566, "4 softmaxes, one per window", "hand", "middle"))

    b.append(R(30, 590, 900, 54, "grey", 6))
    b.append(T(480, 614, "same weights, same parameter count; one pass instead of 4 crops", "numb", "middle"))
    b.append(T(480, 632, "two 2x pools -> window stride = 4 input pixels, not 1", "mut", "middle"))

    b.append(note(30, 664, 900, 76, "The one idea",
                  ["A dense layer rewritten as a convolution turns a per-window classifier into a per-position one, for free.",
                   "The catch: pooling fixes the stride (4 px here), so the positions and shapes in between need anchors."], "handk"))
    svg("sliding-window-as-convolution.svg", 960, 756, "".join(b))

# ---------------- Rashid 1: the forward pass collapses into one matrix multiply ----------------
if want("fwdmat"):
    b = [header("FORWARD PASS", "Many little sums are one matrix multiply", 960)]
    # Panel A: the network, drawn node by node
    b.append(T(30, 112, "A. what the picture says", "h2"))
    ins, hid = [(100, 160), (100, 224), (100, 288)], [(280, 160), (280, 224), (280, 288)]
    for k, (x, y) in enumerate(ins):
        b.append(C(x, y, 19, "blu")); b.append(T(x, y + 5, "i%d" % (k + 1), "num", "middle"))
    for k, (x, y) in enumerate(hid):
        b.append(C(x, y, 19, "grn")); b.append(T(x, y + 5, "h%d" % (k + 1), "num", "middle"))
    for (x1, y1) in ins:
        for (x2, y2) in hid:
            b.append(P("M%d,%d L%d,%d" % (x1 + 19, y1, x2 - 19, y2), "lnd"))
    b.append(T(190, 136, "w(j,i)", "mut", "middle"))
    b.append(T(36, 330, "every link carries input i, scaled", "mut"))
    b.append(T(36, 346, "by w(j,i); each node sums what", "mut"))
    b.append(T(36, 362, "arrives, then squashes it", "mut"))
    # Panel B: written out, then collapsed
    b.append(P("M360,100 L360,382", "lnd"))
    b.append(T(382, 112, "B. the same thing, written out", "h2"))
    for k, r in enumerate(["net(1) = w(1,1) i1 + w(1,2) i2 + w(1,3) i3",
                           "net(2) = w(2,1) i1 + w(2,2) i2 + w(2,3) i3",
                           "net(3) = w(3,1) i1 + w(3,2) i2 + w(3,3) i3"]):
        b.append(T(388, 146 + k * 20, r, "num"))
    b.append(T(388, 228, "the i's repeat down every row and the w's", "hand"))
    b.append(T(388, 246, "across - that pattern IS a matrix product", "hand"))
    b.append(R(388, 264, 200, 74, "grey", 6))
    b.append(T(488, 294, "X = W . I", "h2", "middle"))
    b.append(T(488, 318, "O = sigmoid(X)", "num", "middle"))
    b.append(T(614, 290, "one line, any layer size,", "mut"))
    b.append(T(614, 306, "and numpy hands it to a", "mut"))
    b.append(T(614, 322, "BLAS kernel instead of", "mut"))
    b.append(T(614, 338, "a Python loop", "mut"))
    # Band C: shapes
    b.append(P("M30,398 L930,398", "lnd"))
    b.append(T(30, 426, "C. the shape rule", "h2"))
    b.append(R(36, 446, 70, 54, "yel", 4)); b.append(T(71, 470, "W", "numb", "middle")); b.append(T(71, 488, "3 x 3", "num", "middle"))
    b.append(T(116, 478, ".", "ink", "middle"))
    b.append(R(126, 446, 40, 54, "blu", 4)); b.append(T(146, 470, "I", "numb", "middle")); b.append(T(146, 488, "3 x 1", "num", "middle"))
    b.append(T(176, 478, "=", "ink", "middle"))
    b.append(R(186, 446, 40, 54, "grn", 4)); b.append(T(206, 470, "X", "numb", "middle")); b.append(T(206, 488, "3 x 1", "num", "middle"))
    b.append(T(252, 462, "W is (nodes in the NEXT layer) x (nodes in THIS layer),", "num"))
    b.append(T(252, 480, "so W.I - get it backwards and the shapes refuse to line up.", "num"))
    b.append(T(252, 500, "PyTorch stores it transposed and computes I @ W.T, which is why nn.Linear reads (in_features, out_features)", "mut"))
    b.append(T(252, 516, "while the weight it allocates has shape (out_features, in_features).", "mut"))
    b.append(note(30, 540, 900, 80, "The one idea",
                  ["The matrix form is not a different algorithm - it is the same per-node sums, stacked.",
                   "Write it once as W.I and the code stops caring whether the layer has 3 nodes or 3,000."], "handk"))
    svg("nn-forward-as-matmul.svg", 960, 636, "".join(b))

# ---------------- Rashid 2: two stories about backprop, one of them exact ----------------
if want("bpstory"):
    b = [header("BACKPROPAGATION", "Two stories about blame, and which one is the gradient", 960)]
    # Panel A: proportional split
    b.append(T(30, 112, "A. the book's intuition", "h2"))
    b.append(T(30, 134, "split the blame by weight", "mut"))
    b.append(C(90, 190, 18, "grn")); b.append(T(90, 195, "h1", "num", "middle"))
    b.append(C(90, 266, 18, "grn")); b.append(T(90, 271, "h2", "num", "middle"))
    b.append(C(250, 228, 18, "pnk")); b.append(T(250, 233, "o1", "num", "middle"))
    b.append(A(108, 190, 232, 222)); b.append(A(108, 266, 232, 234))
    b.append(T(165, 186, "w=3", "num", "middle")); b.append(T(165, 272, "w=1", "num", "middle"))
    b.append(T(282, 233, "e1", "numb")); b.append(T(282, 251, "error", "mut"))
    b.append(T(36, 316, "the bigger weight did more of the damage,", "mut"))
    b.append(T(36, 332, "so give it more of the blame:", "mut"))
    b.append(T(36, 358, "e(h1) = e1 * 3/(3+1) = 0.75 e1", "num"))
    b.append(T(36, 376, "e(h2) = e1 * 1/(3+1) = 0.25 e1", "num"))
    b.append(T(36, 404, "the shares add to 1 - it is a split", "hand"))
    # Panel B: what the code does
    b.append(P("M330,100 L330,440", "lnd"))
    b.append(T(352, 112, "B. what the code does", "h2"))
    b.append(T(358, 146, "e(hidden) = W.T . e(output)", "numb"))
    b.append(R(358, 164, 230, 64, "grey", 6))
    b.append(T(473, 188, "e(h1) = 3 e1", "num", "middle"))
    b.append(T(473, 208, "e(h2) = 1 e1", "num", "middle"))
    b.append(T(358, 256, "the denominators are GONE.", "hand"))
    b.append(T(358, 274, "these are 4x the shares in A.", "hand"))
    b.append(T(358, 302, "The book switches from one to the", "mut"))
    b.append(T(358, 318, "other without saying so. The split", "mut"))
    b.append(T(358, 334, "story is a mnemonic for the SHAPE", "mut"))
    b.append(T(358, 350, "of the answer, W.T, not for its size.", "mut"))
    b.append(T(358, 378, "A per-node denominator is not a", "mut"))
    b.append(T(358, 394, "constant, so the learning rate", "mut"))
    b.append(T(358, 410, "cannot absorb it.", "mut"))
    # Panel C: the gradient
    b.append(P("M612,100 L612,440", "lnd"))
    b.append(T(634, 112, "C. the actual gradient", "h2"))
    b.append(T(640, 146, "dE/do(j) = sum over k of", "num"))
    b.append(R(640, 164, 290, 48, "yel", 6))
    b.append(T(785, 194, "e(k) * sig'(net k) * w(j,k)", "numb", "middle"))
    b.append(T(640, 240, "the chain rule puts sig'(net k)", "hand"))
    b.append(T(640, 258, "INSIDE the sum. The book", "hand"))
    b.append(T(640, 276, "leaves it out.", "hand"))
    b.append(T(640, 306, "Measured on one hidden layer:", "mut"))
    b.append(T(640, 328, "cosine vs true gradient  0.9996", "num"))
    b.append(T(640, 346, "magnitude                5.17x", "num"))
    b.append(T(640, 374, "Right direction, wrong length -", "mut"))
    b.append(T(640, 390, "which is why the book's network", "mut"))
    b.append(T(640, 406, "trains anyway and nobody notices.", "mut"))
    b.append(note(30, 456, 900, 80, "The one idea",
                  ["All three stories agree that blame flows back through W transposed. They disagree on the factor.",
                   "Only C is the derivative of the loss; A and B are useful pictures that happen to point the same way."], "handk"))
    svg("nn-backprop-two-stories.svg", 960, 552, "".join(b))

# ---------------- Rashid 3: anatomy of one weight's gradient ----------------
if want("gradanat"):
    b = [header("THE WEIGHT UPDATE", "Three factors, one for each link in the chain", 960)]
    b.append(T(30, 110, "dE/dw(j,k)  =  -( t(k) - o(k) )  *  sig(net k) ( 1 - sig(net k) )  *  o(j)", "h2"))
    cols = [(40, 280, "pnk", "( t(k) - o(k) )", "how wrong the", "output node is",
             "dE/do(k)", "0 when correct, so a", "correct node stops pulling"),
            (348, 280, "yel", "sig(net) (1 - sig(net))", "how responsive the", "node is right now",
             "do(k)/dnet(k)", "peaks at 0.25, ~0 when", "saturated: a stuck node"),
            (656, 274, "grn", "o(j)", "how loud the incoming", "signal was",
             "dnet(k)/dw(j,k)", "0 input means 0 update,", "whatever the error was")]
    for x, w, c, head, l1, l2, deriv, n1, n2 in cols:
        b.append(R(x, 140, w, 56, c, 6))
        b.append(T(x + w / 2, 174, head, "numb", "middle"))
        b.append(T(x, 220, l1, "ink")); b.append(T(x, 238, l2, "ink"))
        b.append(T(x, 266, deriv, "num"))
        b.append(T(x, 294, n1, "mut")); b.append(T(x, 310, n2, "mut"))
    for x in (328, 636):
        b.append(T(x, 174, "x", "ink", "middle"))
    b.append(P("M40,330 L930,330", "lnd"))
    b.append(T(40, 358, "Three ways one weight can stop learning", "h2"))
    for k, s in enumerate(["the node is already right  (factor 1 is 0)",
                           "the node is saturated      (factor 2 is ~0)",
                           "the input was silent       (factor 3 is 0)"]):
        b.append(T(46, 384 + k * 18, s, "num"))
    b.append(T(500, 384, "This is why the book scales inputs to 0.01-1.0", "hand"))
    b.append(T(500, 402, "and targets to 0.01-0.99: a 0 input kills", "hand"))
    b.append(T(500, 420, "factor 3, and a target of 1 is unreachable,", "hand"))
    b.append(T(500, 438, "so it drives weights up forever chasing it.", "hand"))
    b.append(T(40, 466, "w(j,k) <- w(j,k) - alpha * dE/dw(j,k)", "numb"))
    b.append(T(330, 466, "minus, because we walk DOWN the slope", "mut"))
    b.append(note(30, 492, 900, 80, "The one idea",
                  ["Every gradient in a neural net is a product of three local quantities, and a zero anywhere kills it.",
                   "Read the formula as error x responsiveness x signal and you can predict what will fail to learn."], "handk"))
    svg("nn-gradient-anatomy.svg", 960, 588, "".join(b))
