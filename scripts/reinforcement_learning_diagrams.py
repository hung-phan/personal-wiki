"""Diagrams for wiki/machine-learning/reinforcement-learning notes.  Run: python3 scripts/reinforcement_learning_diagrams.py wiki/machine-learning/reinforcement-learning/assets [qlearning ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_svg as ps
from paper_svg import T, R, A, P, C, header, matrix, colvec, note
ps.OUT = sys.argv[1]
os.makedirs(ps.OUT, exist_ok=True)
svg = ps.svg
WHICH = sys.argv[2:] or ["all"]
def want(n): return "all" in WHICH or n in WHICH

# ---------------- Q-learning: values on a grid + the TD update ----------------
if want("qlearning"):
    GRID = ["....G", ".#.#X", "S....", "..#.."]
    V = [[0.73, 0.81, 0.9, 1.0, 0], [0.66, 0, 0.81, 0, 0], [0.59, 0.66, 0.73, 0.66, 0.59], [0.53, 0.59, 0, 0.59, 0.53]]
    POL = ["> > > > G", "^ # ^ # X", "^ > ^ < <", "^ ^ # ^ ^"]
    b = [header("Q-LEARNING", "Values spread back from the reward, one step at a time", 960)]
    b.append(T(40, 104, "1. Bellman values, γ = 0.9", "h2"))
    x0, y0, cs = 40, 120, 76
    for r in range(4):
        for c in range(5):
            ch = GRID[r][c]; x, y = x0 + c * cs, y0 + r * cs
            cls = {"#": "grey", "G": "grn", "X": "pnk"}.get(ch, "cell")
            b.append(R(x, y, cs, cs, cls))
            if ch == "#": continue
            if ch in "GX": b.append(T(x + cs / 2, y + cs / 2 + 6, "+1" if ch == "G" else "−1", "h2", "middle")); continue
            b.append(T(x + cs / 2, y + 30, f"{V[r][c]:.2f}", "numb", "middle"))
            arrow = POL[r].split()[c]
            b.append(T(x + cs / 2, y + 56, {"^": "up", ">": "right", "<": "left", "v": "down"}[arrow], "hand", "middle"))
            if ch == "S": b.append(T(x + 6, y + 14, "S", "mut"))
    b.append(T(40, 440, "each cell: best value, and the move that earns it", "mut"))
    b.append(T(40, 458, "1, 0.9, 0.81, 0.73…: one factor of γ per step from the goal", "mut"))
    b.append(note(460, 96, 470, 150, "Bellman equation", ["V(s) = max over actions a of [ R(s, a) + γ · V(s') ]", "", "γ (gamma) < 1: a reward one step later is worth less,", "so short paths win. With random slips, replace V(s')", "by the expected value over where you might land."], "handk"))
    b.append(note(460, 260, 470, 150, "Q-values: score the moves, not the cells", ["Q(s, a) = R(s, a) + γ · max over a' of Q(s', a')", "V(s) = max over a of Q(s, a)", "", "the agent doesn't know R or s' in advance —", "it learns Q from experience:"], "handk"))
    b.append(R(460, 420, 470, 70, "yel", 6))
    b.append(T(695, 446, "TD = r + γ · max Q(s', a') − Q(s, a)", "numb", "middle"))
    b.append(T(695, 470, "new Q(s, a) = Q(s, a) + α · TD", "numb", "middle"))
    b.append(T(695, 512, "α (learning rate): how far to move toward each new sample;", "mut", "middle"))
    b.append(T(695, 528, "TD shrinks to 0 when estimates agree with experience", "mut", "middle"))
    svg("q-learning-grid.svg", 960, 550, "".join(b))

# ---------------- Deep Q-learning loop ----------------
if want("dqn"):
    b = [header("DEEP Q-LEARNING", "A network replaces the table, plus two stabilisers", 960)]
    def box(x, y, w, h, t1, t2, c):
        b.append(R(x, y, w, h, c, 6)); b.append(T(x + w / 2, y + 22, t1, "h2", "middle")); b.append(T(x + w / 2, y + 40, t2, "mut", "middle"))
    box(40, 110, 200, 60, "environment", "LunarLander, Ms. Pac-Man", "grn")
    box(380, 110, 220, 60, "online Q-network", "state in, one Q per action out", "pnk")
    box(720, 110, 200, 60, "action", "ε-greedy on the Q-values", "yel")
    b.append(A(240, 132, 380, 132)); b.append(T(310, 124, "state s", "mut", "middle"))
    b.append(A(600, 140, 720, 140)); b.append(P("M820,170 L820,200 L140,200 L140,170", "ln", "ar")); b.append(T(480, 194, "act in the environment", "mut", "middle"))
    box(40, 240, 260, 60, "replay buffer", "(s, a, r, s', done), up to 100k", "blu")
    b.append(P("M100,170 L100,240", "ln", "ar")); b.append(T(108, 222, "store every step", "mut"))
    box(380, 240, 220, 60, "random minibatch", "e.g. 100 past steps", "blu")
    b.append(A(300, 270, 380, 270)); b.append(T(340, 262, "sample", "mut", "middle"))
    box(380, 350, 220, 60, "target network", "a slow copy of the online net", "grey")
    b.append(R(660, 240, 270, 170, "note", 8))
    b.append(T(795, 266, "loss for each sampled step", "h2", "middle"))
    b.append(T(795, 294, "target y = r + γ · max Q_target(s')", "num", "middle")); b.append(T(795, 312, "(just r if the episode ended)", "mut", "middle"))
    b.append(T(795, 338, "loss = (Q_online(s, a) − y)²", "num", "middle")); b.append(T(795, 356, "only the action actually taken", "mut", "middle"))
    b.append(T(795, 384, "backprop + Adam on the online net", "hand", "middle"))
    b.append(A(600, 270, 660, 290)); b.append(A(600, 380, 660, 330))
    b.append(P("M490,240 L490,170", "lnr", "arr")); b.append(T(498, 212, "gradient step", "hand"))
    b.append(P("M380,150 C 330,150 330,380 378,380", "lnd", "ar")); b.append(T(320, 330, "soft update:", "hand", "end")); b.append(T(320, 346, "target = τ·online + (1−τ)·target", "num", "end"))
    b.append(note(40, 436, 880, 116, "Why the extra machinery",
                  ["Consecutive steps are almost identical; training on them in order makes the network chase its own tail. Sampling at random",
                   "from a large buffer breaks that correlation and reuses rare experiences. The target network keeps the 'right answer' y from",
                   "moving with every update. Pixels instead of 8 numbers? Put a CNN in front: the network is the only thing that changes."]))
    svg("dqn-loop.svg", 960, 570, "".join(b))

# ---------------- Actor-critic / A3C ----------------
if want("a3c"):
    b = [header("ACTOR-CRITIC", "One network, two heads: what to do, and how good it is here", 960)]
    b.append(T(40, 104, "1. the network (course: Kung Fu Master)", "h2"))
    b.append(R(40, 124, 120, 70, "cell", 4)); b.append(T(100, 150, "4 stacked", "num", "middle")); b.append(T(100, 166, "42×42 gray", "num", "middle")); b.append(T(100, 182, "frames", "num", "middle"))
    b.append(A(160, 159, 190, 159))
    b.append(R(190, 124, 150, 70, "yel", 4)); b.append(T(265, 150, "3 conv layers", "numb", "middle")); b.append(T(265, 168, "32 ch, 3×3, stride 2", "mut", "middle"))
    b.append(A(340, 159, 370, 159))
    b.append(R(370, 124, 110, 70, "blu", 4)); b.append(T(425, 150, "dense", "numb", "middle")); b.append(T(425, 168, "512 to 128", "mut", "middle"))
    b.append(A(480, 145, 520, 118)); b.append(A(480, 175, 520, 202))
    b.append(R(520, 92, 190, 50, "pnk", 4)); b.append(T(615, 113, "actor: 14 logits", "numb", "middle")); b.append(T(615, 131, "softmax: π(a|s)", "mut", "middle"))
    b.append(R(520, 176, 190, 50, "grn", 4)); b.append(T(615, 197, "critic: 1 number", "numb", "middle")); b.append(T(615, 215, "V(s)", "mut", "middle"))
    b.append(note(730, 92, 200, 134, "Advantage", ["A = (r + γ·V(s')) − V(s)", "> 0: better than", "expected, do it more", "< 0: do it less"], "handk"))
    b.append(T(40, 262, "2. the losses", "h2"))
    b.append(note(40, 276, 430, 120, "loss = actor + critic − entropy bonus", ["actor: −log π(a|s) · A (push up good actions)", "critic: A² (predict returns better)", "entropy: keep the policy from collapsing too early", "course: entropy weight 0.001, rewards × 0.01"], "handk"))
    b.append(T(500, 262, "3. asynchronous vs synchronous", "h2"))
    for k in range(4):
        y = 286 + k * 26
        b.append(R(500, y, 90, 20, "cell", 3)); b.append(T(545, y + 14, f"worker {k + 1}", "num", "middle"))
        b.append(A(590, y + 10, 640, 330))
    b.append(R(640, 310, 130, 40, "pnk", 4)); b.append(T(705, 334, "shared network", "numb", "middle"))
    b.append(T(780, 300, "A3C (2016): each worker", "mut")); b.append(T(780, 316, "has its own env + copy,", "mut")); b.append(T(780, 332, "sends gradients whenever", "mut")); b.append(T(780, 348, "it's ready", "mut"))
    b.append(T(780, 374, "A2C: step all envs together,", "mut")); b.append(T(780, 390, "one batched update", "mut"))
    b.append(note(40, 420, 890, 128, "Where PPO and SAC fit",
                  ["Both are actor-critics. PPO (2017) limits each update: the new/old probability ratio is clipped to [1−ε, 1+ε], so one",
                   "batch can't wreck the policy — the default for games and robotics, and the RL step in RLHF. SAC (2018) is off-policy",
                   "for continuous actions: it maximises reward plus an entropy bonus, reusing a replay buffer like DQN.",
                   "The course's own code is A2C-style: 10 environments stepped together in one process, no LSTM."], "handk"))
    svg("actor-critic.svg", 960, 570, "".join(b))

# ---------------- Foundations: how DP, MC and TD estimate V ----------------
if want("foundations"):
    b = [header("ESTIMATING VALUES", "Three ways to solve the Bellman equation, by what they need", 960)]
    cols = [("dynamic programming", "knows the rules p(s', r | s, a)", "V(s) = Σ p · [r + γ V(s')]", "all next states, one step deep", "pnk"),
            ("Monte Carlo", "plays whole episodes", "new V(s) = V(s) + α [G − V(s)]", "one path, all the way to the end", "yel"),
            ("temporal difference", "learns after every step", "new V(s) = V(s) + α [r + γ V(s') − V(s)]", "one path, one step, then guess", "grn")]
    for k, (t1, t2, eq, depth, c) in enumerate(cols):
        x = 40 + k * 300
        b.append(R(x, 100, 280, 56, c, 6)); b.append(T(x + 140, 122, t1, "h2", "middle")); b.append(T(x + 140, 142, t2, "mut", "middle"))
        b.append(C(x + 140, 186, 12, "blu"))                                    # the state being updated
        if k == 0:
            for j, dx in enumerate((-80, 0, 80)):
                b.append(A(x + 140, 198, x + 140 + dx, 238)); b.append(C(x + 140 + dx, 250, 10, "cell"))
        else:
            ys = [250, 310, 370] if k == 1 else [250]
            prev = 186
            for y in ys:
                b.append(A(x + 140, prev + 12, x + 140, y - 12)); b.append(C(x + 140, y, 10, "cell")); prev = y
            if k == 1: b.append(R(x + 128, 392, 24, 16, "grey", 2)); b.append(T(x + 166, 404, "end: G known", "mut"))
            else: b.append(T(x + 160, 254, "V(s') — a guess", "hand"))
        b.append(T(x + 140, 440, eq, "num", "middle")); b.append(T(x + 140, 460, depth, "mut", "middle"))
    b.append(note(40, 484, 880, 76, "Read across",
                  ["DP needs a model but no samples; MC needs no model but must wait for the episode to end (noisy, unbiased);",
                   "TD needs neither: it bootstraps from its own estimate of the next state (less noisy, biased early). Q-learning is TD on Q."], "handk"))
    svg("rl-estimation.svg", 960, 580, "".join(b))

# ---------------- Function approximation: RBF features + TD(lambda) ----------------
if want("fa"):
    b = [header("FUNCTION APPROXIMATION", "Too many states for a table: features, linear Q, traces", 960)]
    b.append(T(40, 104, "1. RBF features (MountainCar)", "h2"))
    b.append(R(40, 124, 120, 60, "cell", 4)); b.append(T(100, 150, "state s", "numb", "middle")); b.append(T(100, 168, "position, velocity", "mut", "middle"))
    b.append(A(160, 154, 190, 154)); b.append(R(190, 124, 110, 60, "grey", 4)); b.append(T(245, 150, "scale", "numb", "middle")); b.append(T(245, 168, "StandardScaler", "mut", "middle"))
    b.append(A(300, 154, 330, 154))
    for k, g in enumerate(("γ=5", "γ=2", "γ=1", "γ=0.5")):
        b.append(R(330, 110 + k * 22, 120, 18, "yel", 3)); b.append(T(390, 123 + k * 22, f"500 RBFs, {g}", "num", "middle"))
    b.append(A(450, 154, 480, 154)); b.append(R(480, 124, 110, 60, "blu", 4)); b.append(T(535, 150, "φ(s)", "numb", "middle")); b.append(T(535, 168, "2,000 features", "mut", "middle"))
    for k, a in enumerate(("left", "none", "right")):
        y = 108 + k * 34
        b.append(A(590, 154, 640, y + 13)); b.append(R(640, y, 150, 26, "pnk", 3)); b.append(T(715, y + 18, f"Q(s, {a}) = w·φ(s)", "num", "middle"))
    b.append(T(810, 140, "one linear model", "mut")); b.append(T(810, 156, "per action", "mut"))
    b.append(T(40, 214, "each RBF: exp(−γ · ‖s − centre‖²) — 1 at its centre, fading with distance; small γ = wide bumps", "mut"))
    b.append(T(40, 262, "2. λ-return: blend every n-step return", "h2"))
    lam = 0.7; bw = 54
    for n in range(1, 9):
        w = (1 - lam) * lam ** (n - 1); h = w * 400
        x = 60 + (n - 1) * (bw + 10)
        b.append(R(x, 420 - h, bw, h, "grn", 2)); b.append(T(x + bw / 2, 414 - h, f"{w:.2f}", "num", "middle")); b.append(T(x + bw / 2, 438, f"n={n}", "mut", "middle"))
    b.append(T(60, 460, "weights (1 − λ) λ^(n−1) for λ = 0.7; they sum to 1 (the tail goes to the full return)", "mut"))
    b.append(note(600, 252, 330, 210, "TD(λ) without waiting", ["δ = r + γ V(s') − V(s)", "trace:  e = γλ · e + ∇V(s)", "weights:  θ = θ + α · δ · e", "", "the trace remembers which features", "were active recently, so one TD error", "updates them all. λ = 0: TD(0);", "λ = 1: Monte Carlo."], "handk"))
    svg("function-approximation.svg", 960, 490, "".join(b))

# ---------------- Policy gradients ----------------
if want("pg"):
    b = [header("POLICY GRADIENTS", "Learn the policy directly: make good actions more likely", 960)]
    steps = [("play an episode", "sample a ~ π(a|s)", "yel"), ("returns", "G_t = r + γ G_(t+1)", "blu"), ("advantage", "G_t − V(s_t)", "grn"), ("update", "loss = −Σ adv · log π", "pnk")]
    for k, (t1, t2, c) in enumerate(steps):
        x = 40 + k * 230
        b.append(R(x, 104, 200, 60, c, 6)); b.append(T(x + 100, 128, t1, "h2", "middle")); b.append(T(x + 100, 148, t2, "num", "middle"))
        if k < 3: b.append(A(x + 200, 134, x + 230, 134))
    b.append(P("M840,164 L840,190 L140,190 L140,164", "ln", "ar")); b.append(T(490, 184, "repeat with the updated policy", "mut", "middle"))
    b.append(T(40, 232, "discrete actions: softmax", "h2"))
    probs = [0.15, 0.6, 0.25]
    for k, p in enumerate(probs):
        x = 60 + k * 100; h = p * 200
        b.append(R(x, 420 - h, 60, h, "blu", 2)); b.append(T(x + 30, 414 - h, f"{p:.2f}", "num", "middle")); b.append(T(x + 30, 438, ["left", "stay", "right"][k], "mut", "middle"))
    b.append(T(40, 466, "positive advantage: push the taken action's bar up", "mut"))
    b.append(T(380, 232, "continuous actions: Gaussian", "h2"))
    pts = " ".join(f"{400 + i * 4},{420 - 160 * __import__('math').exp(-((i - 30) / 12) ** 2 / 2):.1f}" for i in range(61))
    b.append(f'<polyline points="{pts}" class="ln" fill="none"/>')
    b.append(P("M520,250 L520,420", "lnd")); b.append(T(520, 438, "mean μ(s)", "mut", "middle"))
    b.append(A(520, 340, 566, 340)); b.append(T(543, 332, "σ(s)", "mut", "middle")); b.append(T(400, 452, "σ via softplus, so it stays > 0", "mut"))
    b.append(T(400, 474, "action = clip(μ + σ · noise, −1, 1)", "num"))
    b.append(note(680, 232, 250, 236, "Why it works", ["∇ E[G] = E[ G · ∇ log π(a|s) ]", "", "rare actions that paid off", "get a big push (1/π inside", "the log-gradient).", "", "subtracting V(s) changes", "nothing on average but", "removes most of the noise."], "handk"))
    svg("policy-gradients.svg", 960, 490, "".join(b))

# ---------------- Multi-armed bandits: UCB and Thompson ----------------
if want("bandit"):
    import math
    b = [header("MULTI-ARMED BANDITS", "Which ad next? Optimism (UCB) or sampled beliefs (Thompson)", 960)]
    b.append(T(40, 104, "1. UCB: show the ad with the highest upper bound", "h2"))
    ads = [(0.12, 0.05), (0.25, 0.04), (0.18, 0.08), (0.08, 0.25)]
    y0 = 330
    for k, (m, w) in enumerate(ads):
        x = 80 + k * 95
        b.append(P(f"M{x},{y0 - max(m - w, 0) * 500} L{x},{y0 - (m + w) * 500}", "ln"))
        b.append(R(x - 14, y0 - m * 500 - 3, 28, 6, "pnk", 1)); b.append(R(x - 14, y0 - (m + w) * 500 - 2, 28, 4, "grn", 1))
        b.append(T(x, y0 + 20, f"ad {k + 1}", "mut", "middle")); b.append(T(x, y0 + 36, ["shown a lot", "shown a lot", "some", "rarely"][k], "mut", "middle"))
    b.append(T(80 + 3 * 95 + 22, y0 - 0.33 * 500 + 4, "picked: highest bound", "hand"))
    b.append(T(40, 404, "pink: average click rate so far   green: upper bound", "mut"))
    b.append(T(40, 422, "bound = mean + sqrt(1.5 · ln n / N_ad) — shrinks as the ad is shown more", "num"))
    b.append(T(500, 104, "2. Thompson: draw a plausible rate for each ad", "h2"))
    def beta_pdf(x, a, bb): return math.exp((a - 1) * math.log(x) + (bb - 1) * math.log(1 - x) - (math.lgamma(a) + math.lgamma(bb) - math.lgamma(a + bb)))
    curves = [(4, 30, "lnr"), (12, 40, "ln"), (2, 6, "lnd")]
    for a, bb, cl in curves:
        pts = []
        for i in range(1, 100):
            x = i / 100 * 0.6; pts.append(f"{520 + x / 0.6 * 400:.1f},{340 - min(beta_pdf(x, a, bb), 14) * 15:.1f}")
        b.append(f'<polyline points="{" ".join(pts)}" class="{cl}" fill="none"/>')
    b.append(P("M520,340 L920,340", "ln")); b.append(T(720, 360, "click rate", "mut", "middle"))
    b.append(T(800, 300, "few shows: wide (dashed)", "hand")); b.append(T(700, 222, "more shows: narrower", "mut"))
    b.append(T(500, 404, "posterior: Beta(clicks + 1, misses + 1); sample one value per ad,", "num"))
    b.append(T(500, 422, "show the ad with the highest sample, update its counts", "num"))
    b.append(note(40, 444, 890, 100, "Explore vs exploit", ["Every round spent on a bad ad costs a click (regret); never trying alternatives risks missing the best one.",
                  "UCB is deterministic and must update every round; Thompson is random, handles delayed or batched feedback,",
                  "and usually earns more. A/B testing explores all the way, then exploits."], "handk"))
    svg("multi-armed-bandits.svg", 960, 560, "".join(b))

# ---------------- Evolution strategies: one generation + why it parallelises ----------------
if want("es"):
    import math
    b = [header("EVOLUTION STRATEGIES", "A reward-weighted average of random parameter nudges", 960)]

    # --- panel 1: parameter space ---
    b.append(T(30, 104, "1. One generation in parameter space", "h2"))
    b.append(T(30, 128, "dashed arcs: contours of expected return F(theta)", "mut"))
    ox, oy = 400, 170                                   # the peak sits up-right, off the cloud
    for r in (55, 110, 165, 220):
        b.append(P(f"M{ox - r},{oy} A{r},{r} 0 0,0 {ox},{oy + r}", "lnd"))
    b.append(T(412, 152, "higher ->", "hand"))

    tx, ty = 283, 287                                   # theta_t: sits on the r=165 contour
    rs, rl, rv = 52, 80, 88                             # dot ring / A_i label ring / vector-label ring
    b.append(P(f"M{tx - rs},{ty} a{rs},{rs} 0 1,0 {2 * rs},0 a{rs},{rs} 0 1,0 {-2 * rs},0", "lnd"))
    LAB = {292.5: "+1.1", 345: "+1.4", 15: "+0.9", 112.5: "-1.2", 157.5: "-0.8", 202.5: "-1.0"}
    A_AT = {292.5: 292.5, 345: 337.5, 15: 22.5}         # label angle -> dot angle (nudged off the big arrow)
    for k in range(8):
        a = 22.5 + 45 * k
        dx, dy = tx + rs * math.cos(math.radians(a)), ty + rs * math.sin(math.radians(a))
        b.append(P(f"M{tx},{ty} L{dx:.0f},{dy:.0f}", "lnd"))
        b.append(C(f"{dx:.0f}", f"{dy:.0f}", 4, "blu"))
    for la, txt in LAB.items():
        lx = tx + rl * math.cos(math.radians(la)); ly = ty + rl * math.sin(math.radians(la))
        b.append(R(lx - 15, ly - 11, 30, 14, "bg"))     # mask the contour behind the number
        b.append(T(f"{lx:.0f}", f"{ly:.0f}", txt, "numb", "middle"))
    for a, txt in ((67.5, "sigma*eps_i"), (247.5, "sigma*eps_i")):
        lx = tx + rv * math.cos(math.radians(a)); ly = ty + rv * math.sin(math.radians(a))
        b.append(R(lx - 36, ly - 11, 72, 14, "bg"))
        b.append(T(f"{lx:.0f}", f"{ly:.0f}", txt, "mut", "middle"))
    b.append(C(tx, ty, 7, "grn"))
    b.append(P(f"M{tx - rs - 8},{ty} L184,{ty}", "lnd"))
    b.append(R(130, 276, 56, 16, "bg")); b.append(T(182, 291, "theta_t", "handk", "end"))
    b.append(P(f"M{tx + 8},{ty} L{tx + rs - 2},{ty}", "lnr"))
    b.append(R(336, 278, 44, 16, "bg")); b.append(T(340, 291, "sigma", "hand"))
    b.append(A(tx, ty, 389, 181, "hi", "arr"))
    b.append(T(396, 174, "update step", "hand"))
    b.append(T(30, 412, "numbers: standardised return A_i = (F_i - mean F) / std F", "mut"))
    b.append(T(30, 430, "each thin vector is one offspring at theta_t + sigma * eps_i", "mut"))
    b.append(R(30, 448, 490, 58, "yel", 6))
    b.append(T(275, 472, "theta_(t+1) = theta_t + (eta / (N sigma)) * sum_i A_i eps_i", "numb", "middle"))
    b.append(T(275, 492, "positive A_i pulls toward that offspring, negative pushes away", "mut", "middle"))

    # --- panel 2: the map-reduce shape ---
    b.append(T(545, 104, "2. Why it parallelises", "h2"))
    b.append(R(560, 118, 330, 40, "pur", 6))
    b.append(T(725, 143, "theta_t (identical on every worker)", "num", "middle"))
    b.append(T(725, 176, "broadcast seed_i", "mut", "middle"))
    b.append(P("M560,158 L560,380", "ln"))
    for k in range(4):
        y = 196 + 54 * k
        b.append(R(600, y, 270, 44, "blu", 6))
        b.append(T(608, y + 26, f"W{k + 1}", "numb"))
        b.append(T(636, y + 18, "eps_i = randn(seed_i)", "num"))
        b.append(T(636, y + 36, "run 1 episode -> F_i", "num"))
        b.append(A(560, y + 22, 600, y + 22))
        b.append(A(870, y + 22, 890, y + 22))
    b.append(P("M890,218 L890,447 L874,447", "ln", "ar"))
    b.append(T(735, 420, "2 numbers per offspring", "mut", "middle"))
    b.append(R(600, 430, 270, 34, "grn", 6))
    b.append(T(735, 451, "reduce: (seed_i, F_i)", "num", "middle"))
    b.append(A(735, 464, 735, 476))
    b.append(R(600, 476, 270, 34, "yel", 6))
    b.append(T(735, 497, "theta_(t+1)", "numb", "middle"))

    b.append(note(30, 530, 900, 86, "The one idea", [
        "The step is a reward-weighted average of parameter-space nudges: good A_i pulls you in, bad A_i pushes away.",
        "Each worker gets a seed and sends back one scalar, so a whole generation costs two numbers per offspring -",
        "no gradients, no weight vectors and no episodes ever cross the network."], "handk"))
    svg("evolution-strategies.svg", 960, 632, "".join(b))

# ---------------- DDPG: the actor-critic composition, the GAN analogy, soft updates ----------------
if want("ddpg"):
    import math
    b = [header("DDPG", "Two networks glued into one differentiable chain", 960)]

    def chain(y, h, items):
        for i, (x0, w, label, cls) in enumerate(items):
            b.append(R(x0, y, w, h, cls, 6))
            b.append(T(x0 + w / 2, y + h / 2 + 4, label, "numb", "middle"))
            if i:
                px, pw = items[i - 1][0], items[i - 1][1]
                b.append(A(px + pw, y + h / 2, x0, y + h / 2))

    # --- panel 1: the composition ---
    b.append(T(30, 104, "1. The composition", "h2"))
    chain(140, 52, [(40, 60, "s", "blu"), (140, 140, "actor mu(s)", "grn"), (320, 110, "a = mu(s)", "cell"),
                    (470, 150, "critic Q(s,a)", "pnk"), (660, 160, "scalar Q(s,mu(s))", "yel")])
    b.append(T(614, 158, "*", "hand", "end"))
    b.append(P("M70,192 L70,230 L545,230 L545,192", "ln", "ar"))
    b.append(T(300, 224, "the critic sees s and a", "mut"))
    b.append(R(40, 250, 450, 44, "pnk", 6))
    b.append(T(52, 268, "critic update: minimise (Q(s,a) - y)^2", "num"))
    b.append(T(52, 284, "target y comes from the target nets (frozen)", "mut"))
    b.append(R(510, 250, 420, 44, "grn", 6))
    b.append(T(522, 268, "actor update: maximise Q(s, mu(s))", "num"))
    b.append(T(522, 284, "critic weights frozen (*), gradient flows through it", "mut"))

    # --- panel 2: the GAN analogy ---
    b.append(T(30, 336, "2. Same shape as a GAN", "h2"))
    chain(352, 46, [(40, 60, "z", "blu"), (140, 130, "generator G", "grn"), (310, 120, "fake sample", "cell"),
                    (470, 150, "discriminator D", "pnk"), (660, 160, "real / fake score", "yel")])
    b.append(T(614, 370, "*", "hand", "end"))
    b.append(T(40, 420, "train G by maximising D(G(z)) with D frozen  ->  train mu by maximising Q(s, mu(s)) with Q frozen", "num"))

    # --- panel 3: soft vs hard target update ---
    b.append(T(30, 462, "3. Soft vs hard target update", "h2"))
    b.append(P("M70,600 L70,486", "ln")); b.append(P("M70,600 L480,600", "ln"))
    b.append(T(275, 620, "training steps", "mut", "middle"))
    b.append(T(70, 478, "target weight", "mut"))
    x0, x1, ytop, ybot, nstep = 70, 480, 496, 580, 5
    b.append(P(f"M{x0},{ybot} L{x1},{ytop}", "lnd"))          # online weights, climbing
    def online(u): return ybot - (ybot - ytop) * u
    stair = [f"M{x0},{ybot}"]
    for k in range(nstep):                                     # hold, then jump: DQN's hard copy
        xa = x0 + (x1 - x0) * k / nstep; xb = x0 + (x1 - x0) * (k + 1) / nstep
        stair.append(f"L{xb:.0f},{online(k / nstep):.0f}")
        if k < nstep - 1: stair.append(f"L{xb:.0f},{online((k + 1) / nstep):.0f}")
    b.append(P(" ".join(stair), "ln"))
    pts = []
    for i in range(41):                                        # DDPG: exponential lag below online
        u = i / 40
        pts.append(f"{x0 + (x1 - x0) * u:.0f},{online(u) + 15 * (1 - math.exp(-4 * u)):.0f}")
    b.append(P("M" + " L".join(pts), "lnr"))
    b.append(T(492, 500, "online weights", "mut"))
    b.append(T(492, 522, "DQN: hard copy every N steps", "num"))
    b.append(T(492, 544, "DDPG: soft, every step", "hand"))
    b.append(R(690, 486, 240, 76, "yel", 6))
    b.append(T(810, 508, "theta_targ  <-", "num", "middle"))
    b.append(T(810, 526, "tau*theta + (1-tau)*theta_targ", "num", "middle"))
    b.append(T(810, 548, "tau = 0.005: a slow follow", "mut", "middle"))

    b.append(note(30, 640, 900, 86, "The one idea", [
        "DDPG is two networks glued into one differentiable chain: a = mu(s) feeds straight into Q(s, a).",
        "Each update freezes the other end - the critic step freezes the targets, the actor step freezes the",
        "critic (*) - which is exactly how a GAN trains G through a frozen D."], "handk"))
    svg("ddpg.svg", 960, 744, "".join(b))
