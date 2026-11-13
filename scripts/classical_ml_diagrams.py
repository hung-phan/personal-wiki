"""Diagrams for wiki/machine-learning/classical-ml notes.  Run: python3 scripts/classical_ml_diagrams.py wiki/machine-learning/classical-ml/assets [prep ...]"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_svg as ps
from paper_svg import T, R, A, P, C, header, note
ps.OUT = sys.argv[1]
os.makedirs(ps.OUT, exist_ok=True)
svg = ps.svg
WHICH = sys.argv[2:] or ["all"]
def want(n): return "all" in WHICH or n in WHICH

def box(b, x, y, w, h, t1, t2, c):
    b.append(R(x, y, w, h, c, 6)); b.append(T(x + w / 2, y + 22, t1, "h2", "middle")); b.append(T(x + w / 2, y + 40, t2, "mut", "middle"))

# ---------------- Preprocessing: split first, fit on train ----------------
if want("prep"):
    b = [header("DATA PREPROCESSING", "Split first; learn every statistic from the training rows only", 960)]
    box(b, 40, 100, 200, 56, "raw table", "features X, target y", "cell")
    b.append(A(240, 128, 290, 128))
    box(b, 290, 100, 180, 56, "train / test split", "e.g. 80 / 20, fixed seed", "yel")
    b.append(A(470, 118, 530, 100)); b.append(A(470, 140, 530, 200))
    box(b, 530, 74, 180, 56, "training rows", "fit + transform", "grn")
    box(b, 530, 176, 180, 56, "test rows", "transform only", "pnk")
    b.append(R(760, 74, 170, 158, "note", 8))
    for k, s in enumerate(["learned on train:", "imputation means", "scaler mean / std", "category list", "(dummy columns)"]):
        b.append(T(845, 100 + k * 26, s, "hand" if k == 0 else "num", "middle"))
    b.append(P("M760,150 L712,204", "lnd", "ar")); b.append(T(718, 252, "reuse them on test", "hand"))
    b.append(T(40, 300, "steps inside the fitted transformer", "h2"))
    steps = [("missing values", "fill with the mean (numbers)", "blu"), ("categories", "one-hot: one 0/1 column each", "blu"), ("scaling", "standardize or normalize", "blu")]
    for k, (t1, t2, c) in enumerate(steps):
        box(b, 40 + k * 300, 316, 270, 56, t1, t2, c)
        if k < 2: b.append(A(310 + k * 300, 344, 340 + k * 300, 344))
    b.append(note(40, 396, 430, 120, "Two scalings", ["normalize: (x − min) / (max − min), into [0, 1]", "standardize: (x − mean) / std, mean 0, std 1", "", "both per COLUMN; distances and gradients", "need comparable columns"], "handk"))
    b.append(note(500, 396, 430, 120, "Why split first", ["the test set plays 'future data':", "a mean or std computed with it", "leaks the answer into training,", "and scaling test rows by their OWN", "statistics changes what values mean"], "handk"))
    svg("preprocessing.svg", 960, 536, "".join(b))

# ---------------- Classification: boundaries, confusion matrix, CAP ----------------
if want("cls"):
    b = [header("CLASSIFICATION", "Seven models, one question: which side of the boundary?", 960)]
    b.append(T(40, 104, "1. how each model draws its boundary (age vs salary)", "h2"))
    cards = [("logistic", "a straight line"), ("linear SVM", "the widest-margin line"), ("k-NN", "vote of k neighbours"),
             ("kernel SVM", "curves via a kernel"), ("naive Bayes", "per-class bell curves"), ("decision tree", "axis-aligned boxes"), ("random forest", "average of many trees")]
    for k, (t1, t2) in enumerate(cards):
        x, y = 40 + (k % 4) * 225, 120 + (k // 4) * 118
        b.append(R(x, y, 205, 100, "cell", 6)); b.append(T(x + 102, y + 20, t1, "numb", "middle")); b.append(T(x + 102, y + 92, t2, "mut", "middle"))
        cx, cy = x + 102, y + 52
        if k in (0, 1): b.append(P(f"M{cx-60},{cy+22} L{cx+60},{cy-22}", "lnr"))
        if k == 1: b.append(P(f"M{cx-60},{cy+34} L{cx+60},{cy-10}", "lnd")); b.append(P(f"M{cx-60},{cy+10} L{cx+60},{cy-34}", "lnd"))
        if k == 2: b.append(P(f"M{cx-60},{cy+10} L{cx-30},{cy+16} L{cx-10},{cy-4} L{cx+15},{cy+8} L{cx+35},{cy-14} L{cx+60},{cy-6}", "lnr"))
        if k in (3, 4): b.append(P(f"M{cx-60},{cy-20} C {cx-20},{cy+40} {cx+20},{cy+40} {cx+60},{cy-20}", "lnr"))
        if k in (5, 6): b.append(P(f"M{cx-60},{cy+4} L{cx-10},{cy+4} L{cx-10},{cy-26} L{cx+60},{cy-26}" if k == 5 else f"M{cx-60},{cy+8} L{cx-30},{cy+8} L{cx-30},{cy-2} L{cx-5},{cy-2} L{cx-5},{cy-18} L{cx+25},{cy-18} L{cx+25},{cy-26} L{cx+60},{cy-26}", "lnr"))
        for dx, dy, c in [(-40, -20, "pnk"), (-25, -28, "pnk"), (35, 18, "grn"), (45, 4, "grn"), (-45, 20, "grn"), (30, -30, "pnk")]:
            b.append(C(cx + dx, cy + dy, 4, c))
    b.append(note(715, 238, 215, 100, "scale first", ["distances (k-NN, SVM)", "need comparable axes;", "trees don't care"], "handk"))
    b.append(T(40, 380, "2. confusion matrix (sklearn layout)", "h2"))
    labels = [["TN", "FP"], ["FN", "TP"]]; cls = [["grn", "pnk"], ["pnk", "grn"]]
    for i in range(2):
        for j in range(2):
            b.append(R(130 + j * 90, 396 + i * 50, 90, 50, cls[i][j])); b.append(T(175 + j * 90, 427 + i * 50, labels[i][j], "numb", "middle"))
    b.append(T(175, 512, "predicted 0", "mut", "middle")); b.append(T(265, 512, "predicted 1", "mut", "middle"))
    b.append(T(120, 425, "actual 0", "mut", "end")); b.append(T(120, 475, "actual 1", "mut", "end"))
    b.append(T(40, 540, "FP = type I (false alarm), FN = type II (missed)", "mut"))
    b.append(T(430, 380, "3. CAP curve", "h2"))
    ox, oy, w, h = 450, 520, 220, 120
    b.append(P(f"M{ox},{oy} L{ox+w},{oy} M{ox},{oy} L{ox},{oy-h}", "ln"))
    b.append(P(f"M{ox},{oy} L{ox+w},{oy-h}", "lnd")); b.append(P(f"M{ox},{oy} L{ox+w*0.1},{oy-h} L{ox+w},{oy-h}", "lnd"))
    b.append(P(f"M{ox},{oy} C {ox+30},{oy-90} {ox+90},{oy-115} {ox+w},{oy-h}", "lnr"))
    b.append(T(ox + w / 2, oy + 18, "share of people contacted (best scores first)", "mut", "middle"))
    b.append(T(ox + 60, oy - 98, "model", "hand")); b.append(T(ox + 150, oy - 40, "random", "mut")); b.append(T(ox + 30, oy - h - 6, "perfect", "mut"))
    b.append(note(700, 380, 230, 150, "accuracy paradox", ["2% positives: 'always no'", "scores 98% and is useless.", "Look at FN/FP, the CAP", "curve or AUC instead."], "handk"))
    svg("classification.svg", 960, 560, "".join(b))

# ---------------- Regression ----------------
if want("reg"):
    b = [header("REGRESSION", "Five ways to draw a curve through points, and how to score them", 960)]
    pts = [(1, 1.2), (2, 1.9), (3, 3.4), (4, 3.6), (5, 5.3), (6, 5.4), (7, 7.4)]
    def panel(x0, y0, title, sub):
        b.append(R(x0, y0, 280, 190, "cell", 6)); b.append(T(x0 + 140, y0 + 22, title, "numb", "middle")); b.append(T(x0 + 140, y0 + 182, sub, "mut", "middle"))
    def px(x0, y0, x, y): return x0 + 20 + x * 32, y0 + 160 - y * 17
    # 1 OLS
    panel(40, 96, "least squares", "minimise Σ (y − ŷ)²")
    for x, y in pts:
        cx, cy = px(40, 96, x, y); lx, ly = px(40, 96, x, x * 1.02)
        b.append(P(f"M{cx},{cy} L{lx},{ly}", "lnd")); b.append(C(cx, cy, 4, "pnk"))
    a, c = px(40, 96, 0.5, 0.51), px(40, 96, 7.5, 7.65); b.append(P(f"M{a[0]},{a[1]} L{c[0]},{c[1]}", "lnr"))
    # 2 SVR tube
    panel(340, 96, "SVR: ε-insensitive tube", "errors inside the tube are free")
    for off, cl in ((0, "lnr"), (1.0, "lnd"), (-1.0, "lnd")):
        a, c = px(340, 96, 0.5, 0.51 + off), px(340, 96, 7.5, 7.65 + off); b.append(P(f"M{a[0]},{a[1]} L{c[0]},{c[1]}", cl))
    for x, y in pts + [(3.5, 5.6)]:
        cx, cy = px(340, 96, x, y); b.append(C(cx, cy, 4, "pnk"))
    cx, cy = px(340, 96, 3.5, 5.6); lx, ly = px(340, 96, 3.5, 3.57 + 1.0); b.append(P(f"M{cx},{cy} L{lx},{ly}", "ln")); b.append(T(cx + 6, cy + 14, "slack ξ", "hand"))
    # 3 tree
    panel(640, 96, "tree / forest", "a constant per interval (forest: averaged)")
    steps = [(0.5, 2.5, 1.55), (2.5, 4.5, 3.5), (4.5, 6.5, 5.35), (6.5, 7.5, 7.4)]
    for x1, x2, v in steps:
        a, c = px(640, 96, x1, v), px(640, 96, x2, v); b.append(P(f"M{a[0]},{a[1]} L{c[0]},{c[1]}", "lnr"))
    for x, y in pts:
        cx, cy = px(640, 96, x, y); b.append(C(cx, cy, 4, "pnk"))
    b.append(note(40, 312, 430, 200, "Linear in what?", ["polynomial: y = b0 + b1·x + b2·x² + … + bn·x^n", "is still LINEAR regression: linear in the b's.", "", "multiple: y = b0 + b1·x1 + … + bk·xk", "dummy trap: keep one fewer dummy than", "categories (the intercept is the baseline)", "SVR needs scaled X and y; trees need none"], "handk"))
    b.append(note(500, 312, 430, 200, "Scoring", ["R² = 1 − SS_res / SS_tot", "(1 = perfect, 0 = as good as the mean,", " < 0 = worse than the mean on test data)", "", "adjusted R² = 1 − (1 − R²)(n − 1)/(n − k − 1)", "falls when a useless feature is added", "compare models on held-out data"], "handk"))
    svg("regression.svg", 960, 530, "".join(b))

# ---------------- Clustering ----------------
if want("clu"):
    b = [header("CLUSTERING", "Group points with no labels: centroids or a tree of merges", 960)]
    b.append(T(40, 104, "1. k-means: assign, move, repeat", "h2"))
    import random as _r; _r.seed(1)
    groups = [((90, 200), "pnk"), ((200, 150), "grn"), ((160, 260), "blu")]
    for (gx, gy), c in groups:
        for _ in range(9): b.append(C(gx + _r.uniform(-30, 30), gy + _r.uniform(-25, 25), 4, c))
        b.append(R(gx - 7, gy - 7, 14, 14, c, 2))
    b.append(P("M60,140 L90,193", "lnd", "ar")); b.append(T(40, 136, "centroid moves to its points' mean", "hand"))
    b.append(note(40, 310, 380, 196, "Choosing k", ["WCSS = Σ distance(point, its centroid)²", "always falls as k grows (0 at k = n)", "elbow: where the drop flattens", "silhouette: peaks at a good k", "", "k-means++ seeding: next centroid drawn", "with probability proportional to distance²;", "still random — run several starts"], "handk"))
    b.append(T(470, 104, "2. hierarchical: merge the closest, draw the tree", "h2"))
    base = 300; xs = [490, 530, 570, 610, 680, 720, 790, 830, 870]
    for x in xs: b.append(C(x, base + 12, 4, "cell"))
    def merge(x1, x2, h1, h2, h):
        b.append(P(f"M{x1},{base - h1} L{x1},{base - h} L{x2},{base - h} L{x2},{base - h2}", "ln")); return (x1 + x2) / 2
    m1 = merge(490, 530, 0, 0, 20); m2 = merge(570, 610, 0, 0, 25); m3 = merge(m1, m2, 20, 25, 60)
    m4 = merge(680, 720, 0, 0, 18); m5 = merge(790, 830, 0, 0, 15); m6 = merge(m5, 870, 15, 0, 35)
    m7 = merge(m4, m6, 18, 35, 70); merge(m3, m7, 60, 70, 170)
    b.append(P("M475,190 L900,190", "lnr")); b.append(T(905, 194, "cut", "hand"))
    b.append(T(470, 340, "cut across the longest vertical gap with no merge = number of clusters (here 2)", "mut"))
    b.append(note(470, 360, 460, 146, "Distance between clusters (linkage)", ["single: closest pair — follows chains and odd shapes", "complete: furthest pair — compact groups", "average: mean over all pairs", "Ward: the merge that adds least within-cluster variance", "(like k-means; the course's choice)"], "handk"))
    svg("clustering.svg", 960, 526, "".join(b))

# ---------------- Association rules ----------------
if want("assoc"):
    b = [header("ASSOCIATION RULES", "People who bought A also bought B — but more than usual?", 960)]
    b.append(R(40, 100, 380, 230, "cell", 6)); b.append(T(230, 122, "100 viewers", "numb", "middle"))
    b.append(f'<circle cx="190" cy="220" r="80" class="yel"/>'); b.append(f'<circle cx="270" cy="230" r="42" class="blu" opacity="0.8"/>')
    b.append(T(150, 222, "Interstellar 40", "num", "middle")); b.append(T(292, 240, "Ex Machina 10", "num", "middle")); b.append(T(244, 200, "both 7", "numb", "middle"))
    b.append(note(460, 100, 470, 230, "Three numbers per rule A → B", ["support(B) = share of all baskets containing B = 10%", "confidence(A -> B) = P(B | A) = 7 / 40 = 17.5%", "lift = confidence / support(B) = 1.75", "", "lift > 1: A makes B more likely than usual", "Apriori: keep itemsets above min support,", "grow them, keep rules above min confidence/lift", "Eclat: support of itemsets only, no rules"], "handk"))
    b.append(note(40, 356, 890, 100, "Setting thresholds (the course)", ["min support from the business: an item bought 3 times a day, over a week of 7,501 baskets: 3·7/7501 ≈ 0.003;", "min confidence 0.2 (0.8 found no rules, 0.4 a few hundred); min lift 3. Rank rules by lift; check that the sample is large."], "handk"))
    svg("association-rules.svg", 960, 476, "".join(b))

# ---------------- Bag of words ----------------
if want("bow"):
    b = [header("BAG OF WORDS", "Text to a row of word counts, then any classifier", 960)]
    stages = [("review", "\"The crust is not good!\"", "cell"), ("clean", "letters only, lower case", "grey"), ("stop words", "drop 'the', 'is' — keep 'not'", "yel"),
              ("stem", "loved -> love", "yel"), ("count vector", "one column per word", "blu")]
    for k, (t1, t2, c) in enumerate(stages):
        x = 30 + k * 186
        b.append(R(x, 100, 166, 60, c, 6)); b.append(T(x + 83, 124, t1, "h2", "middle")); b.append(T(x + 83, 144, t2, "mut", "middle"))
        if k < 4: b.append(A(x + 166, 130, x + 186, 130))
    words = ["crust", "good", "love", "not", "place", "...", "unknown"]; vals = ["1", "1", "0", "1", "0", "", "0"]
    for k, (w, v) in enumerate(zip(words, vals)):
        b.append(R(120 + k * 90, 200, 90, 36, "cell")); b.append(T(165 + k * 90, 223, v, "numb", "middle")); b.append(T(165 + k * 90, 252, w, "mut", "middle"))
    b.append(T(40, 223, "the row:", "h2")); b.append(T(760, 280, "1,500 most frequent words kept", "hand", "end"))
    b.append(note(40, 296, 430, 170, "Then classify", ["rows = reviews, columns = words: a sparse matrix", "naive Bayes (the course, Python),", "random forest (the course, R), logistic regression", "", "bag = order thrown away: 'not good' and", "'good, not bad' look alike unless you add", "word pairs (bigrams)"], "handk"))
    b.append(note(500, 296, 430, 170, "Where it sits", ["NLP overlap deep learning = deep NLP; seq2seq in it", "rule-based chatbots -> bag of words ->", "CNNs on word vectors -> seq2seq -> transformers", "", "the course's email example: a 20,000-slot", "count vector with start/end/unknown slots,", "feeding a yes/no reply classifier"], "handk"))
    svg("bag-of-words.svg", 960, 486, "".join(b))

# ---------------- Dimensionality reduction ----------------
if want("dim"):
    import random as _r; _r.seed(3)
    b = [header("DIMENSIONALITY REDUCTION", "Fewer, new features: most variance or best separation", 960)]
    b.append(R(40, 100, 420, 300, "cell", 6)); b.append(T(250, 122, "same data, two different 1-D axes", "numb", "middle"))
    for cy, c in ((230, "pnk"), (300, "grn")):
        for _ in range(26): b.append(C(250 + _r.uniform(-170, 170), cy + _r.uniform(-18, 18), 3.5, c))
    b.append(P("M60,265 L440,265", "lnr", "ar")); b.append(T(300, 386, "PCA: widest spread (left-right) — classes overlap", "hand", "middle"))
    b.append(P("M250,370 L250,150", "ln", "ar")); b.append(T(258, 160, "LDA: separates classes", "numb"))
    b.append(note(490, 100, 440, 140, "PCA (unsupervised)", ["standardise -> covariance -> eigenvectors", "keep the top k by eigenvalue (variance)", "new features = principal components,", "not a subset of the old ones"], "handk"))
    b.append(note(490, 256, 440, 144, "LDA (supervised) and kernel PCA", ["LDA: axes maximising between-class vs", "within-class scatter; at most (classes − 1)", "kernel PCA: PCA after an implicit non-linear", "mapping (kernel trick) for curved structure"], "handk"))
    b.append(note(40, 420, 890, 80, "Rule", ["fit the reducer on training rows only (like the scaler), then transform test rows;", "choose k by explained variance (PCA) or by cross-validated accuracy of the downstream model"], "handk"))
    svg("dimensionality-reduction.svg", 960, 520, "".join(b))

# ---------------- Model selection & boosting ----------------
if want("msel"):
    b = [header("MODEL SELECTION AND BOOSTING", "Estimate honestly, tune on folds, then boost", 960)]
    b.append(T(40, 104, "1. 10-fold cross-validation (inside the training set)", "h2"))
    for r in range(5):
        for k in range(10):
            b.append(R(40 + k * 38, 120 + r * 26, 36, 22, "pnk" if k == r else "grn", 2))
        b.append(T(430, 136 + r * 26, f"fit {r + 1}: score fold {r + 1}", "mut"))
    b.append(T(40, 262, "… 10 fits; report the mean and spread; test set touched once at the end", "mut"))
    b.append(T(560, 104, "2. bias vs variance (10 fits as darts)", "h2"))
    labels = [("high bias", "low variance", -28, 0, 6), ("low bias", "high variance", 0, 0, 24), ("high bias", "high variance", -24, 10, 24), ("low bias", "low variance", 0, 0, 6)]
    _r = __import__("random"); _r.seed(5)
    for k, (l1, l2, dx, dy, spread) in enumerate(labels):
        cx, cy = 600 + (k % 2) * 170, 150 + (k // 2) * 98
        for rr in (36, 23, 10): b.append(C(cx, cy, rr, "cell"))
        for _ in range(10): b.append(C(cx + dx + _r.uniform(-spread, spread), cy + dy + _r.uniform(-spread, spread) * 0.8, 2.5, "pnk"))
        b.append(T(cx + 48, cy - 6, l1, "mut")); b.append(T(cx + 48, cy + 10, l2, "mut"))
    b.append(T(40, 310, "3. boosting: each new tree fits what the ensemble still gets wrong", "h2"))
    for k in range(4):
        x = 40 + k * 230
        b.append(R(x, 326, 190, 56, "yel" if k < 3 else "grn", 6)); b.append(T(x + 95, 350, f"tree {k + 1}" if k < 3 else "sum of trees", "numb", "middle"))
        b.append(T(x + 95, 370, ["fit the target", "fit residuals of 1", "fit residuals of 1+2", "prediction"][k], "mut", "middle"))
        if k < 3: b.append(A(x + 190, 354, x + 230, 354))
    b.append(note(40, 400, 430, 120, "Grid search", ["try every combination (C, kernel, gamma)", "score each with k-fold CV, keep the best", "its best CV score is optimistic:", "use a test set or nested CV to report"], "handk"))
    b.append(note(500, 400, 430, 120, "XGBoost", ["gradient-boosted trees with regularisation,", "shrinkage (learning rate) and fast splitting;", "strong default for tabular data —", "but not always the winner (see snippet)"], "handk"))
    svg("model-selection-boosting.svg", 960, 540, "".join(b))

# ---------------- AWS ML lifecycle ----------------
if want("aws"):
    b = [header("ML ON AWS", "Parts 11-15 as one lifecycle, a service per step", 960)]
    stages = [("prepare (Part 11)", ["S3: store raw data", "Glue: ETL jobs", "DataBrew: no-code cleaning", "Data Wrangler (Canvas): clean,", "  encode, scale, PCA"], "yel"),
              ("develop (Part 12)", ["Canvas: AutoML regression /", "  classification, metrics", "built-in XGBoost, tuning", "Model Registry: versions", "AI services: Comprehend, ..."], "blu"),
              ("deploy (Part 13)", ["endpoints: real-time,", "  serverless, async, batch", "containers: ECR, ECS, EKS", "Lambda for light models", "IaC: CloudFormation, CDK"], "grn"),
              ("automate (Part 14)", ["CodePipeline orchestrates:", "  source -> CodeBuild ->", "  CodeDeploy", "SageMaker Pipelines: process,", "  train, register, deploy"], "pnk"),
              ("monitor (Part 15)", ["Model Monitor: drift, quality", "Clarify: bias, explanations", "CloudWatch, X-Ray, Trail", "EventBridge, QuickSight", "Budgets, Spot, Savings Plans"], "grey")]
    for k, (title, lines, c) in enumerate(stages):
        x = 30 + k * 186
        b.append(R(x, 100, 172, 200, c, 6)); b.append(T(x + 86, 124, title, "numb", "middle"))
        for i, l in enumerate(lines): b.append(T(x + 10, 152 + i * 26, l, "mut"))
        if k < 4: b.append(A(x + 172, 200, x + 186, 200))
    b.append(P("M890,300 L890,330 L116,330 L116,302", "lnd", "ar")); b.append(T(500, 348, "drift or new data -> retrain", "hand", "middle"))
    b.append(note(30, 372, 440, 130, "Inference options (as taught)", ["real-time: low latency, always-on instance", "serverless: pay per request, cold starts", "asynchronous: large payloads, queue, long jobs", "batch transform: score a whole dataset offline"], "handk"))
    b.append(note(500, 372, 430, 130, "Cost habits from the course", ["delete Canvas app, user profile, domain,", "endpoints, endpoint configs and models;", "zero-spend and monthly budgets;", "managed Spot training for interruptible jobs"], "handk"))
    svg("aws-ml-lifecycle.svg", 960, 522, "".join(b))
