"""Diagrams for wiki/machine-learning/ml-systems notes.  Run: python3 scripts/ml_systems_diagrams.py wiki/machine-learning/ml-systems/assets [score ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_svg as ps
from paper_svg import T, R, A, P, C, header, note
ps.OUT = sys.argv[1]
os.makedirs(ps.OUT, exist_ok=True)
svg = ps.svg
WHICH = sys.argv[2:] or ["all"]
def want(n): return "all" in WHICH or n in WHICH

# ---------------- Recommenders: the ladder of scoring rules ----------------
if want("score"):
    b = [header("RECOMMENDERS", "One ladder of scores, and the axis that is not a rung", 960)]
    rungs = [("Popularity", ["s = count(j)"], ["nothing - this is", "the baseline"]),
             ("Age-decayed", ["s = votes^0.8", "/ (age+2)^1.8"], ["stale items holding", "the top forever"]),
             ("Raw mean rating", ["s = (sum r) / n"], ["item quality - but", "breaks at small n"]),
             ("Smoothed mean", ["(sum r + lambda*mu0)", "/ (n + lambda)"], ["small n: pulls", "toward a prior"]),
             ("Lower bound", ["5th pct of the", "posterior"], ["small n,", "pessimistically"]),
             ("Sampled", ["draw from the", "posterior"], ["small n AND", "it explores"])]
    # bracket over rungs 2-6: they are all one correction
    b.append(T(556, 100, "rungs 2 and 4-6: the same correction - distrust small samples, distrust old evidence", "mut", "middle"))
    b.append(P("M182,116 L182,108 L930,108 L930,116", "lnd"))
    for k, (title, formula, fixes) in enumerate(rungs):
        x = 30 + k * 152
        # rung 3 is the trap, not a rung: tint it instead of crossing it out, which would strike through its own formula
        b.append(R(x, 120, 140, 92, "pnk" if k == 2 else "yel", 6))
        b.append(T(x + 70, 144, title, "numb", "middle"))
        for i, f in enumerate(formula):
            b.append(T(x + 70, 170 + i * 16, f, "num", "middle"))
        for i, c in enumerate(fixes):
            b.append(T(x + 70, 230 + i * 16, c, "mut", "middle"))
    b.append(T(404, 266, "small-n trap", "hand", "middle"))
    # the step down is a change of axis, not a seventh rung
    b.append(A(480, 274, 480, 308))
    b.append(T(492, 296, "a different axis", "hand"))
    b.append(R(130, 312, 700, 72, "blu", 6))
    b.append(T(480, 336, "Personalised", "h2", "middle"))
    b.append(T(480, 356, "s(i,j) = r_bar_i + weighted average of neighbours' deviations", "num", "middle"))
    b.append(T(480, 376, "same sorting problem - one score per user, not one per item", "mut", "middle"))
    b.append(note(30, 404, 900, 86, "The one idea", [
        "Rungs 2 and 4-6 are all the SAME correction applied to rung 3: distrust a small sample, distrust old evidence.",
        "Personalisation is a different axis, not a seventh rung - it changes whose score, not how you smooth it."], "handk"))
    svg("rs-score-ladder.svg", 960, 500, "".join(b))

# ---------------- Recommenders: user-user vs item-item is one matrix, two readings ----------------
if want("cf"):
    b = [header("COLLABORATIVE FILTERING", "One matrix, two directions to read it", 960)]
    b.append(T(30, 96, "same 6 x 4 ratings matrix on both sides; ? = unrated; the pink cell is the target r(u3,i3)", "mut"))
    b.append(P("M480,112 L480,446", "lnd"))
    vals = [["4", "?", "5", "3"],
            ["5", "2", "4", "?"],
            ["3", "4", "?", "2"],
            ["?", "5", "3", "4"],
            ["4", "3", "5", "?"],
            ["2", "?", "4", "5"]]
    cw, ch, gy = 40, 28, 160
    def grid(gx):
        out = []
        for j in range(4):
            out.append(T(gx + j * cw + cw / 2, 152, "i%d" % (j + 1), "mut", "middle"))
        for i, row in enumerate(vals):
            out.append(T(gx - 6, gy + i * ch + 18, "u%d" % (i + 1), "mut", "end"))
            for j, v in enumerate(row):
                cls = "pnk" if (i, j) == (2, 2) else "cell"
                out.append(R(gx + j * cw, gy + i * ch, cw, ch, cls))
                out.append(T(gx + j * cw + cw / 2, gy + i * ch + 19, v, "num", "middle"))
        out.append(R(gx + 2 * cw, gy + 2 * ch, cw, ch, "hi"))
        return "".join(out)

    # left: user-user. Two ROWS are the unit of comparison.
    b.append(T(30, 124, "user-user: who is like me?", "h2"))
    b.append(grid(82))
    for i in (1, 2):
        b.append(R(82, gy + i * ch, 4 * cw, ch, "hi"))
    b.append(P("M250,188 L262,188 L262,244 L250,244 M262,216 L274,216", "lnr"))
    b.append(T(280, 212, "w from co-rated items", "num"))
    b.append(T(280, 228, "short list -> noisy w", "mut"))
    b.append(T(82, 420, "all pairs: O(N^2 M)", "num"))
    b.append(T(82, 436, "N = 10^5 -> 40 GB of weights", "mut"))

    # right: item-item. Two COLUMNS are the unit of comparison.
    b.append(T(490, 124, "item-item: what is like what I liked?", "h2"))
    b.append(grid(542))
    for j in (1, 2):
        b.append(R(542 + j * cw, gy, cw, 6 * ch, "hi"))
    b.append(P("M586,336 L586,348 L658,348 L658,336 M622,348 L622,360", "lnr"))
    b.append(T(622, 378, "w from co-rating users", "num", "middle"))
    b.append(T(622, 394, "long list -> steadier w", "mut", "middle"))
    b.append(T(542, 420, "all pairs: O(M^2 N)", "num"))
    b.append(T(542, 436, "cheaper whenever N > M", "mut"))

    b.append(note(30, 460, 900, 86, "The one idea", [
        "Same matrix, same formula - the only thing that changes is the direction you read it in.",
        "Columns win twice: more co-raters behind each weight, and M^2 < N^2 pairs whenever N > M - the usual case, not a law."], "handk"))
    svg("rs-user-vs-item-cf.svg", 960, 560, "".join(b))

# ---------------- Matrix factorization: ALS alternates two convex solves ----------------
if want("als"):
    b = [header("MATRIX FACTORIZATION", "ALS: a non-convex problem cut into two convex halves", 960)]
    for x in (225, 615):
        b.append(P("M%d,112 L%d,470" % (x, x), "lnd"))

    # (1) the data: only Omega is known
    b.append(T(30, 124, "1. the data", "h2"))
    b.append(T(36, 150, "R: N users x M items", "mut"))
    filled = ["4", "2", "5", "3", "4", "1", "5", "3"]
    k = 0
    for i in range(7):
        for j in range(6):
            seen = (i * 7 + j * 3) % 6 == 0
            b.append(R(40 + j * 26, 158 + i * 20, 26, 20, "grn" if seen else "grey"))
            if seen:
                b.append(T(53 + j * 26, 172 + i * 20, filled[k % len(filled)], "num", "middle")); k += 1
    for i, s in enumerate(["only Omega (the observed", "cells) is known - under 1%", "filled in MovieLens-20M"]):
        b.append(T(36, 322 + i * 16, s, "mut"))

    # (2) the model: one cell at a time, never the product
    b.append(T(230, 124, "2. the model", "h2"))
    b.append(R(250, 158, 36, 110, "cell")); b.append(R(250, 158, 36, 22, "pur"))
    b.append(T(246, 174, "w_i", "num", "end"))
    b.append(T(268, 284, "W: N x K", "num", "middle")); b.append(T(268, 298, "(users)", "mut", "middle"))
    b.append(T(294, 216, "x", "ink", "middle"))
    b.append(R(306, 158, 100, 36, "cell")); b.append(R(386, 158, 20, 36, "pur"))
    b.append(T(396, 150, "u_j", "num", "middle"))
    b.append(T(356, 212, "U^T: K x M", "num", "middle")); b.append(T(356, 226, "(items)", "mut", "middle"))
    b.append(T(419, 216, "=", "ink", "middle"))
    for i in range(5):
        for j in range(5):
            b.append(R(432 + j * 20, 158 + i * 20, 20, 20, "pnk" if (i, j) == (0, 4) else "cell"))
    b.append(R(512, 158, 20, 20, "hi"))
    b.append(T(482, 276, "one cell of R-hat", "mut", "middle"))
    b.append(T(400, 320, "r_hat(i,j) = w_i . u_j + b_i + c_j + mu", "num", "middle"))
    b.append(P("M548,316 L560,316 L560,192", "lnd"))
    b.append(A(560, 192, 537, 182, "lnr", "arr"))
    b.append(T(240, 344, "K = the inner dim summed over = latent factors (10-50)", "mut"))
    # the X gets an empty box; the label sits beside it, or the strokes strike through their own text
    b.append(R(300, 364, 110, 50, "grey", 6))
    b.append(P("M306,368 L404,410", "lnr")); b.append(P("M404,368 L306,410", "lnr"))
    b.append(T(422, 386, "W @ U^T", "numb")); b.append(T(422, 402, "(N x M)", "mut"))
    b.append(T(420, 434, "never materialise this", "hand", "middle"))

    # (3) the loop: freeze one side, the other side is a ridge solve
    b.append(T(620, 124, "3. the training loop", "h2"))
    half = [("U", "W", ["each row of W is an independent", "ridge regression (an exact solve):",
                        "(sum u_j u_j^T + lambda I) w_i", "  = sum (r_ij - b_i - c_j - mu) u_j",
                        "-> one np.linalg.solve per user"]),
            ("W", "U", ["each row of U is an independent", "ridge regression (an exact solve):",
                        "(sum w_i w_i^T + lambda I) u_j", "  = sum (r_ij - b_i - c_j - mu) w_i",
                        "-> one np.linalg.solve per item"])]
    for s, (froz, free, lines) in enumerate(half):
        y = 152 + s * 152
        b.append(R(620, y, 284, 136, "cell", 6))
        b.append(R(634, y + 10, 78, 26, "grey", 4))
        b.append(T(673, y + 27, "freeze " + froz, "num", "middle"))
        b.append(A(716, y + 23, 736, y + 23))
        b.append(R(740, y + 10, 78, 26, "pur", 4))
        b.append(T(779, y + 27, "solve " + free, "num", "middle"))
        for i, l in enumerate(lines):
            b.append(T(634, y + 54 + i * 16, l, "mut" if i in (0, 1, 4) else "num"))
    b.append(A(762, 288, 762, 302))
    b.append(P("M904,372 L918,372 L918,220 L904,220", "lnr", "arr"))
    b.append(T(762, 460, "repeat; loss never increases", "hand", "middle"))

    b.append(note(30, 482, 900, 86, "The one idea", [
        "Jointly over (W, U) the loss is NOT convex - the product w_i . u_j makes it bilinear in the parameters.",
        "Freeze one side and every row of the other is an exact convex ridge solve, so the loss can only go down.",
        "Price: ALS promises a stationary point, not the global optimum - different seeds land in different places."], "handk"))
    svg("mf-als-alternation.svg", 960, 590, "".join(b))

# ---------------- Exploration: three things you can be uncertain about ----------------
if want("eg"):
    b = [header("EXPLORATION", "Three things a recommender can be uncertain about", 960)]
    panels = [(30, "Reward uncertainty", "UCB, Thompson sampling", "blu",
               "\"Will THIS item pay off?\"", ["widens the confidence", "band on one arm;", "says nothing about", "the rest of the catalogue"]),
              (355, "Surface difference", "diversity, novelty", "yel",
               "\"Is this unlike what I just showed?\"", ["a property of the slate,", "computed from features;", "no model of the user", "enters the decision"]),
              (680, "Belief shift", "epistemic gain", "pnk",
               "\"What would the ANSWER teach me?\"", ["re-encodes the user both ways", "and measures how far the", "WHOLE catalogue's relevance", "distribution moves"])]
    for x, title, sub, col, q, lines in panels:
        b.append(R(x, 92, 250, 300, "note", 8))
        b.append(T(x + 125, 118, title, "h2", "middle")); b.append(T(x + 125, 136, sub, "mut", "middle"))
        b.append(T(x + 125, 160, q, "hand", "middle"))
        for k, l in enumerate(lines): b.append(T(x + 16, 334 + k * 15, l, "mut"))
    # panel 1: one arm with a wide band, others narrow
    for k, (w, lab) in enumerate([(14, "A"), (52, "B"), (10, "C")]):
        y = 196 + k * 40
        b.append(T(46, y + 4, lab, "mut"))
        b.append(R(70, y - 6, 150, 20, "cell", 3))
        b.append(R(145 - w, y - 6, 2 * w, 20, "blu", 3))
        b.append(C(145, y + 4, 3, "ink"))
    b.append(T(46, 312, "B is unexplored -> show B", "handk"))
    # panel 2: spread-out slate
    pts = [(420, 210), (520, 196), (470, 262), (560, 266), (405, 286)]
    for px, py in pts: b.append(C(px, py, 11, "yel"))
    for i in range(len(pts) - 1):
        b.append(P(f"M{pts[i][0]},{pts[i][1]} L{pts[i+1][0]},{pts[i+1][1]}", "lnd"))
    b.append(T(371, 312, "spread the slate -> show the odd one", "handk"))
    # panel 3: two distributions, before and after
    for row, (yy, lab, hs, col) in enumerate([(205, "now", [26, 10, 5, 4, 3], "cell"),
                                              (268, "if engaged", [15, 9, 20, 7, 12], "pnk")]):
        b.append(T(700, yy + 6, lab, "mut"))
        for k, h in enumerate(hs):
            b.append(R(762 + k * 24, yy + 14 - h, 18, h, col, 2))
        b.append(P(f"M758,{yy + 16} L{762 + 5 * 24},{yy + 16}", "lnd"))
    b.append(A(884, 226, 884, 252, "lnr", "arr"))
    b.append(T(696, 312, "how far it moved = EG", "handk"))
    b.append(note(30, 404, 900, 140, "The distinction that matters",
                  ["A bandit asks about the item it is about to show. Epistemic gain asks about every OTHER item:",
                   "one answer re-encodes the user, which moves the predicted relevance of the whole catalogue.",
                   "EG = JS(now, if-engaged)  [+ JS(now, if-ignored), which is 0 when the model only records positives]",
                   "Caveat it cannot fix: EG scores how far the belief MOVES, never whether it moves toward the truth."], "handk"))
    svg("rs-exploration-targets.svg", 960, 560, "".join(b))
