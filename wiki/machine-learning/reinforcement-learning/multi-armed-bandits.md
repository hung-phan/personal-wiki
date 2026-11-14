---
name: multi-armed-bandits
description: Multi-armed bandits for ad and A/B-test optimisation with runnable numpy code — explore vs exploit and regret, why A/B testing is pure exploration, Upper Confidence Bound (mean + sqrt(3/2 · ln n / N), the course's constant vs UCB1's 2), Thompson sampling with Beta(clicks + 1, misses + 1) posteriors and how they narrow with data, random vs UCB vs Thompson on 10 ads over 10,000 and 500 rounds averaged over 20 seeds, deterministic vs probabilistic and delayed feedback, the course's Ads_CTR_Optimisation results in Python and R (random ≈ 1,200 clicks, UCB 2,178, Thompson ≈ 2,600; ad 5 best; UCB wrong at 500 rounds), the 1e400 trick, and why real logs don't look like the course's dataset. Study guide for Machine Learning A-Z, Part 6 (sections 31–33). Also covers ranking a whole list by sampled posterior scores from Recommender Systems and Deep Learning in Python, the frequentist confidence-interval baseline, the Beta-Bernoulli and Gaussian conjugate updates, and why a smoothed average rating IS a posterior mean.
created: 2026-11-13
updated: 2026-11-13
tags: [recommender-systems-course, reinforcement-learning, multi-armed-bandit, ucb, thompson-sampling, ab-testing, explore-exploit, bayesian, numpy, study-guide, ml-a-z]
---
# Multi-Armed Bandits: UCB and Thompson Sampling

> **How to read this note.** Study guide for *Machine Learning A-Z*, Part 6 (sections 31–33, which the course calls reinforcement learning): the multi-armed bandit problem, Upper Confidence Bound and Thompson sampling, each built in Python and R on `Ads_CTR_Optimisation.csv` — 10,000 users × 10 ad versions, with a 1 where that user would click that ad. The snippets simulate ads with fixed click rates. The quizzes aren't included. ε-greedy on a bandit is in [rl-foundations-mc-td](rl-foundations-mc-td.md) §4; full RL with states is in [q-learning](q-learning.md).

## Why This Exists

**Problem.** You have 10 versions of an ad and want the one people click most. Showing each equally until you're sure (an A/B test) wastes thousands of impressions on bad versions; going with the early leader risks locking in a fluke.

**Key insight.** Decide each round from what you've seen so far: either be **optimistic** — show the ad whose click rate *could* plausibly be highest (UCB) — or **sample** a plausible click rate for each ad from your current beliefs and show the winner (Thompson). Both explore uncertain ads and shift to the best one as evidence accumulates.

**Reach for this when:**
- Online choice among a few options with immediate feedback: ads, headlines, recommendations, pricing tests.
- You care about the reward collected *during* the experiment, not just the final answer.

**Don't reach for this when:**
- Actions change future situations (states) — that's full RL ([q-learning](q-learning.md)).
- The best option depends on who the user is — use contextual bandits.
- You need a clean causal estimate with fixed sample sizes — a standard A/B test is simpler to analyse.

## Diagrams

![UCB: each ad's average click rate with an upper bound that is wide for rarely shown ads; the ad with the highest bound is picked. Thompson: Beta posteriors that narrow with more shows; sample one rate per ad and show the highest](assets/multi-armed-bandits.svg)

## 1. The problem

A **multi-armed bandit** is a row of slot machines ("one-armed bandits") with unknown payout rates; each pull is a trial. Every pull of a worse machine costs you the difference — **regret** — but without pulling the others you'll never know they're worse. An A/B test explores all arms equally and only exploits afterwards; bandits blend the two continuously. The course's simplification: each ad has a fixed click-through rate; real rates drift (seasonality, ad fatigue).

## 2. Upper Confidence Bound

For each ad $i$ after $n$ rounds, with $N_i$ shows and average reward $\bar r_i$:

$$\text{UCB}_i = \bar r_i + \sqrt{\frac{3}{2} \cdot \frac{\ln n}{N_i}}$$

Show the ad with the highest bound. Rarely shown ads have wide bonuses, so they get tried; as $N_i$ grows the bonus shrinks and the average takes over. Untried ads get an infinite bound so each is shown once first (the course sets it to `1e400`, which Python and R both read as infinity). The constant 3/2 is the course's; the textbook UCB1 uses 2.

## 3. Thompson sampling

Treat each ad's click rate as unknown with a uniform prior. After $c$ clicks and $m$ misses, the posterior is $\text{Beta}(c + 1, m + 1)$. Each round, draw one sample from each ad's posterior and show the ad with the highest draw. The posterior narrows as evidence grows:

```python
import numpy as np
from scipy import stats
for clicks, shows in [(3, 10), (30, 100), (300, 1000)]:
    post = stats.beta(clicks + 1, shows - clicks + 1)    # posterior for the click rate after a uniform prior
    lo, hi = post.ppf([0.05, 0.95])                       # 90% credible interval
    print(f"{clicks:3d}/{shows:<4d} clicks: CTR around {post.mean():.3f}, 90% interval [{lo:.3f}, {hi:.3f}]")
# Expected output:
#   3/10   clicks: CTR around 0.333, 90% interval [0.135, 0.564]
#  30/100  clicks: CTR around 0.304, 90% interval [0.232, 0.381]
# 300/1000 clicks: CTR around 0.300, 90% interval [0.277, 0.324]
```

Ten shows leave the rate anywhere from 14% to 56%; a thousand pin it to ±2 points. Ads with wide posteriors sometimes draw high values and get explored; clearly bad ads almost never do.

## 4. Comparing them

```python
import numpy as np
ctr = np.array([0.05, 0.13, 0.09, 0.16, 0.27, 0.07, 0.11, 0.21, 0.05, 0.10])   # 10 ads' true click rates (unknown to us)
def run(policy, rounds, seed):
    rng = np.random.default_rng(seed); N, S, total = np.zeros(10), np.zeros(10), 0   # shows and clicks per ad
    for n in range(rounds):
        if policy == "random": a = rng.integers(10)
        elif policy == "ucb":                           # mean + confidence bonus; untried ads first
            with np.errstate(divide="ignore", invalid="ignore"):
                ucb = np.where(N > 0, S / N + np.sqrt(1.5 * np.log(n + 1) / N), np.inf)   # the course's 3/2 constant
            a = int(np.argmax(ucb))
        else:                                           # Thompson: sample a plausible CTR per ad from Beta(clicks+1, misses+1)
            a = int(np.argmax(rng.beta(S + 1, N - S + 1)))
        r = rng.random() < ctr[a]; N[a] += 1; S[a] += r; total += r
    return total, int(np.argmax(N))
for rounds in (10_000, 500):
    print(f"--- {rounds} rounds (best possible ≈ {ctr.max() * rounds:.0f} clicks, best ad = index 4)")
    for p in ("random", "ucb", "thompson"):
        res = [run(p, rounds, s) for s in range(20)]
        print(f"{p:9s} mean clicks {np.mean([t for t, _ in res]):7.0f}   found ad 4 in {sum(b == 4 for _, b in res)}/20 runs")
# Expected output:
# --- 10000 rounds (best possible ≈ 2700 clicks, best ad = index 4)
# random    mean clicks    1244   found ad 4 in 2/20 runs
# ucb       mean clicks    2227   found ad 4 in 20/20 runs
# thompson  mean clicks    2588   found ad 4 in 20/20 runs
# --- 500 rounds (best possible ≈ 135 clicks, best ad = index 4)
# random    mean clicks      60   found ad 4 in 1/20 runs
# ucb       mean clicks      75   found ad 4 in 17/20 runs
# thompson  mean clicks      88   found ad 4 in 11/20 runs
```

Over 10,000 rounds both methods find the best ad every time; Thompson collects ~360 more clicks than UCB (closer to the 2,700 a clairvoyant would get) because it stops wasting shows on clearly worse ads sooner. Over 500 rounds UCB more often ends with the right ad as its most-shown, but Thompson still earns more clicks — it exploits good-enough ads while uncertain. The course's numbers on its dataset: random selection ≈ 1,200–1,250 clicks; UCB 2,178 and Thompson ≈ 2,600 (R), both settling on ad 5 (index 4 in Python); in Python UCB still finds the right ad at 1,000 rounds but picks the wrong one at 500, while Thompson is right at 500.

**Why Thompson is usually preferred** (the course's comparison): UCB is deterministic and must be updated after every round; Thompson is random, so it tolerates **delayed or batched feedback** (a website collecting clicks in batches) and has better empirical results.

**The dataset caveat.** The course's file records every user's reaction to *every* ad — real logs only tell you about the ad you actually showed, which is why a bandit is needed at all. The simulation stands in for that: the algorithm reads only the cell of the ad it chose.

## 5. Ranking a whole list by sampled scores (*Recommender Systems and Deep Learning in Python*)

§3 asks *which single ad do I show next?* Sections 8–9 of *Recommender Systems and Deep Learning in Python* ask the ranking version: you have a page of items — the course's toy example is three movies — and you need an **order**. The answer is the same draw put to different use: sample one plausible score per item from its posterior, then **sort by the samples**. `argsort` where §3 had `argmax`, so one round of draws produces a whole page rather than one impression.

Why that is enough is best seen in two pictures the course draws. With three well-separated posteriors, the samples almost always come out in the same order as the posteriors themselves, so you simply render the right ranking. With one skinny, well-sampled posterior centred near 0.5 and one fat, barely-sampled posterior, the skinny one's draws land in a narrow band around 0.45–0.55, while the fat one has mass both above and below that band — so across hundreds of page views the uncertain item **sometimes surfaces above the known one**, collects data, and narrows, which returns you to the first picture.

So the ranking is random, but **"intelligently random"** in the course's phrase: the randomness is shaped by the posterior, which already encodes how much evidence you have. No $\varepsilon$, no confidence bonus, no schedule, and — the part worth internalising — no confidence interval is ever computed, yet you get the behaviour one would have bought you. The draw itself is the Thompson draw from §3; don't re-derive it.

### The frequentist baseline it is pitched against

The course first builds the textbook interval, as the thing to improve on. The MLE of a rate, $\hat\mu = \frac{1}{n}\sum_i x_i$, is a sum of random variables and therefore random itself; the central limit theorem gives $\hat\mu \sim \text{Normal}(\mu, \sigma^2/n)$. You cannot integrate that by hand, so you rescale to the standard normal and invert its CDF (`scipy.stats.norm.ppf`); a two-sided 95% interval leaves 2.5% in each tail, at $z = \pm 1.96$:

$$\hat\mu \pm 1.96\,\frac{\hat\sigma}{\sqrt{n}} \qquad\qquad \hat p \pm 1.96\sqrt{\frac{\hat p\,(1 - \hat p)}{n}}$$

The variance of the estimate is quadratic in $\sigma$ but only linear in $n$ — the instructor's point being that a small rise in the data's spread costs you many more samples than you expect. The interval shrinks only as $1/\sqrt{n}$, and, more to the point for a ranker, it hands you a pair of numbers you still have to turn into an ordering by hand.

### The two conjugate updates, in the form the course gives them

**Beta–Bernoulli.** Likelihood $x \sim \text{Bernoulli}(\pi)$, prior $\pi \sim \text{Beta}(\alpha, \beta)$. The denominator of Bayes' rule is a constant once $\pi$ is integrated out, so `posterior ∝ likelihood × prior`; the powers of $\pi$ and $(1-\pi)$ combine by ordinary algebra and the result is another Beta:

$$\alpha' = \alpha + \sum_i x_i, \qquad \beta' = \beta + n - \sum_i x_i$$

$\alpha$ counts successes, $\beta$ counts failures. $\text{Beta}(1,1)$ *is* the uniform distribution, which is where §3's $\text{Beta}(\text{clicks}+1, \text{misses}+1)$ comes from — the `+1, +1` is a flat prior, not a smoothing fudge.

**Gaussian with known precision.** The course works in **precision** $\tau = 1/\sigma^2$ because the algebra is easier. With $x \sim \text{Normal}(\mu, \tau)$, $\tau$ known, and prior $\mu \sim \text{Normal}(m, \lambda)$:

$$\lambda' = \lambda + n\tau, \qquad m' = \frac{\lambda m + \tau \sum_i x_i}{\lambda + n\tau}$$

Read the limits and the names explain themselves: precision grows **linearly in n**, so the variance goes to zero; at $n = 0$ the update collapses to $m' = m$ (no data, trust the prior); with plenty of data the sample mean takes over. The two terms in the numerator are weighted by $\lambda$ and $\tau$ — a confident prior pulls harder, precise observations pull harder. This is also the worked answer to the "unbounded rewards with Beta posteriors" pitfall below: for a star rating or a dwell time, this is the posterior you sample from.

**Online updating.** Both updates are plain addition, so you never wait for a batch: start from the prior, observe one $x$, add it in, and **the posterior you just computed is the prior for the next observation**. §4's loop already does this with counts; the point is that it is not an approximation or a streaming trick — the sequentially-updated posterior equals the one you would have computed from the whole batch at once.

### The payoff: smoothing *is* Bayes

Divide the numerator and denominator of $m'$ by $\tau$:

$$m' = \frac{(\lambda/\tau)\,m + \sum_i x_i}{(\lambda/\tau) + n}$$

That is the **smoothed average rating** — the familiar trick of pulling a sparsely-rated item's mean toward a global prior with a damping constant, so three five-star reviews don't outrank three hundred four-star ones. The same holds for the Beta: its posterior mean is $(\alpha + \sum_i x_i)/(\alpha + \beta + n)$, which is add-one smoothing with $\alpha$ and $\beta$ acting as pseudo-counts.

So the smoothing trick is not a hack with a hand-tuned fudge factor. It is **exactly** the posterior mean, and the "fudge factor" $\lambda/\tau$ is a real quantity: how confident you were before seeing data, divided by how precise each observation is. [recommender-systems](../ml-systems/recommender-systems.md) demonstrates this numerically on ratings data — the identity is worth seeing in numbers once, and there is no reason to repeat it here.

### What to distrust

- **Two slips in the lectures.** At the exact moment the Beta–Bernoulli punchline lands, the course says the posterior "is a **Bernoulli** distribution with a new alpha and beta" (and, a line earlier, that both are Bernoulli distributions). It is a **Beta**. The algebra on either side is correct; this is a misspeak, but it is the one sentence a reader would copy. Separately, the confidence-interval lecture reads out $z_{\text{right}} = 0.196$ where **1.96** is meant — the same breath gives $-1.96$ correctly for the left tail.
- **That interval is the Wald interval.** $\hat p \pm 1.96\sqrt{\hat p (1-\hat p)/n}$ is badly calibrated exactly where a ranker lives: small $n$, or rates near 0 or 1 (where it can even run outside $[0,1]$). The course flags only the t-distribution refinement for the Gaussian case, not this. Wilson or Agresti–Coull is the standard fix; the Beta posterior's credible interval does the same job without the approximation.
- **Stationarity and position bias are untouched.** "Precision → ∞, variance → 0" holds only if the click rate is constant. A posterior that never forgets becomes overconfident on a drifting rate and stops exploring — see the drift pitfall below, which these lectures do not address. Nor does anything here model **position bias**: an item ranked first is clicked more *because* it is first, so the feedback you fold into the posterior is contaminated by the ranking that produced it. Ranking a full list by independent per-item draws is done in practice, but the guarantees the course discusses are the single-arm ones.
- **No result numbers.** The course's code simulates three items with true rates 0.2, 0.5 and 0.75 and plots the posteriors out to 2,000 trials. It reports **no reward total, regret, or win rate** — only that the best item's posterior sharpens and marches toward the truth while the other two stay fat, because once they stop being sampled their credible intervals stop shrinking. §4's totals are the numbers to compare methods on; these plots are a picture of the mechanism.
- **Coverage gaps.** The preceding "Basics Review" section is lifted wholesale from another of the instructor's courses (the narration opens with the wrong course title) and is mostly a Keras refresher unrelated to ranking; only its confidence-interval and Gaussian-conjugate-prior lectures are prerequisites for any of the above. One lecture in the ranking section ships without captions, so it is not covered here at all.

## Trade-offs

- **UCB vs Thompson.** Deterministic, simple to reason about vs random, better with delayed feedback, usually more reward.
- **Bandit vs A/B test.** Less regret, adapts during the test vs clean, fixed-sample statistics.
- **Fixed vs drifting rates.** Simple counts vs need discounting or sliding windows to forget stale data.

## Common Pitfalls

- **Treating early winners as final.** Without exploration a lucky ad locks in.
- **Ignoring drift.** Click rates change; old counts make the bandit slow to notice.
- **Single, unseeded runs.** Bandit results vary run to run; compare averages over seeds.
- **Calling bandits full RL.** There's no state or long-term planning; actions don't change the future.
- **Unbounded rewards with Beta posteriors.** Thompson's Beta–Bernoulli form is for 0/1 rewards; use other priors otherwise.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| A few ads, instant clicks | Thompson sampling | Long A/B test | Less regret |
| Feedback arrives in batches | Thompson | UCB | Random draws tolerate delay |
| Need deterministic, auditable choices | UCB | Thompson | Same data → same choice |
| Best ad depends on the user | Contextual bandit (LinUCB, ...) | Plain bandit | Personalisation |
| Choices affect future states | Q-learning / RL | Bandit | State matters |

## References

- [Auer, Cesa-Bianchi & Fischer — Finite-time Analysis of the Multiarmed Bandit Problem (2002)](https://link.springer.com/article/10.1023/A:1013689704352)
- [Russo et al. — A Tutorial on Thompson Sampling (2018)](https://arxiv.org/abs/1707.02038)
- [Chapelle & Li — An Empirical Evaluation of Thompson Sampling (2011)](https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)
- [Sutton & Barto, 2nd ed., ch. 2](http://incompleteideas.net/book/the-book-2nd.html)

## See Also

- [rl-foundations-mc-td](rl-foundations-mc-td.md) §4 — ε-greedy on a bandit.
- [q-learning](q-learning.md) — when actions change the state.
- [statistics-and-probability](../math-for-ml/statistics-and-probability.md) — Beta distributions and Bayesian updating.
