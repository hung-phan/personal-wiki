---
name: statistics-and-probability
description: Statistics and probability foundations for ML — populations vs samples, descriptive statistics (mean, median, variance, covariance, correlation), probability axioms, conditional probability, Bayes' theorem, common distributions (Normal, Bernoulli, Binomial, Poisson, Uniform, Exponential), Central Limit Theorem, Law of Large Numbers, Maximum Likelihood Estimation (MLE), MAP, hypothesis testing, p-values, confidence intervals, bias-variance tradeoff, linear and logistic regression, regularization as a Bayesian prior, entropy, KL divergence, cross-entropy.
created: 2026-06-08
updated: 2026-07-03
tags: [statistics, probability, machine-learning, math, foundations, bayes, distributions, regression, mle, learning-journey]
---

# Statistics & Probability for Machine Learning

> **How to read this note.** Each section starts with a *problem you'd actually face*, then walks toward the math that solves it. The formulas come *after* the intuition. If you find yourself reading a formula and not knowing where it came from, scroll up — the prior paragraph is where the formula was born.

## Why This Exists
**Problem.** You build a model. It claims an image is a cat with 94% confidence. What does that 94% *mean*? Could you bet money on it? You evaluate two models — one is 91% accurate, the other 92%. Is the second actually better, or did it just get lucky on this test set? You see a feature that *looks* correlated with the target. Is that signal you can use, or coincidence?

These are not engineering questions. They are statistical questions. ML without statistics is a black box that occasionally gives you numbers — and you have no language to argue with it.

**Key insight.** Machine learning *is* applied statistics. A loss function is a likelihood. Regularization is a prior. Cross-validation is a hypothesis test. Once you see this, ML stops being a bag of tricks and becomes one coherent thing.

**Reach for this when:**
- Picking a loss function (MSE? cross-entropy? Huber?) — the answer comes from "what's the noise distribution?"
- Interpreting a model's confidence — calibration, predictive intervals.
- Reading a paper that says "MLE", "posterior", "log-likelihood", "KL divergence".
- Debugging a model that overfits — bias-variance, regularization.
- Designing an A/B test or evaluating an experiment.

**Don't reach for this when:**
- You need matrix mechanics — go to [linear-algebra](linear-algebra.md).
- You need optimizer mechanics — go to [calculus](calculus.md).

## Diagrams

```mermaid
flowchart TD
    A[Population<br/>what we want to know about] -->|sample| B[Sample<br/>what we can actually measure]
    B --> C[Descriptive Stats:<br/>summarize the sample]
    B --> D[Inferential Stats:<br/>say something about the population]
    D --> E[MLE / MAP<br/>fit a model]
    D --> F[Hypothesis Tests<br/>is this difference real?]
    G[Probability:<br/>theory of uncertainty] --> H[Distributions]
    G --> J[Bayes' Theorem]
    J --> E
    E --> K[ML Models]
```

## 1. Why we need statistics at all: populations vs. samples

**The problem.** You want to know the average height of every adult on Earth. You can't measure all of them. So you measure a few thousand people and *hope* their average tells you something about everyone.

That gap — between "what we measured" (a sample) and "what we want to know about" (a population) — is the central problem of statistics. Every guarantee, error bar, and confidence interval is some statement about how far the sample can be from the truth.

**Reach for this:**
- The *population* is the full set of things you'd want a claim about. Often hypothetical or impossible to access.
- A *sample* is what you actually measured. ML training data is *always* a sample.
- A *sampling bias* is when the sample doesn't represent the population. This is the silent killer of ML projects: a face recognition model trained mostly on one demographic will fail on others, not because the math is wrong but because the sample wasn't the population it'll be deployed against.
- The bedrock assumption of supervised learning is that samples are **i.i.d.** — independent and identically distributed. In English: every example is drawn the same way, and one doesn't influence the next. Time series violate this. Data leakage between train and test violates this. Covariate shift (deployment data drifts away from training data) violates this.

If you forget every other word in this section, remember: **the model only knows the sample. You're trusting the sample to look like the world.**

## 2. Describing a sample: the bare minimum numbers

### Where is the data centered?

**The problem.** Two classes both averaged 70 on an exam. In one class, everyone got 70. In the other, half got 100 and half got 40. The "average" hides this. You need a number that says *roughly where* the data sits.

That's **central tendency**. Three options, each useful for different reasons:

- **Mean** — add up all the values, divide by count. Simple. But sensitive to outliers: one billionaire in a sample of 100 people obliterates the meaning of "average wealth." (An **outlier** is a data point far from the bulk of the distribution — extreme enough that summary statistics distort around it.)
  $$\mu = \frac{1}{n}\sum_{i=1}^n x_i$$

- **Median** — sort the data, take the middle value. Robust to outliers. The billionaire is still in the sample, but they don't drag the median up to a meaningless number.

- **Mode** — most frequent value. The only one that makes sense for categorical data ("most common eye color in the dataset").

**ML takeaway.** When you're imputing missing values, you're picking one of these. Numeric + symmetric data → mean. Numeric + skewed data → median. Categorical → mode. The right choice prevents your imputation from systematically distorting the distribution.

Watch mean vs. median react to a single outlier — one billionaire tacked onto nine ordinary incomes. (`ddof` (numpy's "delta degrees of freedom") switches between dividing by $n$ (`ddof=0`, population) or $n-1$ (`ddof=1`, sample).)

```python
import numpy as np
incomes = np.array([30, 35, 40, 42, 45, 48, 50, 55, 60, 1_000_000])  # last value = outlier
print(f"mean   = {np.mean(incomes):,.1f}")   # ~100,040 — dragged by outlier
print(f"median = {np.median(incomes):,.1f}") # 46.5 — unmoved
print(f"pop var  (ddof=0) = {np.var(incomes, ddof=0):,.1f}")     # divides by n (MLE)
print(f"sample var (ddof=1) = {np.var(incomes, ddof=1):,.1f}")   # divides by n-1 (Bessel, unbiased)
print(f"std = {np.std(incomes):,.1f}")                            # ddof=0 by default
# Expected output:
# mean   = 100,040.5
# median = 46.5
# pop var  (ddof=0) = 89,991,900,254.1
# sample var (ddof=1) = 99,991,000,282.3
# std = 299,986.5
```

Takeaway: the mean shifts by orders of magnitude; the median barely moves. Sample variance (`ddof=1`) is always larger than population variance (`ddof=0`) by exactly the factor $n/(n-1)$.

### How spread out is the data?

**The problem.** "Average is 70" doesn't tell you whether it's "everyone got 70" or "half got 100, half got 40." We need a *spread* number.

**First attempt.** Average of all the deviations from the mean: $\frac{1}{n}\sum(x_i - \mu)$. Sounds reasonable. But the positives cancel the negatives — by definition, this is always zero. Useless.

**Second attempt.** Average of *absolute* deviations: $\frac{1}{n}\sum|x_i - \mu|$. Works, and it's even called the *mean absolute deviation*. Why don't we use it more?

**Third attempt.** Average of *squared* deviations:
$$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2$$

This is **variance** (specifically the *population* variance, dividing by $n$). When estimating variance *from a sample*, you divide by $n-1$ instead — this is **Bessel's correction**, and it removes a systematic underestimate that comes from using the sample mean in place of the true mean. In numpy this is the difference between `np.var(x, ddof=0)` (population, the default) and `np.var(x, ddof=1)` (sample).

Two reasons we usually pick squared over absolute deviations:
1. Squaring is *differentiable everywhere*. Absolute value isn't (kink at zero). When ML wants to minimize spread via gradient descent, smooth functions cooperate; non-smooth functions don't. (See [calculus](calculus.md) for why this matters.)
2. Squaring penalizes big deviations *much more* than small ones. A deviation of 10 contributes 100 to variance; a deviation of 1 contributes 1. If you want a number that's *very upset* about outliers, variance is it.

The downside: variance is in *squared units* (squared dollars, squared centimeters), which doesn't mean anything to humans. So we usually report **standard deviation** $\sigma = \sqrt{\sigma^2}$ instead — same units as the data, easy to interpret.

To *see* what squaring does — and why it matters — compare the two side by side on the same eight points, one of which is an outlier:

![|deviation| vs. (deviation)² — outlier bar grows from 4.5× to 20× when squared|697](assets/variance-squared-vs-abs.svg)

**ML takeaway.** In regression, a **residual** is the gap between a data point and the model's prediction for it: $r_i = y_i - \hat y_i$. Positive means the model under-predicted; negative means it over-predicted. The reason MSE (mean squared error) is the default regression loss is exactly this: it's *variance of the residuals*. You're minimizing the spread of your prediction errors, weighting big errors much more than small ones (because squaring a big residual makes it *much* bigger). If you don't want that — if your data has heavy tails and big errors are *expected* — use mean absolute error or Huber loss instead.

### How are two variables related?

**The problem.** "When height goes up, weight tends to go up too." Can we put a number on "tends to"?

**Idea.** When $x$ is above its mean, is $y$ also above its mean? Multiply the two deviations and average them:

$$\text{Cov}(X, Y) = \frac{1}{n}\sum_{i=1}^n (x_i - \mu_X)(y_i - \mu_Y)$$

This is **covariance**:
- Positive ⇒ they tend to move together (above-average $x$ pairs with above-average $y$).
- Negative ⇒ they tend to move opposite.
- Zero ⇒ no linear relationship (could still be a nonlinear one!).

**Problem with covariance.** Its magnitude depends on the units. Covariance of (height in cm, weight in kg) is a different number from (height in mm, weight in kg) — but the relationship is the same. Useless for comparing across pairs.

**Fix.** Divide by both standard deviations to make it unitless:

$$\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

This is **Pearson correlation**, always between $-1$ and $+1$:
- $+1$ — perfect linear relationship, going up.
- $-1$ — perfect, going down.
- $0$ — no linear relationship.

Verify on a small dataset with a clear positive relationship (`y ≈ 2x + noise`):

```python
import numpy as np
rng = np.random.default_rng(seed=0)                # seeded RNG — reproducible draws
x = np.linspace(0, 10, 50)                         # 50 evenly spaced points on [0, 10]
y = 2 * x + rng.normal(0, 1.5, size=50)            # 50 draws from N(0, 1.5²) added as noise

cov_matrix = np.cov(x, y, ddof=1)     # 2x2 sample covariance matrix (Bessel corrected)
corr_matrix = np.corrcoef(x, y)       # 2x2 Pearson correlation matrix, entries in [-1, 1]
print(f"Cov(x, y) = {cov_matrix[0, 1]:.2f}")
print(f"Corr(x, y) = {corr_matrix[0, 1]:.3f}")

# Correlation by hand from the covariance and stds
by_hand = cov_matrix[0, 1] / (np.std(x, ddof=1) * np.std(y, ddof=1))  # ddof=1 matches np.cov above
assert np.isclose(by_hand, corr_matrix[0, 1])
print(f"By hand match: {by_hand:.3f}")
# Expected output:
# Cov(x, y) = 19.24
# Corr(x, y) = 0.981
# By hand match: 0.981
```

Correlation lands near 0.98 — strong linear signal even with noise. The by-hand check shows correlation is *literally* covariance normalized by both standard deviations.

**ML takeaways.**
- The covariance matrix of your features (every pair's covariance, in a matrix) is the central object that PCA decomposes — see [linear-algebra](linear-algebra.md).
- Highly correlated features cause **multicollinearity** in linear regression — you can't tell which feature is doing the work, and the coefficients become unstable.
- **Correlation is not causation.** Ice cream sales correlate with drowning deaths (both rise in summer). A model that learned "more ice cream → more drownings" would be statistically right and substantively wrong. Always ask: is there a hidden third variable?

```mermaid
flowchart LR
    S[Summer heat] --> I[Ice cream sales]
    S --> D[Drowning deaths]
    I -.spurious.-> D
```

The dashed edge is what a naive correlation sees; the solid edges are what's actually happening. Always ask: what's the arrow into *both*?

## 3. From describing a sample to reasoning about uncertainty

The previous section was *descriptive*: given numbers, summarize them. Now we want *inferential*: given numbers, say something about what *would* happen.

To do that we need probability — a calculus of uncertainty.

### What is probability, exactly?

**The problem.** You flip a coin. Before it lands, you say "50% heads." What does that 50% mean? You can't observe a 50% — you observe heads or tails.

Two answers, both useful:
- **Frequentist.** "If you flipped this coin a million times, about 500,000 would be heads." Probability = long-run frequency.
- **Bayesian.** "I'm equally willing to bet on either side." Probability = degree of belief.

Most ML lives implicitly in a frequentist world (data is the source of truth) with occasional Bayesian flavor (priors, regularization).

### The minimal rules

A probability $P(A)$ is just a number satisfying:
1. $P(A) \geq 0$ — never negative.
2. $P(\text{anything happens}) = 1$ — total probability sums to 1.
3. If $A$ and $B$ can't both happen (disjoint), then $P(A \text{ or } B) = P(A) + P(B)$.

That's it. Everything else (independence, conditional probability, the entire field) is built on these three rules.

### Random variables and their "shapes"

**The problem.** "Probability of heads = 0.5" is fine for a coin. But what about height? Height is a continuous range — what's the probability someone is *exactly* 175.000... cm? Zero, because there are infinitely many possible heights.

So we don't talk about the probability of a *single value* for continuous things. We talk about the probability of a *range*.

A **random variable** is a function that turns outcomes into numbers (height of a randomly picked person, or 0/1 for a coin flip). The way its probabilities are distributed is described by:

- **PMF (Probability Mass Function)** for discrete variables: $P(X = x)$ — direct probability of each value. Roll of a fair die: $P(X = 1) = P(X = 2) = \dots = P(X = 6) = 1/6$.
- **PDF (Probability Density Function)** for continuous variables: $f(x)$ where $P(a \leq X \leq b) = \int_a^b f(x)\,dx$.
- **CDF (Cumulative Distribution Function)** for both: $F(x) = P(X \leq x)$.

**Crucial trap.** A PDF value is *not* a probability. It can be greater than 1. $f(x) = 5$ doesn't mean a 500% chance — it means probability per unit width. You only get a probability by integrating over a range.

### Joint, marginal, conditional — and why they matter

Given two random variables $X$ (e.g., having a disease) and $Y$ (e.g., test result):

- **Joint** $P(X, Y)$ — both at once. "I have the disease *and* the test is positive."
- **Marginal** $P(X)$ — ignore $Y$. "What fraction of people have the disease?" Get it by summing/integrating over the other variable.
- **Conditional** $P(X | Y)$ — given $Y$. "Given the test is positive, what's the chance I have the disease?"

These three quantities are linked by definition:
$$P(X | Y) = \frac{P(X, Y)}{P(Y)}$$

In English: "$X$ given $Y$" = "both" divided by "$Y$ alone." This obvious-looking formula is the key to Bayes' theorem next.

**Independence.** $X$ and $Y$ are independent ($X \perp Y$) if knowing one tells you nothing about the other. Equivalently: $P(X | Y) = P(X)$, or $P(X, Y) = P(X)P(Y)$.

**Expectation.** The "average value" of a random variable, weighted by its probabilities:
$$E[X] = \sum_x x \cdot P(X = x) \quad \text{or} \quad \int x f(x)\,dx$$

A wonderful, often-underused property: **linearity of expectation**.
$$E[aX + bY] = aE[X] + bE[Y]$$

This holds *even when $X$ and $Y$ are not independent*. It's a freebie that makes a lot of ML math much shorter than you'd expect.

## 4. Bayes' theorem: the formula you'll use forever

### The problem

You take a medical test that's "99% accurate." It comes back positive. Should you panic?

Most people say yes. The math says: probably not.

Here's why. Suppose:
- The disease is rare: 1% of people have it.
- The test is 99% sensitive (true positive rate among sick people).
- The test is 99% specific (true negative rate among healthy people).

Imagine 10,000 people. Roughly:
- 100 have the disease. Of those, 99 test positive (true positives), 1 negative.
- 9,900 are healthy. Of those, 99 test positive (false positives), 9,801 negative.

Total positives: $99 + 99 = 198$. Of those, only $99$ actually have the disease.

**Probability of disease given positive test = 99/198 ≈ 50%.**

A "99% accurate" test, when applied to a rare disease, leaves you with a coin flip. This isn't a math trick — it's how reality works, and people get it wrong constantly. (It has a name: the **base rate fallacy**.)

Compute it directly with Bayes' rule:

```python
p_disease = 0.01         # base rate (prior)
sensitivity = 0.99        # P(positive | disease)
specificity = 0.99        # P(negative | no disease)

# P(positive) via total probability — sum over both disease outcomes
p_pos = sensitivity * p_disease + (1 - specificity) * (1 - p_disease)
# Bayes' rule: posterior = (likelihood × prior) / evidence
p_disease_given_pos = (sensitivity * p_disease) / p_pos
print(f"P(disease | positive) = {p_disease_given_pos:.3f}")  # ~0.500
# Expected output:
# P(disease | positive) = 0.500
```

The result: even with a 99%-accurate test, a positive result on a 1%-prevalence disease implies only ~50% chance of actually being sick. The prior dominates.

Here it is with every one of the 10,000 people drawn as a dot:

![10,000-person grid: tiny red patch of 100 diseased, sea of 9,900 healthy, and ~99 orange false-positive dots that swamp the true positives|697](assets/bayes-base-rate-grid.svg)

### What we just did, in a formula

We started with $P(\text{positive} | \text{disease})$ — probability of evidence given hypothesis. We wanted $P(\text{disease} | \text{positive})$ — the other direction. The bridge between them is **Bayes' theorem**:

$$P(H | D) = \frac{P(D | H) \cdot P(H)}{P(D)}$$

- $P(H | D)$ — **posterior**: probability of hypothesis given data.
- $P(D | H)$ — **likelihood**: probability of data given hypothesis.
- $P(H)$ — **prior**: probability of hypothesis before seeing data.
- $P(D)$ — **evidence** / total probability of data: just a normalizing constant.

In words: *"posterior is proportional to likelihood times prior."* The ML way to read it: **the model updates its belief by combining what it expected (prior) with what it saw (likelihood)**.

### Why this is the most important formula in ML

- **Naive Bayes classifier** literally is Bayes' theorem applied with a "features are independent" assumption.
- **Bayesian inference** is updating model parameters using Bayes' theorem on observed data.
- **MAP estimation** (next section) finds the best parameter under a prior — that's MLE plus a prior, which is just Bayes' rule again.
- **Regularization** can be derived as a prior (see Section 7).
- **Why your spam filter works**, why **medical diagnosis** is a probabilistic task, why **A/B testing** can be done Bayesianly.

If you remember one formula from this whole note, make it Bayes'.

### Watch belief actually shift: prior × likelihood → posterior

Concrete example: you have a coin and want to know $p = P(\text{heads})$. You start with a gentle prior — Beta(2, 2) — that says "probably fair, but I'm not confident." You flip 10 times and see 8 heads. The likelihood peaks at $p = 0.8$. Multiply, renormalize, and you get the posterior — Beta(10, 4), peaking around $p = 0.75$:

![Beta(2,2) prior centered at 0.5, dashed likelihood peaked at 0.8, red posterior Beta(10,4) peaked at 0.75 — belief pulled toward the data|697](assets/bayes-prior-to-posterior.svg)

The posterior lands *between* the prior and the likelihood — a data-weighted compromise. With more flips the likelihood sharpens and the prior loses influence; with fewer flips the prior dominates. That is Bayes updating in one picture.

The Beta distribution is *conjugate* to the Bernoulli — after $h$ heads and $t$ tails, a Beta($\alpha, \beta$) prior becomes a Beta($\alpha + h, \beta + t$) posterior. No integration needed:

```python
from scipy import stats
# Prior: Beta(2, 2) — mild belief the coin is fair
alpha_prior, beta_prior = 2, 2
heads, tails = 8, 2

# Posterior: Beta(alpha + heads, beta + tails)  — conjugate update, no integral needed
alpha_post, beta_post = alpha_prior + heads, beta_prior + tails

prior_mean = alpha_prior / (alpha_prior + beta_prior)     # mean of Beta(a,b) = a/(a+b)
post_mean = alpha_post / (alpha_post + beta_post)
print(f"Prior:     Beta({alpha_prior}, {beta_prior}) mean = {prior_mean:.3f}")
print(f"Posterior: Beta({alpha_post}, {beta_post}) mean = {post_mean:.3f}")

# scipy agrees — stats.beta(a, b) is a frozen Beta distribution object
assert np.isclose(stats.beta(alpha_post, beta_post).mean(), post_mean)
# Expected output:
# Prior:     Beta(2, 2) mean = 0.500
# Posterior: Beta(10, 4) mean = 0.714
```

Prior mean is 0.500 (fair coin); posterior mean shifts to ~0.714 — pulled toward the data (8/10 = 0.8) but tempered by the prior.

## 5. Common distributions: the "shapes" data tends to take

### The problem

Real-world data isn't an arbitrary jumble of numbers. It tends to follow a small number of recognizable shapes — bell curves, exponentials, counts. Knowing these shapes lets you:
- Pick the right model assumption (Gaussian noise? Bernoulli outcome?).
- Generate synthetic data.
- Reason about what's "extreme" (a 7-foot person is extreme; a $7$-billion-dollar net worth is extreme; the math for "extreme" depends on the shape).

So we name and parameterize the most common shapes.

### The most important continuous shape: the Normal (Gaussian)

You measure height. You measure noise on a sensor. You sum a bunch of small independent things. They all look bell-shaped — symmetric, peaking in the middle, tapering off in both directions. That's the Normal distribution:

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

The formula looks ugly, but the *picture* is the bell curve. Two parameters: $\mu$ (where it's centered) and $\sigma$ (how wide).

The 68-95-99.7 rule is worth memorizing: that fraction of values fall within 1, 2, 3 standard deviations of the mean.

**Why does Normal show up everywhere?** Because of the **Central Limit Theorem** (Section 6). It's not magic; it's a deep theorem about averaging.

### The most important discrete shape: Bernoulli / Binomial

A **Bernoulli** trial: one yes/no outcome with probability $p$. The flip of a coin, the click on an ad, the pass/fail of a single test. One parameter, $p$.

A **Binomial** is the count of yeses across $n$ Bernoulli trials. "Out of 100 emails, how many were spam?" Two parameters: $n$ and $p$.

Draw large samples and confirm the sample means match the theoretical ones. Theoretical means: Normal is $\mu$, Bernoulli is $p$, Poisson is $\lambda$:

```python
import numpy as np
rng = np.random.default_rng(seed=0)                       # seeded RNG — reproducible
N = 100_000

normal_samples = rng.normal(loc=5.0, scale=2.0, size=N)   # N draws from N(5, 2²)
bernoulli_samples = rng.binomial(n=1, p=0.3, size=N)      # n=1 → Bernoulli(p=0.3)
poisson_samples = rng.poisson(lam=4.0, size=N)            # N draws from Poisson(λ=4)

print(f"Normal:    theoretical μ=5.0,  sample mean={normal_samples.mean():.3f}")
print(f"Bernoulli: theoretical p=0.3,  sample mean={bernoulli_samples.mean():.3f}")
print(f"Poisson:   theoretical λ=4.0,  sample mean={poisson_samples.mean():.3f}")
# Poisson variance also equals λ — check:
print(f"Poisson variance ≈ λ: {poisson_samples.var():.3f}")   # .var() defaults to ddof=0
# Expected output:
# Normal:    theoretical μ=5.0,  sample mean=4.998
# Bernoulli: theoretical p=0.3,  sample mean=0.300
# Poisson:   theoretical λ=4.0,  sample mean=4.006
# Poisson variance ≈ λ: 4.035
```

All three sample means land within ~0.01 of the theoretical values. For Poisson, mean *equals* variance — a defining property.

### Quick reference — distributions you'll see in ML

A **latent variable** is one the model posits but doesn't observe directly (like the cluster assignment in a mixture model or the topic of a document in LDA).

| Distribution            | Shape / Models                         | When ML uses it                                    |
| ----------------------- | -------------------------------------- | -------------------------------------------------- |
| **Bernoulli**           | One yes/no                             | Binary classification target (cat / not-cat)       |
| **Binomial**            | Count of yeses in $n$ trials           | CTR models, A/B test analysis                      |
| **Categorical**         | One-of-K outcome                       | Multiclass labels (cat / dog / bird)               |
| **Multinomial**         | Counts across K outcomes in $n$ trials | Word counts, bag-of-words                          |
| **Uniform**             | All values equally likely              | Random initialization, sampling                    |
| **Normal (Gaussian)**   | Symmetric bell                         | Regression noise, NN weight init, latent variables |
| **Multivariate Normal** | Bell curve in many dimensions          | Latent variables, GMMs                             |
| **Exponential**         | Time between events                    | Survival analysis                                  |
| **Poisson**             | Count of rare events in fixed interval | Rare-event count regression                        |
| **Beta**                | Probability over $[0,1]$               | Bayesian prior to Bernoulli's $p$                  |
| **Dirichlet**           | Probability vector summing to 1        | LDA topic priors                                   |
| **Gamma**               | Positive continuous                    | Conjugate priors                                   |
| **Laplace**             | Sharp peak, heavy tails                | Connects to L1 regularization                      |

You don't need to memorize every formula. You need to know which shape to *reach for* when describing data. "This is bounded between 0 and 1 — Beta. This is a count — Poisson. This is a sum of small effects — Normal."

## 6. Why averages tend to look Normal: the Central Limit Theorem

### The problem

You flip a coin 1,000 times and count heads. You'd expect about 500. But what's the *distribution* of "number of heads across many such experiments"? Surprise: it's bell-shaped, even though each individual flip is binary.

Roll a die 1,000 times and average the results. The average distribution? Bell-shaped, even though each roll is uniform over 1–6.

Take *any* distribution — uniform, exponential, weird and lumpy — sample $n$ values from it, take the mean. As $n$ grows, the distribution of those means becomes Normal.

This is the **Central Limit Theorem**:

$$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

where $\bar X_n = \tfrac{1}{n}\sum_{i=1}^n X_i$ is the sample mean and $\xrightarrow{d}$ means *converges in distribution* — the histogram of $\bar X_n$ across many samples approaches the Normal shape as $n$ grows.

**In plain English.** The sample mean of i.i.d. data approaches a Normal distribution as you collect more samples, no matter what shape the original data had.

Verify the $\sigma/\sqrt{n}$ shrinkage of the sample-mean's standard deviation. For Uniform(0, 1), the population $\sigma = 1/\sqrt{12} \approx 0.2887$:

```python
import numpy as np
rng = np.random.default_rng(seed=0)
sigma_pop = 1 / np.sqrt(12)  # true σ of Uniform(0, 1)
trials = 10_000

for n in [1, 5, 30]:
    # (trials × n) matrix of U(0,1) draws; .mean(axis=1) collapses each row to its sample mean
    sample_means = rng.uniform(0, 1, size=(trials, n)).mean(axis=1)
    theoretical_std = sigma_pop / np.sqrt(n)               # CLT prediction for SE of the mean
    print(f"n={n:2d}: mean={sample_means.mean():.3f}, "
          f"std={sample_means.std():.4f}, "
          f"σ/√n={theoretical_std:.4f}")
# Expected output:
# n= 1: mean=0.499, std=0.2891, σ/√n=0.2887
# n= 5: mean=0.501, std=0.1287, σ/√n=0.1291
# n=30: mean=0.500, std=0.0529, σ/√n=0.0527
```

The mean stays near 0.5 (the Law of Large Numbers — see below); the std shrinks like $\sigma/\sqrt{n}$ — halving requires quadrupling the sample. That's why "double the data" only cuts your error bars by ~30%, not 50%.

The 3×3 grid below is the payoff — pick any source distribution (row), take averages of size $n$ (column), and watch the shape morph:

![3×3 grid — Uniform / Exponential / Bimodal rows; n=1, n=5, n=30 columns; leftmost column keeps the source shape, rightmost column collapses to a bell for all three|697](assets/clt-convergence.svg)

### Why this is the headline theorem of statistics

- **It explains why Normal is everywhere.** Anything that's an average or a sum of many small effects ends up Normal.
- **It's why "error bars" use $\pm 2\sigma/\sqrt{n}$.** Once you know the means are approximately Normal, the 95% confidence interval falls out for free.
- **It justifies the Gaussian noise assumption** in linear regression: if measurement error is the sum of many tiny independent effects, it's approximately Normal.
- **The Law of Large Numbers** is the simpler companion: as $n \to \infty$, $\bar{X}_n \to \mu$. This is why averaging over a mini-batch gives a useful gradient estimate in **SGD** (stochastic gradient descent) — trains on small random *mini-batches* of data rather than the full set; more in [calculus](calculus.md). Many noisy estimates average out to the truth.

## 7. From data to model parameters: Maximum Likelihood Estimation (MLE)

### The problem

Someone hands you a coin and flips it 20 times: **13 heads, 7 tails**. Is it fair? Biased? What's your best guess for $p$, the probability of heads?

You have data. You have a model (Bernoulli with unknown $p$). You want the "best" value of $p$. What does "best" even mean?

### The intuition — score each guess by how well it explains what happened

Try $p = 0.35$. Under that model, seeing 13 heads out of 20 would be a freakish accident. Try $p = 0.65$. Now those 13 heads are exactly what you'd expect. Try $p = 0.9$. Too extreme — you'd have expected more heads than you saw.

The **plausibility** of a given $p$ = the probability the model assigns to *the data you actually observed*. That number has a name: the **likelihood**.

$$L(p) = P(\text{data} \mid p) = p^{13} (1-p)^{7}$$

(13 heads each contribute a factor of $p$, 7 tails each contribute $1-p$; multiply.)

Note the flip in what's variable: the *data* is fixed (that's what you observed), and $p$ is the free parameter. $L(p)$ is a function of $p$. Plot it:

![Likelihood curve L(p) = p^13 (1-p)^7 peaks sharply at p = 0.65](assets/mle-likelihood-curve.svg)

**The MLE recipe in one sentence:** pick the value that makes the observed data most likely — the peak of $L$.

$$\hat p_{\text{MLE}} = \arg\max_p L(p)$$

($\arg\max_p L(p)$ means *the value of $p$ that makes $L(p)$ largest* — as opposed to $\max L$, which would be the largest value itself.)

For this example, the peak is at $p = 0.65 = 13/20$. That's why we teach kids "13 out of 20 = 0.65" — it's literally MLE for a Bernoulli. Every empirical frequency you've ever computed was an MLE.

### The three moves that solve every MLE problem

Real MLE almost never has just one data point — it has hundreds or thousands. And "argmax the joint probability of *all* of them" isn't something you can plug and chug in that form. There's a standard three-move dance that turns any MLE problem into something solvable.

**Move 1 — factorize (use i.i.d.).**

*Problem.* We want $P(D \mid \theta)$ — the probability of the *whole dataset* under parameter $\theta$. But there's no direct formula for "probability of 1000 numbers all at once." What we *do* have is a formula for one point: $P(x_i \mid \theta)$.

*The move.* If the data points are independent (the i.i.d. assumption from §1), the joint probability equals the product of per-point probabilities:

$$L(\theta) = P(D \mid \theta) = \prod_{i=1}^n P(x_i \mid \theta)$$

(We'll write $\theta$ for the model's parameter(s) — could be $p$, could be $(\mu, \sigma^2)$, could be a whole weight vector, depending on the model.)

*Payoff.* One intractable multi-dimensional object gets swapped for $n$ small, tractable ones. Every distribution's PDF/PMF is a one-liner; multiplying $n$ of them is arithmetic.

**Move 2 — take the log.**

*Problem.* We're now multiplying $n$ numbers, each in $[0, 1]$. Two things break:
1. **Numerics.** A product of a thousand small probabilities underflows to `0.0` in floating point. Every candidate $\theta$ gets scored as zero — you can't compare them.
2. **Algebra.** The derivative of a product of $n$ things means iterated product rule — brutal even for small $n$, and it doesn't simplify.

*The move.* Take the logarithm:

$$\log L(\theta) = \sum_{i=1}^n \log P(x_i \mid \theta)$$

*Payoff.* Three wins from one operation:
- **Product → sum.** The log's defining identity is $\log(ab) = \log a + \log b$; the product just falls apart.
- **No underflow.** `log(1e-300)` is just $-690$, an ordinary float. Sums of these stay well-scaled.
- **Argmax doesn't move.** Since $\log$ is monotonically increasing, $\arg\max_\theta L(\theta) = \arg\max_\theta \log L(\theta)$. We've reshaped the surface (multiplication → addition, tiny numbers → large negatives) but the *location of the peak is unchanged*. That's the entire reason we're allowed to do this.

**Move 3 — set derivative to zero.**

*Problem.* We want the peak of $\log L(\theta)$. How do you find the peak of a smooth function?

*What the notation $\frac{d \log L}{d\theta}$ means.* Read it aloud: *"the derivative of $\log L$ with respect to $\theta$."* It's a function that takes any value of $\theta$ and returns **the slope of the log-likelihood curve at that point** — a number that tells you "if I nudge $\theta$ a tiny bit up, does $\log L$ go up (positive slope) or down (negative slope), and by how much per unit of $\theta$?" It's just the ordinary derivative from calculus, applied to $\log L$ viewed as a function of $\theta$. When $\theta$ has multiple components (like $(\mu, \sigma^2)$), we write $\frac{\partial \log L}{\partial \mu}$ instead — the *partial* derivative, meaning "slope in the $\mu$ direction, holding everything else fixed." (Both are covered in [calculus](calculus.md).)

*Why the derivative = 0 trick works.* Not convention — it's a geometric fact about smooth curves. Think about walking along the graph of a smooth function. Going *uphill* means the tangent slopes up (derivative positive). Going *downhill* means the tangent slopes down (derivative negative). At the exact instant you're crossing over from uphill to downhill — the peak — the tangent is momentarily flat: derivative = zero. Same for a valley (crossing downhill → uphill). So *every extremum of a smooth function has zero derivative*, which means the peak we're looking for must live among the solutions of $\frac{d \log L}{d\theta} = 0$. (Caveats: works only for smooth functions, and it also flags valleys and saddles. In MLE those rarely bite — the log-likelihood is usually concave, so the only critical point *is* the peak.)

*The move.* Compute $\frac{d \log L}{d\theta}$, set it to zero, and solve for $\theta$. That $\theta$ is $\hat\theta_{\text{MLE}}$. (The hat marks it as an *estimate* computed from data, distinct from the unknown true $\theta$.)

*Payoff.* We've converted an optimization problem ("find where the max is") into an algebra problem ("solve one equation"). For a lot of textbook likelihoods — Gaussian, Bernoulli, Poisson — this gives a **closed-form answer**: no computer needed, just calculus. When there's no closed form (e.g. neural networks), you fall back to gradient descent — but that's *still the same idea*, just done numerically instead of algebraically: keep stepping until the gradient is (near) zero.

**One more twist for the "loss function" framing.** By convention we flip the sign and *minimize* the **negative log-likelihood (NLL)** rather than maximize the log-likelihood. Same optimum — negating a function reflects it upside-down, so the maximum becomes the minimum at the same $\theta$. Why bother? Because "minimize a loss" is the standard vocabulary of gradient descent and ML. Same math, different framing.

That's the whole story: **NLL is the loss function of most probabilistic ML models.** Cross-entropy loss for a classifier is NLL under a categorical likelihood; MSE for a regressor is NLL under a Gaussian likelihood. More on that in the next subsection.

### Worked example 1 — coin flips (one parameter)

For our coin, $L(p) = p^{13}(1-p)^7$. Log it:

$$\log L(p) = 13 \log p + 7 \log(1 - p)$$

Differentiate and set to zero:

$$\frac{d \log L}{dp} = \frac{13}{p} - \frac{7}{1 - p} = 0 \;\;\Longrightarrow\;\; 13(1 - p) = 7 p \;\;\Longrightarrow\;\; \hat p = \frac{13}{20} = 0.65$$

The intuitive answer (heads-so-far over flips-so-far) falls out of the derivative — it wasn't a lucky guess. The general pattern: for $k$ heads out of $n$ flips, $\hat p_{\text{MLE}} = k/n$.

Verify in code:

```python
import numpy as np

n_heads, n_tails = 13, 7
p_grid = np.linspace(0.01, 0.99, 99)             # 99 candidate p values in (0, 1)
L = p_grid**n_heads * (1 - p_grid)**n_tails      # elementwise Bernoulli likelihood on the grid
p_hat = p_grid[np.argmax(L)]                     # argmax returns index of the peak

print(f"MLE by grid search:  p̂ = {p_hat:.3f}")
print(f"Closed-form (calc):  p̂ = {n_heads / (n_heads + n_tails):.3f}")
peak = L[np.argmax(L)]
at_half = L[np.argmin(np.abs(p_grid - 0.5))]     # nearest grid point to p = 0.5
print(f"L(0.65) / L(0.5)  = {peak / at_half:.2f}× more likely")
# Expected output:
# MLE by grid search:  p̂ = 0.650
# Closed-form (calc):  p̂ = 0.650
# L(0.65) / L(0.5)  = 2.49× more likely
```

The grid-search peak and the calculus-derived answer agree. And the peak is only ~2.5× more likely than $p = 0.5$ — with only 20 flips there's *huge* uncertainty about whether the coin is really biased. MLE gives a point estimate but no uncertainty; you'd need a full posterior (next subsection) or a confidence interval (§9) for that.

### Worked example 2 — Gaussian sample (two parameters)

**Setup.** You have $n$ measurements $x_1, \ldots, x_n$ — say the heights of $n$ adults in centimeters. You want to fit a Normal distribution $\mathcal{N}(\mu, \sigma^2)$ to them. Two unknowns this time: the mean $\mu$ (where the bell is centered) and the variance $\sigma^2$ (how wide it is). Same three moves as before — just with two parameters instead of one, so we'll differentiate twice.

**Move 1 — factorize.**

Each measurement is one draw from the Gaussian PDF, so the likelihood of all $n$ measurements together is a product:

$$L(\mu, \sigma^2) = \prod_{i=1}^n \underbrace{\frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)}_{\text{Gaussian PDF, evaluated at }x_i}$$

**Move 2 — log.**

Now unpack it step by step. Log of a product = sum of logs:

$$\log L = \sum_{i=1}^n \log\!\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)\right]$$

Two more log identities finish it: $\log(ab) = \log a + \log b$ splits each term into the prefactor plus the exponent, and $\log(\exp(z)) = z$ collapses the second half:

$$\log L = \sum_{i=1}^n \left[\; \log \tfrac{1}{\sqrt{2\pi\sigma^2}} \;-\; \frac{(x_i - \mu)^2}{2\sigma^2} \;\right]$$

The first term inside the brackets doesn't depend on $i$, so summing it $n$ times just multiplies by $n$. Pull the $\frac{1}{2\sigma^2}$ out of the sum. That leaves:

$$\log L = -\frac{n}{2}\log(2\pi\sigma^2) \;-\; \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2$$

Look at what this says: **log-likelihood = constant that penalizes wider bells − a "sum of squared residuals" divided by variance.** Big $\mu$-errors and big $\sigma$-choices both hurt it.

**Move 3a — solve for $\hat\mu$.**

Taking the derivative with respect to $\mu$ is easier than it looks: **only the second term depends on $\mu$**. The first term is entirely $\sigma^2$-and-constants — its $\mu$-derivative is zero. So we only differentiate the sum, using the chain rule on $(x_i - \mu)^2$:

$$\frac{\partial \log L}{\partial \mu} = -\frac{1}{2\sigma^2} \sum_{i=1}^n 2(x_i - \mu)\cdot(-1) = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu)$$

Set that to zero. Since $\sigma^2$ is positive, the fraction dies iff the sum is zero:

$$\sum_{i=1}^n (x_i - \mu) = 0 \;\;\Longrightarrow\;\; \sum_i x_i = n\mu \;\;\Longrightarrow\;\; \boxed{\;\hat\mu = \frac{1}{n}\sum_{i=1}^n x_i = \bar x\;}$$

**The MLE for the mean of a Gaussian is the sample average.** That's not a coincidence — it's a *consequence of assuming Gaussian noise*. "Just take the average" is the right answer *because* the noise is Gaussian. If the noise were Laplace-distributed instead (heavier tails), the same derivation gives the **median**, not the mean. If it were Poisson, something else entirely. The tool "take the mean" isn't neutral — it's a Gaussianity assumption in disguise.

**Move 3b — solve for $\hat\sigma^2$.**

Now differentiate with respect to $\sigma^2$. **This time both terms depend on $\sigma^2$**, so both contribute. Treat $\sigma^2$ as a single variable $v$ if that helps ($v = \sigma^2$); the algebra is cleaner. Then $\frac{d}{dv}\log v = \frac{1}{v}$, and $\frac{d}{dv}(1/v) = -1/v^2$.

$$\frac{\partial \log L}{\partial \sigma^2} = -\frac{n}{2}\cdot\frac{1}{\sigma^2} \;+\; \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \hat\mu)^2$$

(We've already fixed $\mu$ at its MLE $\hat\mu = \bar x$ from Move 3a — that's why $\hat\mu$ appears in the sum.) Set to zero:

$$\frac{n}{2\sigma^2} = \frac{1}{2\sigma^4}\sum_i (x_i - \hat\mu)^2 \;\;\Longrightarrow\;\; n\sigma^2 = \sum_i (x_i - \hat\mu)^2$$

$$\boxed{\;\hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar x)^2\;}$$

**The MLE for the variance is the sample's average squared deviation from its mean.**

**The $\frac{1}{n}$ vs $\frac{1}{n-1}$ subtlety.** MLE gives $\frac{1}{n}$ — the **population** variance formula (numpy's `np.var(x, ddof=0)`, the default). The $\frac{1}{n-1}$ version (`ddof=1`, **Bessel's correction**) is a *different* estimator that fixes a small downward bias in MLE's variance — MLE systematically underestimates $\sigma^2$ because it evaluates deviations from $\hat\mu$ (which is *fit* to the same data) rather than the true unknown $\mu$. MLE doesn't care about bias, only about maximizing likelihood. Both estimators converge to the truth as $n \to \infty$; the correction only matters at small $n$.

**Verify in code.**

```python
import numpy as np
rng = np.random.default_rng(seed=0)
true_mu, true_sigma = 5.0, 2.0
x = rng.normal(true_mu, true_sigma, size=1000)   # 1000 draws from N(5, 2²)

mu_hat = x.mean()
sigma2_hat_mle = x.var(ddof=0)          # MLE — divides by n (biased low)
sigma2_hat_unbiased = x.var(ddof=1)     # Bessel — divides by n-1 (unbiased)
print(f"MLE μ̂  = {mu_hat:.3f}  (true = {true_mu})")
print(f"MLE σ̂² = {sigma2_hat_mle:.3f}  (true = {true_sigma**2})")
print(f"Unbiased σ̂² = {sigma2_hat_unbiased:.3f}")
# Expected output:
# MLE μ̂  = 4.904  (true = 5.0)
# MLE σ̂² = 3.816  (true = 4.0)
# Unbiased σ̂² = 3.820
```

MLE variance (3.816) is *slightly* smaller than the unbiased version (3.820) — that's the bias Bessel corrects. With $n = 1000$ the gap is tiny. Both approach the truth $\sigma^2 = 4$ as $n \to \infty$.

**Zoom out.** Every MLE problem is these same three moves. Only the algebra of Move 3 changes with the distribution you chose. Once you can do it for the coin (one parameter, Bernoulli) and the Gaussian (two parameters, Normal) by hand, you can do it for anything — and once you know the answer is called "the loss function," the whole zoo of ML losses stops being a zoo.

### The connection that unlocks loss design

Different assumed noise distributions yield different MLE losses:

| Assumed noise distribution    | Loss derived from MLE               |
| ----------------------------- | ----------------------------------- |
| Gaussian on regression target | **Mean Squared Error (MSE)**        |
| Laplace on regression target  | **Mean Absolute Error (MAE)**       |
| Bernoulli on binary target    | **Binary cross-entropy** (log loss) |
| Categorical on multiclass     | **Categorical cross-entropy**       |
| Poisson on count target       | **Poisson NLL**                     |

**The big realization.** Picking a loss function = picking a noise distribution. If your residuals are heavy-tailed, MSE (Gaussian assumption) is wrong — it overweights outliers because the Gaussian thinks big residuals are very unlikely. Try Huber loss or MAE. (**heavy-tailed** = extreme values happen much more often than a Gaussian would predict — think the Laplace column in the table below, where a residual of 5 is ~1800× more likely than under Gaussian.)

#### Where does this table come from?

For regression, we model $y = f_\theta(x) + \epsilon$, where $f_\theta(x)$ is the model's prediction for input $x$ given parameters $\theta$. The noise $\epsilon$ is the random junk making predictions miss. We *choose* a distribution for $\epsilon$ — Gaussian is only the default, not a law of nature.

Once you pick that distribution, MLE mechanically produces the loss. The recipe is always the same four steps:

![MLE-to-loss recipe: assume noise → write PDF → take log → negate and sum|697](assets/loss-mle-recipe.svg)

Each row of the table above is that recipe applied to a different PDF:

![Gaussian → MSE, Laplace → MAE, Bernoulli → BCE|697](assets/loss-noise-instances.svg)

Walk through Gaussian → MSE explicitly. This is the four MLE moves you already saw, applied to a specific PDF.

**Setup.** For data point $i$, the model predicts $f_\theta(x_i)$ and the observation is $y_i$. Call the gap the **residual** $r_i = y_i - f_\theta(x_i)$. Assuming Gaussian noise $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ means the residual is Gaussian-distributed, so the probability the model assigns to seeing $y_i$ given $x_i$ is:

$$P(y_i \mid x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{r_i^2}{2\sigma^2}\right)$$

**Step 1 — Move 1 (factorize).** i.i.d. data means the full likelihood is a product across points:

$$L(\theta) = \prod_{i=1}^n P(y_i \mid x_i, \theta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{r_i^2}{2\sigma^2}\right)$$

**Step 2 — Move 2 (log).** Taking the log turns the product into a sum and cancels the `exp`:

$$\log L(\theta) = \sum_{i=1}^n \left[\; \log\!\underbrace{\tfrac{1}{\sqrt{2\pi\sigma^2}}}_{\text{constant in }\theta} \;-\; \frac{r_i^2}{2\sigma^2} \;\right]$$

**Step 3 — Drop the constant.** The first term doesn't depend on $\theta$, so it's a constant offset. It can't change the *location* of the argmax — only shift the whole surface up or down. Drop it.

$$\log L(\theta) = -\frac{1}{2\sigma^2}\sum_{i=1}^n r_i^2 \;+\; \text{const}$$

**Step 4 — Negate (Move 4).** By convention we minimize NLL rather than maximize log-likelihood. Flip the sign:

$$\text{NLL}(\theta) = \frac{1}{2\sigma^2}\sum_{i=1}^n r_i^2 \;+\; \text{const}$$

**Step 5 — Drop the $\frac{1}{2\sigma^2}$ prefactor.** Since $\sigma^2$ is a fixed positive constant (from the noise model, not $\theta$), scaling the whole thing doesn't change where the minimum sits. Drop it too — you're left with:

$$\text{NLL}(\theta) \;\propto\; \sum_{i=1}^n r_i^2 \;=\; \sum_{i=1}^n (y_i - f_\theta(x_i))^2$$

**That's MSE.** The **squared** term isn't a design choice — it fell out of the exponent of the Gaussian PDF. Same drill for Laplace noise ($\propto \exp(-|r|/b)$) gives $\sum |r_i|$ — that's MAE. For Bernoulli it gives $-\sum [y_i \log p_i + (1-y_i)\log(1-p_i)]$ — that's binary cross-entropy.

**The equivalence.** "I chose MSE" and "I assumed my residuals are Gaussian" are the **same statement**. Same optimization problem, same $\hat\theta$.

#### Why MSE panics at outliers, and MAE doesn't

Look at how fast each PDF decays as the residual grows. Both are normalized so their peak at zero is 1:

| \|residual\| | Gaussian ∝ e^(−r²/2) | Laplace ∝ e^(−\|r\|) |
| ------------ | -------------------- | -------------------- |
| 0            | 1.000                | 1.000                |
| 1            | 0.607                | 0.368                |
| 2            | 0.135                | 0.135                |
| 3            | 0.011                | 0.050                |
| 4            | 0.0003               | 0.018                |
| 5            | 0.000004             | 0.0067               |
| 6            | 0.00000001           | 0.0025               |

A residual of 5 is "1 in 270,000" under Gaussian, but "1 in 150" under Laplace. When MLE encounters that residual under the Gaussian assumption, the model literally believes it shouldn't have happened — so it rewrites $\theta$ hard to shrink that one residual, even if that worsens fit on the other honest points.

The two shapes side by side (both peak at zero):

![Gaussian has rounded top and thin tails; Laplace has a sharp spike and heavier tails|697](assets/pdf-gaussian-vs-laplace.svg)

#### The loss shapes MLE actually minimizes

That tail behavior propagates into the shape of the loss itself. Per-example loss as a function of residual $r$:

| $r$  | MSE = $r^2$ | MAE = $\|r\|$ | Huber ($\delta$ = 1) |
| ---- | ----------- | ------------- | -------------------- |
| 0    | 0           | 0             | 0                    |
| 0.5  | 0.25        | 0.5           | 0.125                |
| 1    | 1           | 1             | 0.5                  |
| 2    | **4**       | 2             | 1.5                  |
| 3    | **9**       | 3             | 2.5                  |
| 5    | **25**      | 5             | 4.5                  |
| 10   | **100**     | 10            | 9.5                  |

MSE grows **quadratically** — one point with $r = 10$ contributes as much loss as a hundred points with $r = 1$. A single outlier drowns out 100 good points. MAE grows **linearly** — an outlier's vote is bounded. Huber is quadratic near zero (efficient on the bulk) and linear far away (tolerant of outliers).

![MSE curves quadratically, MAE stays linear, Huber tapers from quadratic near zero to linear at the tails|697](assets/loss-curves-mse-mae-huber.svg)

#### Huber loss — the pragmatic middle

**The problem.** MSE is smooth and gradient-friendly but panics at outliers. MAE ignores outliers but has a kink at zero — the gradient jumps from $-1$ to $+1$ across the origin, which makes optimization noisy near the minimum. You want the *smoothness of MSE for the well-behaved bulk* and the *tolerance of MAE for the tails*.

**The fix.** Glue them together at some threshold $\delta$:

$$
L_\delta(r) = \begin{cases}
\tfrac{1}{2} r^2 & \text{if } |r| \le \delta \\[2pt]
\delta\bigl(|r| - \tfrac{1}{2}\delta\bigr) & \text{if } |r| > \delta
\end{cases}
$$

Quadratic inside the "trust region" $[-\delta, \delta]$; linear outside. The $-\tfrac{1}{2}\delta$ term inside the linear branch isn't magic — it's what makes the two pieces meet smoothly at $|r| = \delta$ (both the value and the first derivative are continuous there). That C¹ continuity is why Huber is gradient-friendly everywhere, unlike MAE's kink.

**What $\delta$ controls.** It's the "how big does a residual have to be before I stop trusting it as normal noise?" knob.
- **Small $\delta$** (e.g. 0.5·σ of clean residuals) → most residuals treated as outliers → Huber behaves close to MAE.
- **Large $\delta$** (e.g. 5·σ) → almost every residual is inside the quadratic region → Huber behaves close to MSE.
- **Standard rule of thumb:** $\delta \approx 1.345 \cdot \hat\sigma$ (Huber's own recommendation) gives 95% asymptotic efficiency vs MSE when the data really is Gaussian, plus robustness when it isn't.

**Probabilistic interpretation.** In the MLE frame we've been using, MSE ↔ Gaussian noise, MAE ↔ Laplace noise. Huber corresponds to a noise distribution that's **Gaussian in the middle and Laplace in the tails** — the noise model you'd write down if you believed *most* observations are clean but some fraction are contaminated by heavy-tailed junk.

**Trying it out.**

```python
import numpy as np

def huber(r, delta=1.0):
    r = np.asarray(r, dtype=float)                          # coerce list/scalar to float array
    quad = 0.5 * r**2                                       # inner branch (|r| ≤ δ)
    lin = delta * (np.abs(r) - 0.5 * delta)                 # outer branch (|r| > δ)
    return np.where(np.abs(r) <= delta, quad, lin)          # elementwise pick per residual

r = np.array([0, 0.5, 1, 2, 3, 5, 10])
print("MSE  ", 0.5 * r**2)
print("MAE  ", np.abs(r))
print("Huber", huber(r, delta=1.0))
# Expected output:
# MSE   [ 0.     0.125  0.5    2.     4.5   12.5   50.   ]
# MAE   [ 0.   0.5  1.   2.   3.   5.  10. ]
# Huber [0.    0.125 0.5   1.5   2.5   4.5   9.5  ]
```

At $r = 10$: MSE contributes 50, MAE contributes 10, Huber contributes 9.5 — Huber matches MAE's growth rate for big residuals while keeping MSE's smoothness for small ones. (This snippet uses $\tfrac{1}{2}r^2$ so the quadratic-region math lines up with the definition above; the earlier table showed $r^2$ directly.)

**Where you'll meet it.** `sklearn.linear_model.HuberRegressor`, `scipy.special.huber`, PyTorch's `nn.SmoothL1Loss` (that's Huber with $\delta = 1$ rebranded), object-detection heads in Faster/Mask R-CNN and RetinaNet (the "smooth L1" for bounding-box regression). Robust statistics uses the same idea under the name *M-estimation*.

**Common pitfalls.**
- **Picking $\delta$ by eyeball on the training set** — do it on a validation split or use a robust scale estimate like $\hat\sigma = 1.4826 \cdot \text{MAD}$ (median absolute deviation).
- **Forgetting $\delta$ isn't scale-invariant** — if you standardize your target, you also have to rescale $\delta$, or its meaning flips.
- **Treating Huber as a silver bullet** — if outliers dominate (say >30% of the data is contaminated), the linear tail still lets them tilt the fit. At that point you want a redescending M-estimator like Tukey's biweight, which sends the loss to a constant in extreme tails.

#### What this does to a fit line

Five points on a line, one bad sensor reading:

![MSE fit tilts up to pacify the outlier; MAE fit ignores it and tracks the trend|697](assets/outlier-fit-mse-vs-mae.svg)

MSE sees $r^2$ blow up and bends $\theta$ to reduce that one residual, even though it worsens fit on the four honest points. MAE weights the outlier the same as any residual of comparable size and stays flat.

#### The decision, as a flow

![Decision tree: bell-shaped residuals → MSE; otherwise pick MAE, Huber, Poisson NLL, or cross-entropy by target shape|697](assets/loss-decision-tree.svg)

The habit worth building: before you pick a loss, **look at the residuals**. Losses are noise assumptions in disguise. Match the assumption to the data.

### Adding a prior: from MLE to MAP and regularization

**Problem with MLE.** Sometimes there isn't enough data to nail down $\theta$. MLE will pick whatever fits the limited data best, even if that's a wild solution.

**Fix.** Use Bayes' theorem to combine the data-driven likelihood with a prior belief — a built-in preference for "reasonable" $\theta$:

$$\hat\theta_{\text{MAP}} = \arg\max_\theta P(D | \theta) P(\theta)$$

Take logs: minimize $-\log P(D|\theta) - \log P(\theta)$. The first term is the data fit (the loss you already had). The second term is a penalty for $\theta$ being far from what the prior considers "reasonable."

This is **regularization**. And it's the *same idea* as Bayesian inference, just kept as a point estimate.

| Prior on $\theta$ | Equivalent regularizer |
|---|---|
| Gaussian $\mathcal{N}(0, \sigma^2)$ | **L2** regularization (ridge) — penalizes $\sum \theta_i^2$ |
| Laplace $\text{Lap}(0, b)$ | **L1** regularization (lasso) — penalizes $\sum |\theta_i|$ |

This explains a famous behavior: L1 produces *sparse* solutions (many zero coefficients), L2 doesn't. The Laplace prior has a sharp peak (non-differentiable kink) at zero, giving it a strong preference for exact zeros; the Gaussian is smooth and rounded at zero, so it only shrinks coefficients toward zero without setting them to it. The geometry of this is in [linear-algebra](linear-algebra.md) (the L1 ball has corners on the axes).

## 8. Why models fail to generalize: bias and variance

### The problem

You train a model. It nails the training set. On the test set, it bombs. Why?

Or: your training error and test error are both high. Different problem. Why?

These two failure modes are the bias-variance trade-off, and they're worth understanding deeply.

### The decomposition

For a model $\hat{f}(x)$ trying to predict $y = f(x) + \varepsilon$ where $\varepsilon$ is noise:

$$E[(y - \hat{f}(x))^2] = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\hat{f}(x))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible}}$$

The expected squared error breaks into three pieces. Each one tells you something different.

- **Bias** — on average across many possible training sets, how far is your model from the truth? High bias = your model is *systematically* wrong (too simple — straight line trying to fit a curve). This is **underfitting**.
- **Variance** — how much does your model change when you change the training set? High variance = your model latches onto noise (too flexible — fitting every wiggle in the data). This is **overfitting**.
- **Irreducible noise** — the part you can't beat. Even a perfect model has some error because the world has randomness.

The dartboard picture makes bias and variance concrete — each dart is one model trained on one dataset; repeated training gives the whole pattern:

![2×2 bullseye grid — low/high bias × low/high variance, with 10 darts each showing centered-vs-off and tight-vs-scattered|697](assets/bias-variance-bullseye.svg)

### Measure it in code

Fit a low-complexity model (degree-1 polynomial) and a high-complexity model (degree-10) on many bootstrap resamples. Evaluate each at a fixed test point $x^\* = 0.75$ (where $\sin(2\pi x^\*) = -1$). Compute $\text{Bias}^2$ (systematic offset from truth) and $\text{Variance}$ (spread across resamples):

```python
import numpy as np
rng = np.random.default_rng(seed=0)

def true_f(x):
    return np.sin(2 * np.pi * x)

n = 30                                    # training set size
x_star = np.array([0.75])                 # single test point
y_star_true = true_f(x_star)[0]           # = -1
n_bootstraps = 500                        # number of resampled training sets

def predictions(degree):
    preds = []
    for _ in range(n_bootstraps):
        x_train = rng.uniform(0, 1, size=n)                   # fresh random x on [0, 1]
        y_train = true_f(x_train) + rng.normal(0, 0.2, size=n)  # noisy targets, σ=0.2
        coef = np.polyfit(x_train, y_train, degree)           # least-squares polynomial fit
        preds.append(np.polyval(coef, x_star)[0])             # evaluate the fit at x*
    return np.array(preds)

for deg in [1, 10]:
    preds = predictions(deg)
    bias_sq = (preds.mean() - y_star_true) ** 2               # (avg prediction − truth)²
    variance = preds.var()                                    # spread of preds across bootstraps
    print(f"degree={deg:2d}: bias²={bias_sq:.4f}, variance={variance:.4f}, "
          f"total={bias_sq + variance:.4f}")
# Expected output:
# degree= 1: bias²=0.2646, variance=0.0179, total=0.2825
# degree=10: bias²=0.0001, variance=0.0189, total=0.0191
```

Degree-1 has high bias² (~0.26 — a line can't hit $-1$ at $x=0.75$ while averaging through 0-mean noise) and low variance. Degree-10 has near-zero bias² but higher variance (each bootstrap fits a wildly different curve). The two failure modes trade off — that's the whole story.

### Why this is a *trade-off*

Make the model more complex (more parameters, more flexibility) → bias goes down (it can fit more shapes), variance goes up (it fits the noise too).

Make the model simpler → bias goes up, variance goes down.

You're looking for the sweet spot. Tools that pull toward the bias side: regularization, simpler models, more data. Tools that pull toward the variance side: more parameters, more features.

### Modern caveat

This trade-off is the classical story for small models. Large neural networks routinely have low bias *and* low variance once they're heavily overparameterized — a phenomenon called **double descent**. For practical purposes, the trade-off remains a useful diagnosis tool: high training error → bias problem, low training error + high test error → variance problem.

## 9. Hypothesis testing: is this difference real?

### The problem

You ran an A/B test. In variant A, 500 out of 10,000 users converted (5.0%). In variant B, 520 out of 10,000 converted (5.2%). Is B *actually better*, or was that 20-user gap just coin-flip luck?

Everything in ML has this shape:
- Model A scores 91.0% on the test set, Model B scores 91.5%. Real improvement, or noise?
- You added a regularizer. Validation loss dropped from 0.412 to 0.397. Worth keeping?
- One feature's coefficient came out to 0.02. Is it *actually correlated* with the target, or would a random feature also produce a small nonzero number?

**The core difficulty.** Samples are always noisy. Even if A and B are truly identical, running an A/B test on 20,000 users won't give exactly equal conversion rates — you'll always see *some* gap, just from random variation in *which* users happened to be shown each variant. So the raw gap alone tells you nothing. What you need is: **"how big a gap would you expect purely by chance?"** If your observed gap is comfortably inside "random luck range," you shrug. If it's *way* outside that range, you have evidence the variants really differ.

Hypothesis testing is a framework for answering that "way outside" question with numbers.

### The core idea — "innocent until proven guilty"

Statisticians don't try to prove "B is better" directly. That's hard: any observed gap is *compatible* with a real difference, but it's also compatible with pure luck. So the workflow flips it around: **assume there's no difference, then ask how surprising the data is under that assumption.**

- If the data would be routine under "no difference" → shrug. You have no case.
- If the data would be *shocking* under "no difference" → the assumption looks unlikely, so you reject it.

This is exactly like the presumption of innocence in court. The defendant (the null hypothesis) is *assumed* innocent. The prosecutor (your data) has to produce evidence so strong that "innocent" no longer fits. If they can't, the defendant walks — *not* because innocence was proven, but because guilt wasn't.

That's the whole idea. Now the vocabulary:

- **Null hypothesis $H_0$** = "no effect, no difference" — the defendant. Concrete A/B version: "A and B have the same true conversion rate."
- **Alternative $H_1$** = "there is a difference" — what you'd conclude if $H_0$ falls.
- **Test statistic** = a single number summarizing the data. For our A/B example, the observed conversion-rate gap $\hat p_B - \hat p_A = 0.052 - 0.050 = 0.002$.
- **Null distribution** = the distribution of the test statistic *if $H_0$ is true*. Effectively: "if A and B are truly identical, and we ran this experiment a million times, what does the histogram of observed gaps look like?" It's usually roughly bell-shaped and centered at 0, with width set by the sample size (bigger $n$ → tighter bell → more precise experiment).
- **p-value** = the probability of seeing a gap *at least as extreme* as the one you actually got, under $H_0$. A tiny p-value means "if $H_0$ were true, this data would be freakishly rare."
- **Significance level $\alpha$** = the p-value threshold you commit to *before* looking (conventionally 0.05). "If p < $\alpha$, I'll reject $H_0$." It's the false-positive rate you're accepting.

### The classical (frequentist) workflow, on the A/B example

Working the example end to end:

1. **State $H_0$ and $H_1$.** $H_0$: A and B convert at the same rate. $H_1$: they don't.

2. **Pick the test statistic.** Observed gap: $\hat p_B - \hat p_A = 0.002$ (0.2 percentage points).

3. **Get the null distribution.** *If* the two variants convert identically, both are Binomial samples from the same rate $p \approx 0.051$ (pooled). The Central Limit Theorem (§6) tells us that with $n = 10{,}000$ per arm, the observed gap under $H_0$ is approximately $\mathcal{N}(0, \sigma^2)$ with $\sigma \approx 0.0031$ (from the standard two-proportion formula). *This is the "expected range of pure noise."*

4. **Compute the p-value.** Our gap is $0.002$. How many standard deviations out? $0.002 / 0.0031 \approx 0.65$. Under a Normal, the two-tailed probability of seeing $|z| \ge 0.65$ is about $0.52$. So **p ≈ 0.52** — half the time under $H_0$, we'd see a gap at least this big just by luck. This is *not surprising at all*.

5. **Compare to $\alpha$.** $0.52 > 0.05$. **Fail to reject $H_0$.** We do *not* have evidence that B is better. (Note: "fail to reject" ≠ "proven equal." Absence of evidence isn't evidence of absence.)

If instead we'd observed 570 out of 10,000 conversions in B (a 0.7-percentage-point gap), we'd get $z \approx 2.3$, p ≈ 0.02 → reject $H_0$ at $\alpha = 0.05$, conclude B is likely different.

Same logic runs behind t-tests (small samples, unknown variance), chi-squared tests (categorical data), z-tests (large samples, known variance), etc. Different test statistics, different null distributions — but always: *state $H_0$, compute the statistic, ask how weird it is under $H_0$, compare p to $\alpha$.*

### What the p-value is *not*

The p-value is the single most-misread number in statistics. Reciting what it isn't:

- **p is *not* "the probability the null is true."** It's the other direction: $P(\text{data this extreme} \mid H_0 \text{ true})$, not $P(H_0 \text{ true} \mid \text{data})$. Those are Bayes-related but *very* different quantities (§4).
- **p is *not* "the probability that your effect is real."** A p-value of 0.03 does *not* mean "there's a 97% chance B beats A." It means "if A = B, we'd see data this extreme 3% of the time." Same words, different math.
- **A small p doesn't mean a *big* effect.** With huge samples, even a laughable 0.001-percentage-point difference can be "significant." That's why you should always report effect sizes (and confidence intervals) alongside p-values.
- **A large p doesn't mean $H_0$ is *true*.** It only means you don't have enough evidence to reject. Maybe your sample is too small, or the effect is real but weak.

### Two ways your decision can be wrong

Once you decide "reject" or "fail to reject," reality gets a vote. Four outcomes, two of which are errors:

| | Reality: $H_0$ true (no effect) | Reality: $H_0$ false (real effect) |
|---|---|---|
| You reject $H_0$ | **Type I error** (false positive) — rate $\alpha$ | Correct |
| You fail to reject | Correct | **Type II error** (false negative) — rate $\beta$ |

- **Type I (false positive).** You concluded there's a difference when there isn't. Rate = $\alpha$ (which is why you set $\alpha = 0.05$: you're accepting a 5% false-positive rate).
- **Type II (false negative).** You missed a real effect. Rate = $\beta$. Depends on how big the true effect is *and* your sample size.
- **Power = $1 - \beta$.** The probability of detecting a real effect *if one exists*. Small samples → wide null distribution → low power → you miss real effects.

The picture below shows both errors at once. $H_0$'s distribution (left bell, centered at 0) is what you'd see if there's no effect. $H_1$'s distribution (right bell, shifted) is what you'd see if the true effect really is nonzero. The dashed vertical line is your decision threshold — reject $H_0$ if the test statistic falls to the right of it. **Red tail** = area under $H_0$ past the threshold = $\alpha$ (you rejected when you shouldn't have). **Orange tail** = area under $H_1$ *before* the threshold = $\beta$ (you failed to reject when you should have). Sliding the threshold left cuts $\beta$ but grows $\alpha$; sliding it right does the reverse. You can't shrink both at once — only more data can do that (narrower bells → less overlap → both errors shrink together).

![Two overlapping bell curves — H₀ centered at 0 and H₁ centered at 2, with red α tail on H₀'s right, orange β tail on H₁'s left, and purple power region highlighted|697](assets/type-i-ii-errors.svg)

### Confidence intervals — the more useful cousin

Reporting "**95% CI: [4.8%, 5.6%]**" for the conversion rate is almost always more useful than "p = 0.03." A p-value is a yes/no verdict against one specific null value; a **confidence interval** hands you the whole *range of parameter values compatible with the data*. If your CI doesn't overlap zero, you have (rough) evidence against the null — and you also see how big or small the plausible effect is.

The formally correct reading of "95% CI" is subtle: *"if I repeated this experiment many times and built a CI each time, 95% of those intervals would contain the true parameter."* People instinctively read it as "95% chance the true value is in *this* interval" — that's technically a **Bayesian credible interval**, philosophically different (talks about parameter uncertainty directly) but often numerically close in practice.

Rule of thumb: in a report, prefer effect size + CI over a naked p-value. It gives your audience the actual information they need.

### ML twins of statistical sins

The same failure modes reappear in ML dressed up in different words:

- **p-hacking** = running many statistical tests and reporting only the "significant" ones. **ML twin:** trying many architectures, hyperparameter combos, or random seeds and reporting only the best test-set number. Result: you've selected the lucky outlier — the score won't hold up in production. Fix: hold out a *final* test set you touch exactly once.
- **Multiple comparisons.** Running 20 independent tests at $\alpha = 0.05$ gives roughly one false positive *on average* even if every null is true. **ML twin:** running an ablation over 20 features and celebrating the "significant" one. Fix: correct for multiple comparisons (Bonferroni divides $\alpha$ by the number of tests; Benjamini-Hochberg / FDR is less conservative).
- **Data dredging / feature hunting.** Trying transformations, interactions, subsets until something works. Same problem — you've turned a test into a search, and searches find noise. Fix: pre-register hypotheses, or use a strict train/val/test split.
- **Confusing statistical significance with practical significance.** A model that improves accuracy from 91.00% to 91.02% may be "significant" with enough test data — but it's also probably not worth shipping. Report both p and effect size; make the ship decision on the effect.

## 10. Linear regression: the simplest "real" model

### The problem

You have features $X$ and a target $y$. You believe (or hope) that $y$ is roughly a linear function of $X$, plus some noise. Find the best linear function.

### Build it from intuition

Linear in the features means: $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots = X\beta$ (in matrix form, including a bias column), where $\beta_0$ is the **intercept** (the model's prediction when all features are 0) and $\beta_1, \ldots, \beta_p$ are the **slopes** — one per feature. (Reusing Greek letters — this $\beta$ is a regression coefficient, unrelated to Type II error $\beta$ in §9.)

"Best fit" means: minimize the sum of squared errors $\sum_i (y_i - \hat{y}_i)^2$.

We pick squared errors not because they're "obvious" but because (Section 7) MSE is the MLE under a Gaussian noise assumption. If we set up the problem as $y = X\beta + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ and solve $\arg\max_\beta P(y | X, \beta)$, we get exactly: minimize $\|y - X\beta\|^2$.

Take the derivative, set to zero, and you get a closed-form solution:

$$\hat\beta = (X^TX)^{-1} X^T y$$

This is the **Normal Equation**. Geometrically, it's the *projection* of $y$ onto the column space of $X$ — the closest point to $y$ that the model is capable of producing. (See [linear-algebra](linear-algebra.md) for the projection picture.)

Verify that MSE-minimization and Gaussian-NLL-minimization return the *same* $\hat\beta$:

```python
import numpy as np
rng = np.random.default_rng(seed=0)
n = 100
X = np.column_stack([np.ones(n), rng.normal(size=n)])  # bias column + one N(0,1) feature
true_beta = np.array([1.0, 2.0])
sigma = 0.5
y = X @ true_beta + rng.normal(0, sigma, size=n)       # @ is matrix multiply; add Gaussian noise

# Method 1: least-squares
beta_ls, *_ = np.linalg.lstsq(X, y, rcond=None)        # solves min ‖Xβ − y‖²; rcond=None silences warning

# Method 2: minimize Gaussian NLL directly
#   NLL = n/2 · log(2πσ²) + (1/2σ²) · Σ(y − Xβ)²
# For fixed σ, this is minimized by the same β as MSE.
resid = y - X @ beta_ls
mse = (resid ** 2).mean()
nll = 0.5 * n * np.log(2 * np.pi * sigma**2) + resid @ resid / (2 * sigma**2)  # resid @ resid = Σr²
print(f"β̂ (least squares) = {beta_ls}")     # ≈ [1.0, 2.0]
print(f"MSE = {mse:.4f}, NLL = {nll:.2f}")
print(f"NLL = (const) + MSE·n/(2σ²): "
      f"{0.5*n*np.log(2*np.pi*sigma**2) + mse*n/(2*sigma**2):.2f}")
# Expected output:
# β̂ (least squares) = [0.97248127 2.02754144]
# MSE = 0.2276, NLL = 68.11
# NLL = (const) + MSE·n/(2σ²): 68.11
```

The NLL is MSE plus additive constants that don't depend on $\beta$ — so *any* $\beta$ that minimizes MSE also minimizes NLL. That's what "MSE = MLE under Gaussian" means numerically.

### What we assumed without saying so

The closed-form solution + standard inferences (p-values on coefficients, confidence intervals) rest on assumptions:

1. **Linearity** — the relationship between features and target is, well, linear.
2. **Independence** of observations.
3. **Homoscedasticity** — error variance is constant across $X$.
4. **Normality of errors** — for inference (CIs, p-values). Not strictly needed just for prediction.
5. **No perfect multicollinearity** — features aren't linearly dependent (else $X^TX$ is singular and you can't invert).

Violations don't always kill you, but they invalidate the *uncertainty* claims about coefficients. Use diagnostic plots (residuals vs. fitted, QQ plots). (**QQ plot** — plots your residuals' quantiles against a Normal's; a straight line means residuals really are Gaussian.)

### When OLS isn't enough: regularized variants

When the model is unstable (multicollinearity) or has too many features for the data, plain OLS coefficients go wild. The fix is regularization — equivalent (Section 7) to adding a prior:

- **Ridge regression** — adds $\lambda \|\beta\|_2^2$. Shrinks all coefficients, never to zero. Prior: Gaussian.
- **Lasso** — adds $\lambda \|\beta\|_1$. Drives some coefficients exactly to zero (feature selection!). Prior: Laplace.
- **Elastic Net** — mixes both.

## 11. Logistic regression: the bridge to classification

### The problem

You want to predict a binary outcome — spam or not, click or not. You can't just use linear regression: it'll predict negative values and values bigger than 1, neither of which is a probability.

### Build it from intuition

What we want: the model's output should be in $[0, 1]$, interpretable as $P(y=1 | x)$.

What we have: a linear combination $z = x^T\beta$, which is a real number on $(-\infty, \infty)$.

We need to *squash* $z$ into $[0, 1]$. The standard squashing function is the **sigmoid**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

It's S-shaped: $\sigma(-\infty) = 0$, $\sigma(0) = 0.5$, $\sigma(+\infty) = 1$. Smooth, differentiable everywhere.

So: $P(y=1 | x) = \sigma(x^T\beta)$.

### Why sigmoid (not some other squash)?

Take the inverse. The **logit** (log-odds) is:
$$\log\frac{P(y=1)}{P(y=0)} = x^T\beta$$

So logistic regression is "linear in log-odds." A unit increase in feature $x_j$ multiplies the *odds* of $y=1$ by $e^{\beta_j}$. This is the standard interpretation people use when communicating logistic regression results.

### The loss

By Section 7 (MLE for a Bernoulli outcome), the loss falls out as **binary cross-entropy** / log loss:

$$\mathcal{L} = -\sum_i \left[y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i)\right]$$

If $y_i = 1$, the loss is $-\log \hat{p}_i$ — small when the model said high probability of class 1. If $y_i = 0$, the loss is $-\log(1 - \hat{p}_i)$ — small when the model said low probability of class 1. So pushing the right way reduces loss, in both cases.

### The catch

Unlike OLS, **there's no closed-form solution** for logistic regression. You must optimize numerically — gradient descent, Newton's method, or IRLS. This is why it's the natural bridge to neural networks: same loss family, just optimized iteratively. See [calculus](calculus.md) for the optimization side.

## 12. Information theory in two pictures

### Why ML cares

Cross-entropy is everywhere — it's what every classifier minimizes. But where does cross-entropy *come from*? From a beautifully physical idea: **information**.

### Entropy: how surprised should you be?

**The problem.** A weather app says "70% rain." If it rains, you're a little surprised but not shocked. If it had said "1% rain" and it rains, you'd be very surprised.

So: information content of an event = how surprised you are. Mathematically, surprise should be:
- Bigger when the event is unlikely.
- Zero for certain events.
- Additive for independent events (two independent surprises = total surprise).

The function that satisfies all three is: $\text{surprise}(x) = -\log P(x)$.

The **entropy** of a distribution is the *expected surprise*:
$$H(X) = -\sum_x P(x) \log P(x)$$

A uniform distribution has the highest entropy (you can never predict the outcome). A spike at one value has zero entropy (you always know what's coming).

### Cross-entropy: the cost of using the wrong distribution

**Problem.** You think the data follows distribution $Q$ (your model). The true distribution is $P$. How "expensive" is your model's mismatch with reality?

The expected number of bits needed to encode samples from $P$ using a code that's optimal for $Q$:
$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

If $Q = P$, cross-entropy equals entropy — minimum cost. The further $Q$ is from $P$, the higher the cost.

**ML connection.** When training a classifier, $P$ is the true label distribution (often a one-hot vector — all probability on the true class), $Q$ is your model's prediction. Cross-entropy loss is *literally measuring how badly your model's distribution differs from the truth*.

### KL divergence: how different are two distributions?

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

Cross-entropy minus entropy. Always $\geq 0$, zero iff $P = Q$. **Not symmetric** — $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$.

Verify all three quantities by hand and against `scipy.stats.entropy`:

```python
import numpy as np
from scipy import stats

P = np.array([0.7, 0.2, 0.1])   # true distribution
Q = np.array([0.5, 0.3, 0.2])   # model distribution

# By hand — using natural log (nats). scipy.stats.entropy uses natural log by default.
H_P = -np.sum(P * np.log(P))                    # Shannon entropy of P
H_PQ = -np.sum(P * np.log(Q))                   # cross-entropy H(P, Q)
KL_PQ = np.sum(P * np.log(P / Q))               # D_KL(P || Q)
KL_QP = np.sum(Q * np.log(Q / P))               # asymmetry check

print(f"H(P)         = {H_P:.4f}")
print(f"H(P, Q)      = {H_PQ:.4f}")
print(f"D_KL(P || Q) = {KL_PQ:.4f}")
print(f"D_KL(Q || P) = {KL_QP:.4f}   (≠ D_KL(P || Q) — asymmetric)")

# scipy check
assert np.isclose(stats.entropy(P), H_P)        # entropy(pk) → Shannon entropy of pk
assert np.isclose(stats.entropy(P, Q), KL_PQ)   # entropy(pk, qk) → KL(pk‖qk), NOT cross-entropy!
# Identity: H(P, Q) = H(P) + D_KL(P || Q)
assert np.isclose(H_PQ, H_P + KL_PQ)
# Expected output:
# H(P)         = 0.8018
# H(P, Q)      = 0.8869
# D_KL(P || Q) = 0.0851
# D_KL(Q || P) = 0.0920   (≠ D_KL(P || Q) — asymmetric)
```

Note the scipy gotcha: `stats.entropy(P, Q)` returns *KL divergence*, not cross-entropy. The identity `H(P, Q) = H(P) + D_KL(P||Q)` is why minimizing cross-entropy loss (with fixed one-hot $P$) is equivalent to minimizing KL — $H(P)$ is constant.

**ML uses:**
- **VAEs** (variational autoencoders) — loss has a "KL to prior" term.
- **Variational inference** — approximate intractable posteriors by minimizing KL.
- **Reinforcement learning** — policy regularization, often a KL constraint between old and new policy (TRPO, PPO).
- **Distillation** — train a small model to match a big model's output distribution by minimizing KL.

## Trade-offs

- **Frequentist vs. Bayesian.** Frequentist: parameters are fixed, data is random. Bayesian: parameters have distributions, posteriors describe belief. Bayesian is more principled and gives uncertainty for free; it's also slower and harder to compute. Most ML is implicitly frequentist, with occasional Bayesian flavor (priors as regularizers, MC dropout for uncertainty).
- **MLE vs. MAP vs. full Bayesian.** MLE: point estimate, no prior. Cheap, the default. MAP: point estimate + prior. Adds regularization. Full Bayesian: posterior distribution over parameters, uncertainty everywhere. Expensive but unbeatable when calibration matters.
- **Parametric vs. non-parametric models.** Parametric (linear regression, neural nets) have fixed-size parameter vectors. Non-parametric (kNN, kernel methods, Gaussian processes) grow with the data. Parametric is faster at scale; non-parametric is more flexible but doesn't scale to huge datasets.

## Common Pitfalls

- **Confusing PDF with probability.** $f(x) = 5$ does not mean a 500% chance. It means probability per unit width — only meaningful when integrated.
- **Misinterpreting p-values.** A p-value is *not* the probability the null is true. It's $P(\text{data} | H_0)$, full stop.
- **Multiple comparisons.** 20 tests at $\alpha = 0.05$ → ~1 false positive even when nothing's there. Use Bonferroni / FDR correction.
- **Correlation = causation.** They aren't equal. Always ask about confounders.
- **i.i.d. violation in time series.** Random train/test splits leak the future into training. Always split temporally.
- **Trusting confidence intervals as credible intervals.** They're philosophically different (frequentist vs. Bayesian) and occasionally numerically different. Be careful with very small samples or non-Normal posteriors.
- **MLE without enough data.** With limited data, MLE picks degenerate solutions (a Gaussian fit to one point has zero variance). Use MAP / regularization.
- **Picking MSE for heavy-tailed targets.** MSE assumes Gaussian noise. Outliers will bully your model. Try MAE or Huber.

## Decision Table
| Scenario | Tool | Why |
|----------|------|-----|
| Continuous target, residuals look bell-shaped | Linear regression with MSE | MLE under Gaussian noise |
| Binary target | Logistic regression / NN with sigmoid + BCE | MLE under Bernoulli |
| Multiclass target | Softmax + cross-entropy | MLE under Categorical |
| Heavy-tailed residuals | Huber / MAE / Laplace likelihood | MSE is bullied by outliers |
| Want feature selection | L1 (lasso) | Laplace prior pushes coefficients to zero |
| Want gentle shrinkage | L2 (ridge) | Gaussian prior shrinks but doesn't zero out |
| Need calibrated uncertainty | Bayesian inference / MC dropout / ensembles | Point estimates don't quantify uncertainty |
| Comparing two models on one test set | Paired t-test / McNemar | Account for the same items |
| Reporting accuracy with error bars | Bootstrap CI / CLT-based CI | Error bars *are* the statistical claim |

## References
- [*Mathematics for Machine Learning* — Deisenroth et al. (free PDF)](https://mml-book.github.io/) — Chapters 6 (probability) and 8 (continuous optimization).
- [*An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani](https://www.statlearning.com/) — Ch. 3 (linear regression), Ch. 4 (logistic regression / classification). Readable, applied.
- [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability) — full free curriculum, perfect for shoring up gaps.
- [*Deep Learning* — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/) — Chapter 3 (probability) is the cleanest deep-learning-flavored intro.
- [Mathematics for Machine Learning Specialization (Coursera)](https://www.coursera.org/specializations/mathematics-machine-learning) — formal coverage, with PCA capstone.

## See Also
- [Math for ML — Roadmap](INDEX.md)
- [linear-algebra](linear-algebra.md) — covariance matrices, regression as projection, the geometry behind L1 vs. L2
- [calculus](calculus.md) — how the models in this note actually get fit
