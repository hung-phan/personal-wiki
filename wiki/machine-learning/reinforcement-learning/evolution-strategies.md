---
name: "evolution-strategies"
description: "Evolution Strategies for reinforcement learning with runnable numpy code — perturbing policy *parameters* instead of actions, the generation loop (sample N Gaussian perturbations, run one episode each, z-score the returns, take a reward-weighted average of the perturbations), why the update is exact gradient ascent on the Gaussian-smoothed objective rather than a Taylor approximation, ES on CartPole with a 4-parameter linear policy reaching the 500 return cap with no backprop and no value function, the role of sigma and the learning rate, why ES parallelises almost perfectly and the shared-random-seed trick that sends only (seed, scalar reward) across the network, ES vs DDPG and policy gradients, and what the course omits from Salimans et al. 2017 (mirrored/antithetic sampling, rank-based fitness shaping). Study guide for Cutting-Edge AI: Deep Reinforcement Learning in Python, section 5."
created: 2026-11-13
updated: 2026-11-13
tags: [reinforcement-learning, evolution-strategies, derivative-free-optimization, black-box-optimization, parallelism, continuous-actions, numpy, study-guide, cutting-edge-deep-rl]
---
# Evolution Strategies: Reinforcement Learning Without a Gradient

> **How to read this note.** Study guide for *Cutting-Edge AI: Deep Reinforcement Learning in Python* (Lazy Programmer), section 5 (lectures 30–38): Evolution Strategies as a third family alongside value-based and policy-based RL. Read [policy-gradients](policy-gradients.md) first — ES is best understood as its mirror image, and that note's §4 covers hill climbing, ES's crude single-offspring ancestor. The gradient-based method for the same continuous-control tasks is [ddpg-continuous-control](ddpg-continuous-control.md). Returns and the basic RL loop are in [rl-foundations-mc-td](rl-foundations-mc-td.md); the instructor's shared appendix advice is in that note's §5. Snippets need only `numpy` and `gymnasium`.

## Why This Exists

**Problem.** Every method in [policy-gradients](policy-gradients.md) and [ddpg-continuous-control](ddpg-continuous-control.md) needs a differentiable path from a scalar loss back to the policy weights. That path forces real commitments: a stochastic policy you can take log-probabilities of, a value function or a bootstrap target, a discount factor to keep the sums finite, and a reward signal smooth enough that backprop through hundreds of steps means anything. Some problems refuse all of that — the fitness is a simulator's exit code, a game score, a non-differentiable metric, or arrives once after ten thousand steps.

**Key insight.** Stop perturbing *actions* and perturb *parameters* instead. Scatter a cloud of mutated weight vectors around your current policy, play one episode per mutant, and step in the reward-weighted average of the perturbations that scored above average. No backpropagation anywhere — yet the step you take is a genuine estimate of the gradient of expected return.

**Reach for this when:**
- The fitness is non-differentiable, discontinuous, or a black box (a score, a simulator, a hand-written metric).
- Rewards are sparse or arrive only at the end, and credit assignment across a long horizon is the actual difficulty.
- You have many cheap CPUs and few GPUs — ES turns wall-clock time into core count almost linearly.
- The policy is small (a linear map or a one-hidden-layer MLP) and you want something you can write in twenty lines of numpy.

**Don't reach for this when:**
- You have true labels and a true gradient. Supervised learning is exactly where ES loses; the course demonstrates this on MNIST and says so.
- Environment samples are expensive (real robots, paid API calls, human feedback). ES pays for its simplicity in episodes.
- The policy has millions of parameters. The estimator's sample requirement grows with the parameter count — see §5.

## Diagrams

![Two panels. Left: a reward contour surface with the current parameter vector at the centre, a Gaussian cloud of offspring around it, and the reward-weighted average of the perturbations as the update arrow pointing uphill. Right: the same generation distributed over workers, where each worker returns only a scalar return, and with a shared random seed only a seed plus a number crosses the network](assets/evolution-strategies.svg)

<!-- DIAGRAM SPEC: name=evolution-strategies | 2 panels side by side.
PANEL A (left), title "one generation in parameter space": concentric contour rings of a reward surface, higher toward the top right, labelled "expected return F(theta)". A filled green dot at the centre labelled "theta_t". Around it ~10 small blue dots on a dashed circle of radius labelled "sigma", each connected to theta_t by a thin grey vector labelled (only two of them, to avoid clutter) "sigma * eps_i". Next to three dots on the uphill side write small "+1.4", "+0.9", "+1.1"; next to three on the downhill side write "-1.2", "-0.8", "-1.0"; caption under them "standardised return A_i". One thick dark arrow from theta_t pointing uphill, labelled "theta_(t+1) = theta_t + (eta / N sigma) * sum_i A_i * eps_i". Small note: "positive A_i pulls toward that offspring, negative A_i pushes away".
PANEL B (right), title "why it parallelises": at the top a box "theta_t (identical on every worker)". Below it four worker boxes W1..W4, each containing two stacked lines "eps_i = randn(seed_i)" and "run 1 episode -> F_i". Arrows from the top box down to the workers labelled "broadcast seed_i". Arrows from each worker back up to a box "reduce" labelled "(seed_i, F_i)  = 2 numbers". From "reduce" one arrow to "theta_(t+1)". A callout beside the return arrows: "no gradients, no parameter vectors, no episodes cross the network".
ONE IDEA TO LAND: the update is a reward-weighted average of parameter-space perturbations, and because each worker needs only a scalar back, the whole generation costs two numbers of communication per offspring. -->

## 1. From one offspring to a population

The course opens with natural selection: many offspring, each only *slightly* different from the parent ("you're not going to have a monkey which gives birth to a giraffe"), fitness decides who reproduces, and small good changes compound over generations. The instructor is careful to add that biologically *inspired* is not biologically *accurate*, the same way neural networks have little to do with brains. He also makes the point that fitness is environment-relative — the sickle-cell allele resists malaria, so the same mutation is a liability in one environment and an asset in another. In RL the environment is fixed, so this doesn't bite, but it is a good reminder that "fitness" is a property of a pairing, not of a parameter vector.

The direct ancestor is **hill climbing**, covered in [policy-gradients](policy-gradients.md) §4: perturb the current best weights, score them, keep the perturbation if it beat the incumbent, discard otherwise. That is ES with a population of one and a hard accept/reject rule. Two things are missing, and both are what ES adds:

- **Multiple offspring per generation.** Once you have several, discarding all but the best throws away information: two mutants can be equally good in different directions, and the interesting question becomes how to *combine* them rather than which to keep.
- **Averaging over noise.** A fixed policy scores differently every run. One episode is a noisy sample of a mutant's fitness; a population gives you a usable estimate of the local shape of the return surface instead of a single accept/reject bit.

## 2. The algorithm

Three hyperparameters — learning rate $\eta$, perturbation standard deviation $\sigma$, population size $N$ — plus an initial parameter vector $\theta_0$. Parameters are always a flat 1-D vector; reshaping into weight matrices happens inside the fitness function, so the optimiser never knows what the policy looks like.

```
for t in 0, 1, 2, ...:
    for i in 1..N:                          # fully independent - this is the parallel part
        eps_i ~ N(0, I)                     # a vector, same dimension as theta
        F_i   = f(theta_t + sigma * eps_i)  # fitness: one episode's return (or accuracy, or f(x))
    A_i = (F_i - mean(F)) / std(F)          # standardise the returns across the population
    theta_(t+1) = theta_t + (eta / (N * sigma)) * sum_i A_i * eps_i
```

**Reading the update.** Each $\varepsilon_i$ is a direction in parameter space. The step is a sum of those directions weighted by how well they scored, so a mutant that did well pulls you toward it and — because the weight is negative after standardisation — a mutant that did badly pushes you the opposite way. That is the whole idea: *good reward, go there; bad reward, go away from there.*

**Why standardise.** With raw returns $F_i$, if every return is positive you move in *every* perturbation direction at once, including ones that are worse than where you already are: the population average drags you along. Subtracting the mean makes below-average offspring repel you and above-average offspring attract you; dividing by the standard deviation makes the step size independent of whether your returns are measured in units of 1 or 10,000. The course rightly calls this the same logic as using the advantage instead of the raw return in A2C ([actor-critic-a3c](actor-critic-a3c.md)).

**What ES lets you delete.** No Bellman equation, no MDP assumption, no value function, no discount factor, no backpropagation, no autodiff library — the course's ES scripts are pure numpy, and the instructor points out that shuttling these tiny policies to a GPU could be a net loss. The environment does not even have to *be* an MDP; ES never asks. Delayed rewards and very long episodes are handled without special machinery, because the only two things ES ever sees are a parameter vector and the total return it produced. The pile of hyperparameters you normally tune collapses to three.

## 3. It really is a gradient

The course derives the update with a first-order Taylor expansion (it says you may skip it). Scalar case: $f(\theta + \sigma\varepsilon) \approx f(\theta) + f'(\theta)\sigma\varepsilon$. Multiply by $\varepsilon$, take expectations over $\varepsilon \sim \mathcal{N}(0,1)$, and use $E[\varepsilon] = 0$ and $E[\varepsilon^2] = \mathrm{Var}(\varepsilon) = 1$:

$$E\big[\varepsilon\, f(\theta + \sigma\varepsilon)\big] \approx f(\theta)\underbrace{E[\varepsilon]}_{0} + f'(\theta)\,\sigma\underbrace{E[\varepsilon^2]}_{1} = \sigma f'(\theta) \quad\Longrightarrow\quad f'(\theta) \approx \frac{1}{N\sigma}\sum_i F_i\,\varepsilon_i.$$

The vector case is the same algebra on $f(x) \approx f(a) + (x-a)^\top \nabla f(a)$ with $\varepsilon \sim \mathcal{N}(0, I)$: independence kills every cross term $E[\varepsilon_i \varepsilon_j] = 0$ for $i \neq j$, only $E[\varepsilon_i^2] = 1$ survives, and stacking the surviving partials gives the gradient. The instructor flags his own index collision — in the derivation $i$ indexes a *component* of $\varepsilon$, in the algorithm it indexes *which offspring*. His punchline is worth keeping: "the nice thing about genetic algorithms is that we don't need gradient descent, but now we find that we're still kind of doing gradient descent."

**The cleaner statement, which the course does not give.** Define the **Gaussian-smoothed objective** $J(\theta) = E_{\varepsilon \sim \mathcal{N}(0,I)}\big[F(\theta + \sigma\varepsilon)\big]$ — the expected fitness in a blurred ball of radius $\sigma$ around $\theta$. Then

$$\nabla_\theta J(\theta) = \frac{1}{\sigma}\, E_{\varepsilon}\big[F(\theta + \sigma\varepsilon)\,\varepsilon\big],$$

and this is an **exact identity**, not a first-order approximation — it is the score-function (Stein) identity for a Gaussian, the same trick that produces the policy gradient in [policy-gradients](policy-gradients.md), applied to parameter noise instead of action noise. So the honest description of ES is: *exact stochastic gradient ascent on a smoothed version of the objective*, which is a stronger and more useful claim than "an approximation to gradient ascent". The smoothing is not an error term to apologise for — it is why ES tolerates a discontinuous fitness at all. (Caveat: standardising the returns turns the estimator into something that is no longer an unbiased estimate of that gradient. It is a variance-reduction heuristic, exactly like the baseline in REINFORCE. The course does not mention the bias.)

**Does twenty lines of this actually learn a policy?** The snippet below runs the course's version of ES — z-scored returns, no mirroring — on CartPole-v1 with a four-parameter linear policy (`push right if s · w > 0`), then tests the result on twenty episodes with seeds it never trained on, against a random weight vector.

```python
import numpy as np, gymnasium as gym
env = gym.make("CartPole-v1")
def rollout(w, seed):                                        # linear policy: push right if s.w > 0
    s, _ = env.reset(seed=seed); total = 0.0
    for _ in range(500):
        s, r, term, trunc, _ = env.step(int(s @ w > 0)); total += r
        if term or trunc: break
    return total
def es(gens=30, pop=50, sigma=0.1, lr=0.05, seed=0):         # the course's version: z-scored rewards, no mirroring
    rng = np.random.default_rng(seed); w = np.zeros(4)
    for g in range(gens):
        eps = rng.normal(size=(pop, 4))                      # one perturbation per population member
        R = np.array([rollout(w + sigma * e, seed=g) for e in eps])   # the only costly part - embarrassingly parallel
        A = (R - R.mean()) / (R.std() + 1e-8)                # standardise so the step size is scale-free
        w += lr / (pop * sigma) * (eps.T @ A)                # reward-weighted average of the perturbations
        if g % 10 == 9: print(f"  gen {g+1:2d}: population mean return {R.mean():6.1f}")
    return w
w = es()
rng = np.random.default_rng(7)
print("trained policy, 20 fresh episodes:", np.mean([rollout(w, seed=1000 + i) for i in range(20)]))
print("random policy,  20 fresh episodes:", np.mean([rollout(rng.normal(size=4), seed=1000 + i) for i in range(20)]))
print("no gradients, no backprop, no value function - only episode returns")
# Expected output:
#   gen 10: population mean return  499.5
#   gen 20: population mean return  500.0
#   gen 30: population mean return  500.0
# trained policy, 20 fresh episodes: 500.0
# random policy,  20 fresh episodes: 55.85
# no gradients, no backprop, no value function - only episode returns
```

Trace the loop against the pseudocode in §2: `eps = rng.normal(size=(pop, 4))` is the whole generation of perturbations drawn at once (one row per offspring — the course's scripts do the same, to cut RNG calls); the list comprehension over `rollout(w + sigma * e, ...)` is the only expensive line and the only one that touches the environment, which is why it is the line you would hand to fifty workers; `A = (R - R.mean()) / (R.std() + 1e-8)` is the standardisation; and `eps.T @ A` is the reward-weighted sum $\sum_i A_i \varepsilon_i$ written as one dot product. The population mean return climbs to 500 — CartPole-v1's truncation cap, i.e. every member of the population now balances the pole for the full episode — by generation 10, and the trained weights score 500 on all twenty fresh episodes against roughly 56 for random weights. Nothing in the file computes a derivative, a value estimate, or a discounted return.

Two caveats about that snippet, so you don't over-read it. All fifty offspring in a generation share one environment seed (`seed=g`), which removes rollout noise from the *comparison* within a generation — a sensible trick, and not something the course does. And CartPole with four parameters is the friendliest possible case for ES; §5 is about what happens as the parameter count grows.

## 4. Why ES parallelises almost perfectly

The inner loop's iterations do not communicate. The course's illustration: one episode takes an hour, population 50, so 50 hours sequentially — or ask fifty friends to each do one hour and email you the number back. That "email you the number" is the real content: **a worker's entire output is one scalar.** Not a gradient, not a parameter vector, not a trajectory.

**The seed trick, which the course never mentions.** Salimans et al. push this one step further, and it is the actual reason their results scale. Every worker holds the *same* $\theta_t$ and the same RNG implementation, so a worker can reconstruct any other worker's perturbation from the integer that seeded it. Workers therefore exchange $(\text{seed}_i, F_i)$ — a few bytes per offspring — and each reconstructs the full update locally. There is no parameter server, no gradient reduction, and the communication per generation is $O(N)$ *numbers* rather than $O(N \cdot |\theta|)$. The course gives the right intuition for the parallelism (independent iterations, "ask fifty friends") and quotes the paper's timings, but never explains the mechanism that makes those timings achievable. If you take one engineering idea from this note, take this one.

**The numbers the course quotes from the paper** (all from Salimans et al. 2017, read out in lecture 32 — not measured by the course, and not by this note):

| Claim | Figure |
|---|---|
| 3D Humanoid, same final performance | 18 cores → about 11 hours; 1,440 cores → about 10 minutes |
| Atari suite with CNN policies, ES vs A3C | ES better on 23 games, worse on 28 — "definitely comparable" |
| Wall clock | one day of A3C training ≈ one hour of ES |
| Data efficiency | ES needs 3–10× as many episodes |
| Compute | ES needs about 3× less, because there is no backpropagation |

**What you actually get on a laptop.** The course parallelises with `multiprocessing.pool.ThreadPool` and a `map` call — a *thread* pool that happens to live in the `multiprocessing` package, which the lecture's phrasing ("the multiprocessing library helps you do multithreading") obscures. Use `concurrent.futures.ThreadPoolExecutor` / `ProcessPoolExecutor` instead; the intent is legible from the class name. Three practical warnings, two of them the course's own:

- **Threads buy you little for environment stepping.** The GIL serialises Python-level game logic, which is most of what a rollout is. Numpy matmuls do release the GIL, which is why the course sees some benefit on its MNIST experiment and little elsewhere. Real ES speedup needs processes — or a free-threaded Python 3.13+ build, which changes the calculus.
- **Thread-unsafe environments fail silently.** PyGame Learning Environment is not thread safe: the course's Flappy Bird script produces consistently poor results with the pool enabled, with no error and no crash. The threading imports are commented out in that file for exactly this reason. Gym/MuJoCo environments are fine, so the MuJoCo script re-enables the pool.
- **Every generation waits for its slowest straggler.** The course observes that the best offspring in a generation typically scores far above the population average (it estimates roughly 10×), and in environments where a better policy survives longer — Flappy Bird, CartPole — that means the generation's wall-clock cost is set by its single luckiest mutant. Fixed-length episodes (HalfCheetah) don't have this problem.

## 5. Sigma, the learning rate, and the thing that gets worse with scale

$\sigma$ is doing two jobs at once, and the $1/\sigma$ in the update is the tell. It is the **radius of the smoothing** — how blurred the objective you are actually climbing is — and simultaneously the **finite-difference step** you are dividing by to recover a derivative. Push it either way and something breaks:

- **$\sigma$ too small.** Every offspring's return is within rollout noise of every other's. Standardisation then amplifies pure noise into unit-variance weights, and you random-walk. The failure is invisible in the loss curve, because the population mean looks flat rather than wrong.
- **$\sigma$ too large.** You are optimising a heavily blurred function whose optimum need not be anywhere near the real one, and the local linear picture behind §3's derivation stops holding.

The learning rate $\eta$ then scales a step whose *direction* is already normalised by $N$ and $\sigma$ and whose reward weights are already unit-variance — which is the useful consequence of standardisation: $\eta$ means roughly the same thing across environments with wildly different reward scales. The course's Flappy Bird script decays it each generation by the 300th root of 0.1 (so it reaches 10% of its initial value after 300 generations, which is also the only hint that the script runs 300 generations). Decaying $\sigma$ is implemented in the same file but commented out: the instructor tried it, saw no positive effect, and notes the paper reports the same.

**Stated hyperparameters: exactly four numbers, for one script.** The course gives `population_size=50`, `sigma=0.1`, `lr=1e-3` and `num_iters=500` for `es_simple.py`, which minimises a three-variable quadratic with a known optimum at $(0, 1, -2)$. For the MNIST, Flappy Bird and HalfCheetah scripts it reads out **no** population size, sigma, learning rate, generation count, hidden-layer width or thread count. All of its reported results are qualitative too: the quadratic's parameters come out "very close to their optimal values", MNIST is "not as good as a neural network trained with gradient descent, but still not that bad", Flappy Bird "learns to play quite well", and HalfCheetah "performs pretty nicely — not as good as DDPG, but not bad either". There are no accuracies, returns or reward curves anywhere in the section. Treat every number you want from this section as something you will have to tune yourself, and note that the scripts are unseeded, so even the instructor's own runs are not reproducible.

**The scaling weakness the course never raises.** A perturbation-based gradient estimate has variance that grows roughly with the **number of parameters**: you are probing a $d$-dimensional space with $N$ random directions, and $N$ is set by your core count, not by $d$. This is why ES is used with small policies — and why the course's own networks are, tellingly, a linear map and a single hidden layer — and why nobody trains large networks this way. The section presents ES's generality without this caveat, so supply it yourself: ES buys freedom from the gradient at a price measured in parameters, not just episodes.

## 6. What the course teaches vs what the paper does

The course implements **one** of the techniques in Salimans et al. 2017, and it is worth being precise about which, because the omissions are not footnotes in the paper — they are central to it.

| Technique | In the course? | What it is |
|---|---|---|
| Reward standardisation (z-score) | **Yes** — the only one | $A_i = (F_i - \bar F)/\mathrm{std}(F)$, as in §2 |
| Mirrored / antithetic sampling | **Never mentioned** | Evaluate $\theta + \sigma\varepsilon$ *and* $\theta - \sigma\varepsilon$ as a pair, so each $\varepsilon$ is used twice with opposite signs |
| Rank-based fitness shaping | **Never mentioned** | Replace returns by a centred rank transform before weighting, making the update depend only on the *ordering* of returns |
| Weight decay on $\theta$ | **Never mentioned** | Ordinary L2 regularisation on the parameter vector |
| Virtual batch normalisation | **Never mentioned** | Fixed reference-batch statistics, used for the paper's Atari policies |

Mirrored sampling exists to cancel the odd-order terms in the expansion of §3, and rank shaping exists to make the update invariant to reward scale and to stop one outlier offspring from dominating a generation — note that the course's own observation about the best offspring scoring roughly 10× the population mean is precisely the pathology rank shaping addresses. **Those are the paper's stated rationales, not measurements:** nothing in this note or the course quantifies what either technique buys, so don't carry away a number for "halves the variance" or "trains twice as fast". What you *can* carry away is that an ES implementation built only from this course's material is missing most of the reference implementation, and that if your ES run is unstable or sensitive to a single lucky episode, these are the first two things to add.

## 7. ES vs gradient-based RL

| | Policy gradients / DDPG | Evolution Strategies |
|---|---|---|
| What is perturbed | **actions** (sampled from $\pi$, or $\mu(s)$ + noise) | **parameters** ($\theta + \sigma\varepsilon$) |
| What crosses the wire from a worker | gradients or trajectories | one scalar (or `(seed, scalar)`) |
| Needs backprop | yes | **no** |
| Needs a value function / critic | A2C, DDPG: yes | no |
| Needs a discount factor | yes | no |
| Needs the problem to be an MDP | assumed | never asked |
| Long horizons, sparse or delayed reward | credit assignment is the hard part | indifferent — only the total return is seen |
| Non-differentiable or discontinuous fitness | can't | fine |
| Environment samples needed | fewer | **3–10× more** (paper's figure) |
| Scaling in parameter count | fine — millions | degrades; keep policies small |
| Hyperparameters | many | three: $\eta$, $\sigma$, $N$ |

**Where ES clearly loses: supervised learning.** With real labels you have a *true* gradient, and the course's position — gradient descent "pretty much consistently outperforms" ES there — is correct. Its MNIST run is a demonstration of generality, not a recommendation, and it comes with a methodological wrinkle to avoid copying: the fitness is **training accuracy**, with no validation split, after which both train and test scores are reported. That is model selection on the training set. (The lecture also tried log-likelihood as the fitness and found it worse than accuracy, which is why the likelihood path is commented out in the shipped code.)

RL is the opposite case, and that is the course's real argument: when the "gradient" you would be following is itself bootstrapped from the model's own estimates — a TD target, a learned critic — it was never a true gradient anyway, so giving it up costs less than it sounds like.

**Running the course's examples in 2026.** Three of its environment and tooling claims have expired. PLE (PyGame Learning Environment) is effectively abandoned, and the whole clone-and-`pip install -e .` install lecture is best replaced by `pip install flappy-bird-gymnasium`, which already speaks the Gymnasium API the course spends a lecture hand-wrapping. MuJoCo is no longer a $500-per-year licensed binary with an `mjkey.txt` dance — DeepMind open-sourced it under Apache 2.0 in 2021, `pip install mujoco gymnasium[mujoco]` is the whole install, Windows works, and the environment IDs have moved (`HalfCheetah-v2` → `v4`/`v5`). And the section's framing claim that Boston Dynamics' robots famously do *not* use machine learning to move was defensible when recorded and is not now: RL-trained locomotion and manipulation policies are mainstream in legged robotics.

## Trade-offs

- **Simplicity and generality vs sample cost.** Twenty lines of numpy, no autodiff, no MDP assumptions — paid for with 3–10× the episodes (the paper's figure). Free if your environment is a fast simulator, fatal if it's a real robot.
- **Cores vs time.** ES converts core count into wall clock almost linearly, which is a bargain when CPUs are the thing you have. It does not convert a single laptop into a cluster, and Python threads barely help (§4).
- **Parameter-space exploration vs a stochastic policy.** Mutation already explores, so the course's policies are deterministic (`argmax`, or `tanh` scaled to the action range) and the sampling code is commented out. You lose the entropy knob and any principled notion of policy stochasticity in return.
- **Smoothing vs fidelity.** $\sigma$ buys you tolerance of discontinuous fitness by optimising a blurred objective. Blur too much and the optimum you climb to is not the one you wanted.
- **Three hyperparameters vs no levers when it stalls.** Fewer knobs is genuinely easier — but when ES plateaus there is very little to adjust beyond $\sigma$, $N$ and the shaping tricks of §6.

## Common Pitfalls

- **Expecting ES to scale with the policy, not just the cluster.** Estimator variance grows with the parameter count while $N$ stays pinned to your core budget, so a big network silently converts your update into noise. Keep policies small; the course's own are a linear map and one hidden layer.
- **Shipping only z-scored rewards and calling it the ES paper.** Mirrored sampling and rank-based fitness shaping are missing from the course entirely (§6). An implementation without them is not the reference algorithm, so don't debug it against the paper's results.
- **Taking $\sigma$ from a toy problem.** The course's `sigma=0.1` was stated for a three-variable quadratic and nothing else. Too small and the whole population scores within rollout noise; standardisation then amplifies that noise into a confident-looking random walk.
- **Reporting training accuracy as the result.** The course's MNIST fitness *is* train accuracy with no validation split. Optimising a metric directly and then quoting it is model selection on the training set.
- **Enabling a thread pool on an unsafe environment.** PLE produces consistently poor results with no error at all. If parallelising makes your scores worse rather than faster, suspect the environment before the algorithm.
- **Running unseeded and trusting one run.** None of the course's ES scripts seed numpy, so neither its initial parameters nor its perturbations are reproducible. Seed, and compare several seeds, as in [policy-gradients](policy-gradients.md) §2.
- **Forgetting the straggler.** In a survive-longer environment the better your policy gets, the slower each generation runs, and the generation ends when its *last* episode does. Budget wall clock accordingly, or cap the episode length.
- **Carrying the course's install instructions forward.** The PLE clone, the MuJoCo licence key and the `-v2` environment IDs are all dead; see §7.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Non-differentiable or black-box fitness | ES | Policy gradients | No gradient path exists to follow |
| Continuous control, small policy, many CPUs | ES | [DDPG](ddpg-continuous-control.md) | Near-linear scaling, nothing to stabilise |
| Continuous control, sample budget is the constraint | [DDPG](ddpg-continuous-control.md) / TD3 / SAC | ES | Replay reuses every transition; ES throws the episode away |
| Sparse reward after a very long horizon | ES | TD-based methods | Only the total return matters; no credit assignment needed |
| Supervised learning with real labels | Gradient descent | ES | A true gradient beats an estimated one — the course agrees |
| Large network (millions of parameters) | Gradient-based RL | ES | Perturbation variance grows with parameter count |
| One perturbation, one accept/reject, a handful of weights | Hill climbing ([policy-gradients](policy-gradients.md) §4) | Full ES | Nothing to combine; ES's averaging buys nothing |
| ES unstable or dominated by one lucky episode | Add mirrored sampling + rank shaping | More population | Addresses the cause rather than averaging over it (§6) |

## References

- [Salimans et al. — Evolution Strategies as a Scalable Alternative to Reinforcement Learning (2017)](https://arxiv.org/abs/1703.03864) — the paper behind this section; source of the Humanoid/Atari numbers, mirrored sampling, rank shaping and the shared-seed scheme.
- [Wierstra et al. — Natural Evolution Strategies](https://arxiv.org/abs/1106.4487) — the search-gradient framework the above specialises.
- [Hansen — The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772) — the adaptive-covariance branch of ES, which the course does not cover.
- [Such et al. — Deep Neuroevolution (2017)](https://arxiv.org/abs/1712.06567) — a plain genetic algorithm as a further point of comparison on Atari.
- [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) — the environment in §3's snippet, including the 500-step truncation cap.
- [MuJoCo](https://mujoco.org/) · [flappy-bird-gymnasium](https://pypi.org/project/flappy-bird-gymnasium/) — current replacements for the course's two install lectures.
- [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) — the modern thread/process pool the course's `ThreadPool` code predates.

## See Also

- [policy-gradients](policy-gradients.md) — perturbing actions instead of parameters; §4 is hill climbing, ES's single-offspring ancestor.
- [ddpg-continuous-control](ddpg-continuous-control.md) — the gradient-based method for the same continuous-control tasks, from the same course.
- [rl-foundations-mc-td](rl-foundations-mc-td.md) — returns, the RL loop, and (§5) the instructor's shared appendix advice.
- [actor-critic-a3c](actor-critic-a3c.md) — the advantage, whose logic standardised ES returns reuse, and A3C, the baseline the ES paper compares against.
- [model-selection-boosting](../classical-ml/model-selection-boosting.md) — random search over hyperparameters: the same "sample, score, keep what works" idea with no update direction at all.
