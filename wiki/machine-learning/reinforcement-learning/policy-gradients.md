---
name: policy-gradients
description: Policy gradient methods explained with runnable PyTorch and numpy code — parametrising the policy with a softmax over action scores, the performance objective, the policy gradient theorem and log-derivative trick giving loss = −Σ G·log π(a|s) (REINFORCE), why a value baseline (advantage G − V(s)) cuts variance, REINFORCE with vs without baseline on CartPole over 3 seeds, Gaussian policies for continuous actions (mean, softplus standard deviation, clipping, entropy bonus) with autograd gradients, hill climbing on MountainCarContinuous as a gradient-free alternative, the course's settings (Adagrad 0.1, value net with 10 hidden units, RBF features, 50-episode actor-critic run). Study guide for Deep Reinforcement Learning in Python, section 5.
created: 2026-11-13
updated: 2026-11-13
tags: [reinforcement-learning, policy-gradient, reinforce, baseline, gaussian-policy, continuous-actions, hill-climbing, pytorch, study-guide, deep-rl-python]
---
# Policy Gradients: Learning the Policy Directly

> **How to read this note.** Study guide for *Advanced AI: Deep Reinforcement Learning in Python*, section 5 (lectures 36–45): the policy gradient derivation, CartPole with a softmax policy, then MountainCarContinuous with a Gaussian policy — first by hill climbing, then by gradient ascent. The course's code is TF1/Theano; here it's PyTorch and numpy on Gymnasium, and the course's own settings (checked against its public repo) are given alongside. Read [rl-foundations-mc-td](rl-foundations-mc-td.md) and [function-approximation-td-lambda](function-approximation-td-lambda.md) first. The actor-critic, A3C and PPO continuation is in [actor-critic-a3c](actor-critic-a3c.md).

## Why This Exists

**Problem.** Value-based methods pick actions by taking a max over Q-values. That breaks when actions are continuous (a throttle, a steering angle — infinitely many actions), and it can only produce deterministic policies, even when acting randomly is the right strategy.

**Key insight.** Make the policy itself a differentiable function that outputs action **probabilities**, play, and nudge the parameters so that actions followed by high returns become more likely: the gradient is $E[G \cdot \nabla \log \pi(a \mid s)]$, which you can estimate from episodes.

**Reach for this when:**
- Actions are continuous or numerous.
- You want a stochastic policy (games with bluffing, partial observability).
- Reading PPO, A2C, SAC or RLHF code — they all start from this loss.

**Don't reach for this when:**
- Environment steps are expensive — plain policy gradients throw each episode away after one update; off-policy methods (DQN, SAC) reuse data.
- A small discrete problem — Q-learning is simpler.

## Diagrams

![Policy gradient loop: play an episode by sampling from π, compute returns backwards, subtract the baseline V(s) to get the advantage, minimise −Σ advantage · log π; softmax for discrete actions; a Gaussian with mean μ(s) and softplus σ(s) for continuous actions](assets/policy-gradients.svg)

## 1. From "make good actions likely" to a loss

**Parametrise the policy.** Score each action with a linear model (or a network) of the state and turn the scores into probabilities with softmax: $\pi(a \mid s) = \text{softmax}(\theta^\top s)_a$.

**The objective.** The course calls it the *performance*: the expected return from the start, $\eta(\theta) = E[G_0]$. We want to climb it.

**The gradient.** The **policy gradient theorem** says $\nabla \eta \propto \sum_s \mu(s) \sum_a Q(s, a)\, \nabla \pi(a \mid s)$, where $\mu$ is how often states are visited. Multiply and divide by $\pi$, and use $\nabla \log \pi = \nabla \pi / \pi$:

$$\nabla \eta \propto E\big[\, Q(s, a)\, \nabla \log \pi(a \mid s) \,\big] = E\big[\, G_t\, \nabla \log \pi(a_t \mid s_t) \,\big],$$

replacing $Q$ by a sampled return $G_t$ (its expectation). Average over an episode, and since optimisers minimise:

$$\text{loss} = -\sum_t G_t \log \pi(a_t \mid s_t).$$

This is **REINFORCE** (the course doesn't use the name). Read the single-sample step $\theta \leftarrow \theta + \alpha\, G\, \nabla \pi / \pi$: the push is proportional to the return, in the direction that raises the action's probability, and *divided* by that probability — so a rare action that paid off gets a big boost. It's a Monte Carlo method: it needs the whole episode's returns. (The state distribution $\mu$ isn't dropped: sampling states on-policy supplies it. What practical implementations drop is a $\gamma^t$ factor on each step's term, which makes the estimate biased for the discounted objective. It's standard practice anyway.)

## 2. The baseline: same direction, less noise

$G_t$ varies wildly between episodes, so the gradient estimate is noisy. Subtract any function of the state alone — it doesn't change the expected gradient — and the natural choice is the value $V(s)$, learned by ordinary regression on the returns. The difference $G_t - V(s_t)$ is the **advantage**: was this better than usual from here?

```python
import numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.set_num_threads(1)
def train(baseline, episodes=400, seed=0):
    torch.manual_seed(seed); env = gym.make("CartPole-v1")
    policy = nn.Linear(4, 2)                                     # the course's choice: linear scores -> softmax
    value = nn.Sequential(nn.Linear(4, 10), nn.Tanh(), nn.Linear(10, 1))
    opt = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=1e-2)
    lengths = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 1000 + ep); S, A, R, done = [], [], [], False
        while not done:
            a = torch.distributions.Categorical(logits=policy(torch.tensor(s))).sample().item()
            S.append(s); A.append(a); s, r, term, trunc, _ = env.step(a); R.append(r); done = term or trunc
        G, g = [], 0.0
        for r in reversed(R): g = r + 0.99 * g; G.insert(0, g)  # returns, computed backwards
        S, A, G = torch.tensor(np.array(S)), torch.tensor(A), torch.tensor(G)
        V = value(S).squeeze(-1)
        adv = (G - V.detach()) if baseline else G               # advantage = G − V(s): same direction, less noise
        logp = torch.distributions.Categorical(logits=policy(S)).log_prob(A)
        loss = -(adv * logp).sum() / 100 + ((G - V) ** 2).mean() * baseline
        opt.zero_grad(); loss.backward(); opt.step(); lengths.append(len(R))
    return round(float(np.mean(lengths[-50:])))
for b in (False, True):
    print(f"baseline={b!s:5}  mean length of last 50 episodes over 3 seeds:", [train(b, seed=k) for k in range(3)])
print("random policy: about 22")
# Expected output:
# baseline=False  mean length of last 50 episodes over 3 seeds: [314, 290, 208]
# baseline=True   mean length of last 50 episodes over 3 seeds: [378, 446, 376]
# random policy: about 22
```

Same linear softmax policy, same learning rate, same episodes: the baseline version ends higher on every seed and varies less across seeds. Both crush the random policy. The course's CartPole: linear softmax policy trained with Adagrad (learning rate 0.1), a value network with one hidden layer of 10 units, γ = 0.99, 1,000 episodes, one update per episode. The instructor stresses how sensitive it is to the optimiser choice — try several.

**Actor-critic.** Replace $G_t$ by the one-step target $r + \gamma V(s')$ and you can update after every step: the policy is the *actor*, $V$ the *critic*. The course mentions this here and uses it for MountainCarContinuous (§4); the full treatment is in [actor-critic-a3c](actor-critic-a3c.md).

## 3. Continuous actions: a Gaussian policy

For a continuous action, output the **mean** $\mu(s)$ and **standard deviation** $\sigma(s)$ of a normal distribution, sample from it, and clip to the valid range. $\sigma$ must stay positive, so it goes through softplus ($\log(1 + e^x)$) or exp; the course adds a small constant so it can't collapse to zero. The loss is unchanged: $-\text{advantage} \cdot \log \mathcal{N}(a; \mu, \sigma)$.

```python
import torch
# Continuous action (e.g. throttle in [−1, 1]): the policy outputs a Gaussian's mean and spread.
mean, std = torch.tensor(0.2, requires_grad=True), torch.tensor(0.5, requires_grad=True)
dist = torch.distributions.Normal(mean, std)
a = torch.tensor(0.9)                                    # the action we sampled and took
for adv in (+1.0, -1.0):                                 # it turned out better (+) or worse (−) than expected
    mean.grad = std.grad = None
    loss = -adv * dist.log_prob(a); loss.backward(retain_graph=True)   # same policy-gradient loss as the discrete case
    print(f"advantage {adv:+.0f}: gradient descent moves mean {'up' if mean.grad < 0 else 'down'}"
          f" (−gradient = {-mean.grad.item():+.2f}), std {'up' if std.grad < 0 else 'down'} (−gradient = {-std.grad.item():+.2f})")
print("entropy of N(0.2, 0.5):", round(dist.entropy().item(), 3), "nats — grows with std, so an entropy bonus keeps it exploring")
print("sampled action, clipped to the valid range:", torch.clamp(torch.tensor(1.7), -1, 1).item())
# Expected output:
# advantage +1: gradient descent moves mean up (−gradient = +2.80), std up (−gradient = +1.92)
# advantage -1: gradient descent moves mean down (−gradient = -2.80), std down (−gradient = -1.92)
# entropy of N(0.2, 0.5): 0.726 nats — grows with std, so an entropy bonus keeps it exploring
# sampled action, clipped to the valid range: 1.0
```

An action that turned out better than expected pulls the mean toward it — and, because it was far from the mean (0.9 vs 0.2), *widens* σ: the policy learns that exploring further paid off. A worse-than-expected action does the opposite. An **entropy bonus** (entropy of a Gaussian grows with σ) keeps σ from shrinking too early.

Why not Q-learning here? One Q output per action can't cover a continuum, and maximising over a continuous action needs an inner optimisation. (The course says it's impossible; DDPG, TD3 and SAC do it by learning a policy network that outputs the maximising action — they're actor-critics.)

## 4. MountainCarContinuous: hill climbing, then gradients

The task: one action, a force in [−1, 1]; +100 for reaching the flag, minus 0.1 × force² every step. **Solved** = average return of 90 or more over 100 episodes. The fuel penalty is a trap for gradient methods: early on, any force costs reward and the flag is rarely reached, so the gradient says "do nothing".

**Hill climbing** (a gradient-free search the course uses first): perturb the current best parameters, play a few episodes, keep the perturbation if it scores higher, occasionally restart from scratch.

```python
import numpy as np, gymnasium as gym
rng = np.random.default_rng(0)
env = gym.make("MountainCarContinuous-v0")               # action: force in [−1, 1]; +100 at the flag, −0.1·force² per step
feat = lambda s: np.array([s[0], s[1] * 10, 1.0])       # position, scaled velocity, bias
def score(w, episodes=3, seed=0):
    total = 0.0
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i); done = False
        while not done:
            a = np.clip([np.tanh(feat(s) @ w)], -1, 1)   # deterministic policy for the search
            s, r, term, trunc, _ = env.step(a); total += r; done = term or trunc
    return total / episodes
best_w = np.zeros(3); best = score(best_w)
print(f"do nothing: {best:.1f}")
for it in range(100):                                    # hill climbing: perturb, keep if better
    w = best_w + rng.normal(0, 0.5, 3)
    if rng.random() < 0.1: w = rng.normal(0, 1, 3)       # occasionally restart from scratch (as in the course)
    if (sc := score(w)) > best: best_w, best = w, sc
print(f"after 100 hill-climbing steps: {best:.1f}   weights {best_w.round(2)}")
print(f"fresh episodes: {score(best_w, 20, seed=1000):.1f}   (solved = average 90+)")
# Expected output:
# do nothing: 0.0
# after 100 hill-climbing steps: 99.1   weights [-0.2   0.7  -0.09]
# fresh episodes: 93.9   (solved = average 90+)
```

Three weights (position, velocity, bias) are enough: the found policy pushes in the direction of the car's velocity — rock with the motion until the momentum carries it over the hill. The course does the same with a Gaussian policy on 100 RBF features: 100 search iterations, 3 episodes per candidate, a 10% chance of a full restart, then a 100-episode check. It's not a policy-gradient method, despite the scripts' names.

**Gradient ascent version.** The course then trains properly: a Gaussian policy and a linear value model on RBF features, updated **every step** with the TD advantage (an actor-critic), Adam (1e-3 for the policy, 0.1 for the value), entropy weight 0.1, γ = 0.95, 50 episodes. The instructor reports that runs are inconsistent — some reach the flag, some don't — and plots the cost-to-go of the learned value function.

## Trade-offs

- **Policy gradients vs Q-learning.** Continuous and stochastic actions, smooth policy changes vs less sample-efficient (on-policy), high-variance gradients.
- **REINFORCE vs actor-critic.** Unbiased (only if the $\gamma^t$ factor is kept, see §1), simple, waits for episode ends vs every-step updates and lower variance, but a biased critic.
- **Gradient methods vs hill climbing.** Scale to millions of parameters vs robust to deceptive rewards and trivial to implement, but only for a handful of parameters.
- **Entropy weight.** More exploration vs slower convergence.

## Common Pitfalls

- **Not detaching the baseline.** The advantage must be a constant in the policy loss; otherwise the policy loss trains the value network.
- **σ collapsing.** Without softplus plus a floor (or an entropy bonus), the policy becomes deterministic and stops exploring.
- **Clipping before computing log π.** Take the log-probability of the *sampled* action; clip only what you send to the environment.
- **Judging from one seed.** Policy gradients are noisy; compare over several seeds (as in §2).
- **Big learning rates.** One bad update can wreck the policy; PPO's clipping exists for this ([actor-critic-a3c](actor-critic-a3c.md) §5).
- **Ignoring the reward's shape.** Penalties on effort make "do nothing" a local optimum; search, shaping or more exploration gets past it.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Discrete actions, cheap episodes | REINFORCE + baseline | Without baseline | Same expected gradient, less noise |
| Continuous actions | Gaussian policy (or SAC/PPO) | One Q output per action | Infinite actions |
| Few parameters, deceptive reward | Hill climbing / evolution strategies | Plain gradients | Gradient points to "do nothing" |
| Updates must happen mid-episode | Actor-critic (TD advantage) | REINFORCE | No need to wait for G |
| Unstable training | PPO clipping, smaller LR | Bigger network | Bounds each policy change |

## References

- [Williams — Simple statistical gradient-following algorithms (REINFORCE, 1992)](https://link.springer.com/article/10.1007/BF00992696)
- [Sutton et al. — Policy Gradient Methods for RL with Function Approximation (2000)](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- [Sutton & Barto, 2nd ed., ch. 13](http://incompleteideas.net/book/the-book-2nd.html) · [OpenAI Spinning Up — intro to policy optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Gymnasium MountainCarContinuous](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)

## See Also

- [actor-critic-a3c](actor-critic-a3c.md) — actor-critic, A3C/A2C, PPO and SAC.
- [function-approximation-td-lambda](function-approximation-td-lambda.md) — the RBF features the course's policies use.
- [rl-foundations-mc-td](rl-foundations-mc-td.md) — returns and Monte Carlo, which REINFORCE builds on.
- [qlora-fine-tuning](../llm-engineering/qlora-fine-tuning.md) — RLHF puts a policy-gradient step on top of fine-tuning.
