---
name: actor-critic-a3c
description: Actor-critic reinforcement learning and A3C explained with runnable PyTorch code — policy (actor) and value (critic) heads on one shared network, the advantage A = r + gamma·V(s') − V(s), policy-gradient loss −log π(a|s)·A, critic loss, entropy bonus, n-step returns, asynchronous (A3C) vs synchronous batched (A2C) workers, Gymnasium vector environments and autoreset; the course's Kung Fu Master agent (4 stacked 42x42 grayscale frames, 3 conv layers, actor 14 actions, critic 1 value, 10 environments, Adam 1e-4, entropy 0.001, reward scaling 0.01); an A2C that learns CartPole in seconds; PPO's clipped objective and SAC's entropy-regularised value with worked numbers, and the AWS DeepRacer workshop; the asynchronous TF1 A3C for Breakout from Deep RL in Python (threads per CPU, 5-step rollouts, entropy 0.01, RMSProp, gradient clipping at norm 5). Study guide for AI A-Z Parts 7–8 (sections 14–17) and Deep Reinforcement Learning in Python section 7.
created: 2026-11-13
updated: 2026-11-13
tags: [reinforcement-learning, actor-critic, deep-rl-python, a3c, a2c, policy-gradient, ppo, sac, gymnasium, pytorch, study-guide, ai-a-z]
---
# Actor-Critic, A3C, PPO and SAC

> **How to read this note.** Study guide for *AI A-Z*, Part 7 (A3C, sections 15–16: playing Kung Fu Master) and Part 8 (PPO and SAC, section 17). Part 8 is a two-minute motivation video plus a link to an external one-hour AWS DeepRacer workshop that trains PPO and SAC for a self-driving car; that workshop isn't part of the course text, so §5 covers the two algorithms' core ideas on its own. Read [q-learning](q-learning.md) and [deep-q-learning](deep-q-learning.md) first; [policy-gradients](policy-gradients.md) derives the actor's loss. §6 adds the threaded A3C from *Deep Reinforcement Learning in Python* (section 7). The snippets need `torch` and `gymnasium`; the CartPole actor-critic trains in about 5 seconds.

## Why This Exists

**Problem.** DQN learns values and acts greedily on them. That struggles with stochastic policies, large or continuous action spaces, and it learns from one stream of experience at a time. And the A3C diagrams — asynchronous agents, shared networks, two heads, an LSTM — are confusing until you see the few lines of loss that make it work.

**Key insight.** Learn the **policy** directly (the actor) and use a learned **value** (the critic) as a baseline: after each step, compare what happened with what the critic expected. Better than expected → make that action more likely; worse → less. Running many environments at once gives varied, decorrelated experience without a replay buffer.

**Reach for this when:**
- Working through AI A-Z Parts 7–8.
- You need a stochastic policy, or many/continuous actions.
- Reading PPO or SAC papers and configs (both are actor-critics).

**Don't reach for this when:**
- The problem is small and discrete — DQN or even tabular Q-learning is simpler.
- Environment steps are very expensive — off-policy methods (DQN, SAC) reuse data better than on-policy A2C/PPO.

## Diagrams

![Actor-critic: stacked frames through three conv layers and a dense layer into an actor head (14 action logits) and a critic head (V(s)); the advantage; loss = actor + critic − entropy bonus; A3C asynchronous workers vs A2C batched updates; where PPO and SAC fit](assets/actor-critic.svg)

## 1. Actor and critic

**The question.** How do you improve a policy when you only see rewards?

Try an action, then ask: *was that better or worse than usual from here?* The **critic** answers "usual": it learns $V(s)$, the expected return from state $s$. The **actor** is the policy $\pi(a \mid s)$, a probability for each action. After a step, the **advantage**

$$A = \underbrace{r + \gamma V(s')}_{\text{what happened}} - \underbrace{V(s)}_{\text{what was expected}}$$

says how much better the action did than the critic predicted. The losses:

- **Actor:** $-\log \pi(a \mid s) \cdot A$ — gradient descent on this raises the probability of actions with positive advantage and lowers it for negative ones. (This is the policy-gradient idea; subtracting $V(s)$ as a baseline cuts its variance enormously.)
- **Critic:** $A^2$ — learn to predict returns better.
- **Entropy bonus:** subtract $\beta H(\pi)$ so the policy doesn't collapse to one action before it has explored. (The course narrates the entropy as "added"; added to the objective, subtracted from the loss.)

Both heads share the layers below them: features that help predict value also help choose actions. In the course's terminology, the actor outputs are sometimes called Q-values; they're policy logits.

## 2. Asynchronous, advantage, actor-critic

The 2016 **A3C** paper ran many workers, each with its own environment and network copy, computing gradients on short rollouts and applying them to a **shared** network whenever ready — no replay buffer needed, because the workers' experiences are naturally different. Later work found the asynchrony itself wasn't the key: **A2C**, which steps all environments together and makes one batched update, works as well and uses GPUs better. The course's code is A2C in practice: 10 environments stepped in lockstep in one process. The intuition lectures also describe an **LSTM** layer for memory; the course's implementation doesn't use one (it stacks frames instead).

## 3. A working actor-critic

CartPole again (see [deep-q-learning](deep-q-learning.md) §4 for DQN on the same task), with 8 environments stepped together and **5-step rollouts**: the return used for the advantage is $r_t + \gamma r_{t+1} + \dots + \gamma^4 r_{t+4} + \gamma^5 V(s_{t+5})$ — the n-step idea from the course's eligibility-trace lectures, which spreads reward back faster than one step.

```python
import numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.manual_seed(0); torch.set_num_threads(1)
N, STEPS = 8, 5                                            # 8 parallel envs, 5-step rollouts
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(N)],
                                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)   # finished envs restart at once
class ActorCritic(nn.Module):                              # one shared body, two heads
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 128), nn.ReLU())
        self.actor, self.critic = nn.Linear(128, 2), nn.Linear(128, 1)   # action logits, state value V(s)
    def forward(self, s):
        h = self.body(s); return self.actor(h), self.critic(h).squeeze(-1)
net = ActorCritic(); opt = torch.optim.Adam(net.parameters(), lr=7e-4)
gamma, beta = 0.99, 0.01                                   # discount, entropy bonus weight
s, _ = envs.reset(seed=0); ep_ret, finished = np.zeros(N), []
for it in range(2000):
    logps, values, ents, rewards, alive = [], [], [], [], []
    for _ in range(STEPS):                                 # collect a short rollout from every env
        logits, v = net(torch.tensor(s)); dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
        s, r, term, trunc, _ = envs.step(a.numpy())
        logps.append(dist.log_prob(a)); values.append(v); ents.append(dist.entropy())
        rewards.append(torch.tensor(r, dtype=torch.float32)); alive.append(torch.tensor(1.0 - (term | trunc), dtype=torch.float32))
        ep_ret += r; done = term | trunc; finished += list(ep_ret[done]); ep_ret[done] = 0
    with torch.no_grad(): R = net(torch.tensor(s))[1]      # bootstrap from the critic after the last step
    returns = []
    for r, m in zip(reversed(rewards), reversed(alive)):  # n-step return: r_t + γ r_t+1 + ... + γ^n V(s_t+n)
        R = r + gamma * R * m; returns.insert(0, R)
    adv = torch.stack(returns) - torch.stack(values)       # advantage: better or worse than the critic expected?
    actor_loss = -(torch.stack(logps) * adv.detach()).mean()   # raise probability of better-than-expected actions
    critic_loss = adv.pow(2).mean()                        # critic learns to predict the return
    loss = actor_loss + 0.5 * critic_loss - beta * torch.stack(ents).mean()   # entropy bonus keeps it exploring
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 500 == 0: print(f"update {it + 1}: mean return of last 20 episodes {np.mean(finished[-20:]):6.1f}")
# Expected output:
# update 500: mean return of last 20 episodes   35.4
# update 1000: mean return of last 20 episodes   81.3
# update 1500: mean return of last 20 episodes  145.3
# update 2000: mean return of last 20 episodes  218.6
```

Returns climb from ~35 to ~220 in 2,000 small updates, a few seconds of CPU. A one-step version of the same code (the course's choice) learned too, but much more noisily. Two Gymnasium 1.x details matter: vector environments **reset finished episodes automatically**, and the default mode does it on the *next* call — mixing a bogus transition into training. `AutoresetMode.SAME_STEP` avoids that.

## 4. The course's Kung Fu Master agent

| Piece | Course setting |
|---|---|
| Environment | `KungFuMasterDeterministic-v0`, 14 actions |
| Input | 4 stacked grayscale frames, 42×42 (stacking shows motion) |
| Network | 3 conv layers (32 channels, 3×3, stride 2) → dense 512 → 128 → actor (14) and critic (1) |
| Parallelism | 10 environments, batched |
| Optimiser | Adam, learning rate 1e-4 |
| Discount / reward scaling | γ = 0.99; rewards × 0.01 to keep losses small |
| Entropy weight | 0.001 |
| Advantage | one-step TD, critic loss weight 1 |
| Training | 3,000 iterations on CPU |

Average score over 10 evaluation episodes went from 440 (untrained) to around 1,000 (980, 1,040, 940 at successive checkpoints); a recorded episode scored about 1,500. The intuition lectures were recorded for an earlier version of the course (Breakout and Doom examples), but the ideas carry over.

## 5. PPO and SAC

### PPO: don't let one update change the policy too much
Policy-gradient updates can be too big: one unlucky batch shoves the policy somewhere bad and the agent never recovers. **PPO** (Proximal Policy Optimization) measures how much the new policy changed an action's probability, $\rho = \pi_{\text{new}}(a \mid s) / \pi_{\text{old}}(a \mid s)$, and clips the objective so moving $\rho$ outside $[1 - \epsilon, 1 + \epsilon]$ earns nothing more:

$$L = \min\big(\rho A,\ \text{clip}(\rho, 1 - \epsilon, 1 + \epsilon) A\big)$$

SAC, below, adds an entropy term to the value. Both in numbers:

```python
import numpy as np
eps = 0.2                                                  # PPO clip range
def ppo_objective(ratio, adv):                             # ratio = π_new(a|s) / π_old(a|s)
    return min(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
for ratio, adv in ((1.1, 2.0), (1.6, 2.0), (0.5, 2.0), (0.5, -2.0), (1.6, -2.0)):
    print(f"ratio {ratio:.1f}, advantage {adv:+.0f}: unclipped {ratio * adv:+.2f}  PPO {ppo_objective(ratio, adv):+.2f}")
p = np.array([0.7, 0.2, 0.1]); alpha = 0.2                 # SAC: reward plus a bonus for keeping options open
H = -(p * np.log(p)).sum()
print(f"policy entropy {H:.3f} nats -> SAC adds alpha*H = {alpha * H:.3f} to the value of this state")
# Expected output:
# ratio 1.1, advantage +2: unclipped +2.20  PPO +2.20
# ratio 1.6, advantage +2: unclipped +3.20  PPO +2.40
# ratio 0.5, advantage +2: unclipped +1.00  PPO +1.00
# ratio 0.5, advantage -2: unclipped -1.00  PPO -1.60
# ratio 1.6, advantage -2: unclipped -3.20  PPO -3.20
# policy entropy 0.802 nats -> SAC adds alpha*H = 0.160 to the value of this state
```

With a good action (A = +2), pushing its probability 60% higher only counts as 20% — no incentive to overshoot. With a bad action (A = −2), the objective never hides how bad a big move is (the min keeps the pessimistic value). PPO reuses each batch for several epochs of updates safely, which made it the default in game AI, robotics and the reinforcement-learning step of RLHF for language models.

### SAC: reward plus "keep your options open"
**Soft Actor-Critic** targets continuous actions (steering, throttle). It's off-policy — it learns from a replay buffer like DQN — and maximises reward **plus an entropy bonus** $\alpha H(\pi)$, so among equally good behaviours it prefers the most random one. That keeps exploring and makes it robust; it typically learns from fewer environment steps than PPO.

### The DeepRacer workshop
The course's Part 8 link trains both on **AWS DeepRacer** (a 1/18-scale autonomous car in a simulator): PPO with discrete and continuous steering, SAC with continuous. You write a reward function (e.g. stay near the centre line), pick an algorithm and train in the AWS console.

## 6. A real A3C in *Deep RL in Python* (Breakout)

Section 7 of *Advanced AI: Deep Reinforcement Learning in Python* implements genuinely asynchronous A3C in TF1 for `Breakout-v0`:

| Piece | Course setting (from its public repo, `rl2/a3c`) |
|---|---|
| Workers | one **thread** per CPU core, each with its own environment and a local copy of the networks |
| Networks | policy and value networks sharing a conv body: 16 filters 8×8 stride 4 → 32 4×4 stride 2 → dense 256; then policy softmax or one value |
| Rollouts | 5 steps per update; n-step returns bootstrapped from V(s) unless the episode ended |
| Policy loss | −(log π(a\|s) · advantage + 0.01 · entropy) |
| Optimiser | RMSProp (learning rate 2.5e-4, decay 0.99); each worker clips its gradients to global norm 5, applies them to the **global** network, then copies the global weights back |
| γ / budget | 0.99 / 5 million global steps, threads stopped by a TF `Coordinator` |

Compared with the AI A-Z version (§4): separate value and policy losses instead of one combined loss, n-step instead of one-step advantages, and true asynchrony. The transcript reports no scores — only that it learns in far less wall-clock time than DQN. Python threads share one interpreter lock, so the speed comes from TensorFlow releasing it during graph execution; with PyTorch today, batched A2C over vector environments (§3) is simpler and as good.

## Trade-offs

- **Actor-critic vs DQN.** Handles stochastic and continuous actions, learns from parallel envs vs typically less sample-efficient (on-policy) and more hyperparameters.
- **One-step vs n-step advantages.** Low variance, slow credit assignment vs faster learning, noisier targets.
- **Entropy weight.** More exploration and robustness vs slower convergence to a decisive policy.
- **A3C vs A2C.** Asynchronous CPU workers vs one batched update that suits GPUs; similar results.
- **PPO vs SAC.** Stable, simple, on-policy vs sample-efficient, continuous, off-policy with more moving parts.

## Common Pitfalls

- **Advantage without detaching.** The actor loss must treat A as a constant; otherwise the actor's gradient also trains the critic in odd ways.
- **No entropy bonus.** The policy collapses to one action early and stops exploring.
- **Unscaled rewards.** Large game scores blow up losses; scale or normalise (the course multiplies by 0.01).
- **Vector-env autoreset surprises.** Know your library's reset mode, and don't bootstrap across episode boundaries.
- **Single frames in games.** Stack frames (or use recurrence) so the state shows motion.
- **Calling batched A2C "A3C".** Fine in conversation; matters when you reproduce papers.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Discrete actions, cheap simulator | A2C / PPO with parallel envs | Single-env DQN | Varied experience, fast wall-clock |
| Continuous control, costly steps | SAC | PPO | Off-policy reuse of experience |
| Policy collapses early | Raise entropy weight | Lower learning rate only | Restores exploration |
| Updates sometimes wreck the policy | PPO clipping | Vanilla policy gradient | Bounded policy change |
| Partial observability (motion) | Frame stacking or LSTM | Single frames | Makes the state Markov |

## References

- [Mnih et al. — Asynchronous Methods for Deep RL (A3C, 2016)](https://arxiv.org/abs/1602.01783)
- [Schulman et al. — Proximal Policy Optimization (2017)](https://arxiv.org/abs/1707.06347) · [Generalized Advantage Estimation (2015)](https://arxiv.org/abs/1506.02438)
- [Haarnoja et al. — Soft Actor-Critic (2018)](https://arxiv.org/abs/1801.01290)
- [OpenAI Spinning Up — key concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) · [Sutton & Barto, 2nd ed., ch. 13 (policy gradients)](http://incompleteideas.net/book/the-book-2nd.html)
- [ALE Kung Fu Master](https://ale.farama.org/environments/kung_fu_master/) · [AWS DeepRacer](https://aws.amazon.com/deepracer/)

## See Also

- [policy-gradients](policy-gradients.md) — REINFORCE, baselines and Gaussian policies: where the actor loss comes from.
- [deep-q-learning](deep-q-learning.md) — value-based deep RL, replay buffers and target networks.
- [q-learning](q-learning.md) — Bellman, TD and the discount factor.
- [qlora-fine-tuning](../llm-engineering/qlora-fine-tuning.md) — fine-tuning LLMs; RLHF adds a PPO-style step on top.
- [convolutional-neural-networks](../deep-learning/convolutional-neural-networks.md) — the conv layers reading the game screen.
