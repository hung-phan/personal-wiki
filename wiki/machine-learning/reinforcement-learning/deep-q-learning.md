---
name: deep-q-learning
description: Deep Q-learning (DQN) and deep convolutional Q-learning explained with runnable PyTorch code — a neural network that maps a state to one Q-value per action, the target r + gamma · max Q(s') and squared or Huber loss on the action taken, experience replay to break correlation between consecutive steps, a target network with soft updates (tau), epsilon-greedy vs softmax action selection, terminated vs truncated episodes; the course's LunarLander agent (8 inputs, 64-64 hidden, Adam 5e-4, buffer 100k, minibatch 100, gamma 0.99, tau 1e-3, solved at average 200) and Ms. Pac-Man agent (128x128 RGB frames, four conv + batchnorm blocks flattening to 12,800, 9 actions); a DQN that learns CartPole in seconds with a random-policy baseline; eligibility traces / n-step returns, double DQN and prioritized replay. Breakout DQN from Deep RL in Python (84x84 grayscale, 4 stacked frames, Nature conv stack, 500k-frame circular replay buffer, hard target copy every 10k steps), recurrent Q-networks for partially observable MDPs. Study guide for AI A-Z Parts 5–6 (sections 8–13) and Deep Reinforcement Learning in Python section 6.
created: 2026-11-13
updated: 2026-11-13
tags: [reinforcement-learning, dqn, deep-rl-python, deep-q-learning, experience-replay, target-network, gymnasium, pytorch, atari, study-guide, ai-a-z]
---
# Deep Q-Learning: From a Table to a Network

> **How to read this note.** Study guide for *AI A-Z*, Part 5 (deep Q-learning, sections 9–10: landing a lunar module) and Part 6 (deep convolutional Q-learning, sections 12–13: playing Ms. Pac-Man from pixels). Read [q-learning](q-learning.md) first. §8 adds the Breakout DQN from *Deep Reinforcement Learning in Python* (section 6). The course trains LunarLander for ~1,000 episodes and Ms. Pac-Man for ~2 hours on a GPU; here the same algorithm learns **CartPole** in about 15 seconds on a CPU, and the course's networks and settings are reproduced and checked. Install: `VIRTUAL_ENV=/Users/phahng/Desktop/workarea/.venv uv pip install torch "gymnasium[box2d]" pillow`.

## Why This Exists

**Problem.** A Q-table needs one row per state. A lunar lander's state is 8 real numbers; a game screen is 100,000 pixels. No table can hold that, and states the agent has never seen exactly would get no value at all.

**Key insight.** Replace the table with a **neural network** that takes the state and outputs a Q-value for every action, trained so that $Q(s, a)$ moves toward $r + \gamma \max_{a'} Q(s', a')$. Two tricks make that stable: **experience replay** (train on random past steps, not the latest ones in order) and a **target network** (compute the targets with a slowly updated copy).

**Reach for this when:**
- Working through AI A-Z Parts 5–6, or any DQN tutorial.
- The state is a vector or an image and the actions are a small discrete set.
- A DQN trains unstably and you need to know which part to check.

**Don't reach for this when:**
- Actions are continuous (steering angle, motor torque) — taking a max over infinitely many actions is the problem; use actor-critic methods such as PPO or SAC ([actor-critic-a3c](actor-critic-a3c.md)).
- The state space is tiny — tabular Q-learning is simpler and exact enough ([q-learning](q-learning.md)).
- You want sample-efficient training on a real robot — model-based or offline methods fit better.

## Diagrams

![DQN loop: the environment's state goes into the online Q-network, epsilon-greedy picks an action, each step is stored in a replay buffer, random minibatches are trained against targets r + gamma max Q_target(s') from a slow target network that follows the online network by soft updates](assets/dqn-loop.svg)

## 1. From Q-table to Q-network

**The question.** How do you generalise Q-values to states you've never seen?

Let a network do it. Input: the state (8 numbers for the lander). Output: one number per action (4 for the lander: do nothing, fire left, main or right engine). Pick the action with the largest output. Training: for each experienced step $(s, a, r, s')$, the Bellman equation says what $Q(s, a)$ *should* be,

$$y = r + \gamma \max_{a'} Q(s', a') \quad (\text{or just } y = r \text{ if the episode ended}),$$

so minimise $(Q(s, a) - y)^2$ by backprop — **only for the action actually taken**; the other outputs get no target from this step. (The course's loss slide sums over all four outputs; the code correctly uses only the taken action via `gather`.)

## 2. The two stabilisers

**Experience replay.** Consecutive steps are nearly identical (the lander moved a fraction of a pixel). Training on them in order is like studying one page of a textbook a hundred times in a row: the network overfits to the current situation and forgets the rest. Store every step in a big buffer (the course: 100,000) and train on **random minibatches** of past steps. It also reuses rare experiences — a near-crash — many times.

**Target network.** If the same network produces both $Q(s, a)$ and the target $y$, every update moves the target too: a dog chasing its tail. Keep a second copy for computing $y$ and let it follow slowly. The course uses a **soft update** after each learning step,

$$\theta_{\text{target}} \leftarrow \tau\, \theta_{\text{online}} + (1 - \tau)\, \theta_{\text{target}}, \quad \tau = 0.001,$$

the original DQN copied the weights every few thousand steps instead. (The course's intuition lectures don't mention the target network; the code has one.)

## 3. Choosing actions: explore vs exploit

Early on the Q-network knows nothing, so the agent must explore. Two common rules:
- **ε-greedy**: random action with probability ε, otherwise the best. ε starts at 1 and decays (course: ×0.995 per episode down to 0.01).
- **Softmax** (Boltzmann): pick actions with probability $\propto e^{Q/T}$, so near-best actions are tried often and bad ones rarely; temperature $T$ controls how greedy.

```python
import numpy as np
rng = np.random.default_rng(0)
q = np.array([1.0, 1.4, 0.9, 1.3])                        # Q-values for 4 actions in some state
def eps_greedy(q, eps): return rng.integers(len(q)) if rng.random() < eps else int(q.argmax())
def softmax_pick(q, T):                                   # better actions more likely, never certain
    p = np.exp(q / T); p /= p.sum(); return rng.choice(len(q), p=p)
for name, f in (("ε-greedy ε=0.1", lambda: eps_greedy(q, 0.1)), ("softmax T=0.1", lambda: softmax_pick(q, 0.1)),
                ("softmax T=1.0", lambda: softmax_pick(q, 1.0))):
    picks = np.bincount([f() for _ in range(10_000)], minlength=4) / 10_000
    print(f"{name:15s} action shares {np.round(picks, 3)}")
# Expected output:
# ε-greedy ε=0.1  action shares [0.027 0.922 0.025 0.026]
# softmax T=0.1   action shares [0.015 0.714 0.005 0.266]
# softmax T=1.0   action shares [0.212 0.323 0.183 0.283]
```

ε-greedy treats the second-best action (1.3) like the worst (0.9); softmax at T = 0.1 tries it a quarter of the time. The course's theory lecture favours softmax; its code uses ε-greedy.

## 4. A complete DQN, small enough to run

CartPole: balance a pole on a cart by pushing left or right; +1 per step, up to 500. Same algorithm as the course's lander — online and target networks, replay buffer, minibatches, soft updates, ε decay — with a Huber (smooth L1) loss, which is less thrown by occasional huge errors than squared error.

```python
import random, collections, numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.manual_seed(0); random.seed(0); np.random.seed(0); torch.set_num_threads(1)   # repeatable on one machine + library versions only
env = gym.make("CartPole-v1"); env.action_space.seed(0)   # 4 numbers in, 2 actions; seed the action sampler too
def net(): return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
q, target = net(), net(); target.load_state_dict(q.state_dict())   # online network + slow-moving target copy
opt = torch.optim.Adam(q.parameters(), lr=5e-4)
memory = collections.deque(maxlen=50_000)                  # experience replay buffer
gamma, tau, eps, scores = 0.99, 0.005, 1.0, []
for episode in range(300):
    s, _ = env.reset(seed=episode); total, done = 0.0, False
    while not done:
        a = env.action_space.sample() if random.random() < eps else int(q(torch.tensor(s)).argmax())
        s2, r, term, trunc, _ = env.step(a); done = term or trunc
        memory.append((s, a, r, s2, term)); s = s2; total += r
        if len(memory) >= 1000:
            S, A, Rw, S2, T = map(np.array, zip(*random.sample(memory, 64)))   # random minibatch breaks correlation
            S, S2 = torch.tensor(S), torch.tensor(S2)
            with torch.no_grad(): y = torch.tensor(Rw, dtype=torch.float32) + gamma * target(S2).max(1).values * torch.tensor(1 - T, dtype=torch.float32)
            pred = q(S).gather(1, torch.tensor(A)[:, None]).squeeze(1)       # Q(s, a) for the actions taken
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            for tp, p in zip(target.parameters(), q.parameters()): tp.data.mul_(1 - tau).add_(tau * p.data)   # soft update
    eps = max(0.01, eps * 0.98); scores.append(total)
    if (episode + 1) % 50 == 0: print(f"episodes {episode - 48:3d}-{episode + 1:3d}: mean score {np.mean(scores[-50:]):6.1f}")
def random_episode(seed):                                  # baseline: push left/right at random
    env.reset(seed=seed); total, done = 0, False
    while not done:
        _, r, term, trunc, _ = env.step(env.action_space.sample()); total += r; done = term or trunc
    return total
print("random policy mean score:", np.mean([random_episode(1000 + i) for i in range(50)]))
# Expected output:
# episodes   1- 50: mean score   17.0
# episodes  51-100: mean score   10.8
# episodes 101-150: mean score   11.9
# episodes 151-200: mean score   87.1
# episodes 201-250: mean score  102.9
# episodes 251-300: mean score  124.9
# random policy mean score: 24.64
```

Random pushing keeps the pole up for ~21–25 steps; after 300 episodes this run averages 125 over its last 50. That number is one seed: changing all seeds to 1–4 gave 108–137 (torch 2.15 CPU, Gymnasium 1.4), and another seeding scheme reached ~190 — either way, far above random. Your exact numbers can differ from these on another platform or library version even with every seed fixed. The first ~100 episodes show nothing — the buffer is filling and ε is high — which is normal. Two details that cause silent bugs: Gymnasium returns **`terminated`** (the pole fell: no future reward) and **`truncated`** (time limit hit: the future still exists); only `terminated` should zero the bootstrap term. And `env.action_space` has its *own* random generator — seed it, or runs won't repeat.

## 5. The course's lunar lander

The environment: 8 state numbers (position, velocity, angle, angular velocity, two leg-contact flags) and 4 actions. Shaped rewards for approaching the pad slowly and upright, +10 per leg touching down, small costs for firing engines, −100 for a crash, +100 for landing. **Solved = average score ≥ 200 over 100 episodes.**

```python
import numpy as np, gymnasium as gym
env = gym.make("LunarLander-v3"); env.action_space.seed(0)   # the course used v2 (gone in Gymnasium 1.x)
print("state:", env.observation_space.shape, "| actions:", env.action_space.n)   # 8 numbers, 4 engines-or-nothing
scores = []
for ep in range(20):
    s, _ = env.reset(seed=ep); total, done = 0.0, False
    while not done:
        s, r, term, trunc, _ = env.step(env.action_space.sample()); total += r; done = term or trunc
    scores.append(total)
print(f"random policy: mean score {np.mean(scores):.0f} (solved = average 200 over 100 episodes)")
# Expected output:
# state: (8,) | actions: 4
# random policy: mean score -227 (solved = average 200 over 100 episodes)
```

The course's setup:

| Setting | Value |
|---|---|
| Network | 8 → 64 → 64 → 4, ReLU |
| Optimiser | Adam, learning rate 5e-4 |
| Minibatch / buffer | 100 / 100,000 |
| γ / τ | 0.99 / 0.001 (soft update after each learning step) |
| Learn every | 4 environment steps |
| ε | 1.0 → 0.01, ×0.995 per episode |
| Budget | up to 2,000 episodes × 1,000 steps |

It reached an average of 201.6 at episode 963 and saved the weights; a final video shows the trained agent landing. The code was written for `LunarLander-v2`; Gymnasium 1.x only has `v3`.

## 6. From numbers to pixels: deep convolutional Q-learning

**The question.** Ms. Pac-Man gives you the screen, not a tidy vector. What changes?

Only the front of the network. Put a CNN ([convolutional-neural-networks](../deep-learning/convolutional-neural-networks.md)) before the fully connected layers so it can learn walls, dots and ghosts from pixels. The course's network for `MsPacmanDeterministic-v0` (9 actions), on frames resized from 210×160 to 128×128 RGB and scaled to [0, 1]:

```python
import torch, torch.nn as nn, numpy as np
from PIL import Image
frame = np.random.default_rng(0).integers(0, 256, (210, 160, 3), dtype=np.uint8)   # an Atari frame: 210 x 160 RGB
x = torch.tensor(np.array(Image.fromarray(frame).resize((128, 128))), dtype=torch.float32).permute(2, 0, 1)[None] / 255
def block(i, o, k, s): return [nn.Conv2d(i, o, k, s), nn.BatchNorm2d(o), nn.ReLU()]
net = nn.Sequential(*block(3, 32, 8, 4), *block(32, 64, 4, 2), *block(64, 64, 3, 1), *block(64, 128, 3, 1),
                    nn.Flatten(), nn.Linear(12800, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 9))
h = x
for layer in net[:12]:
    h = layer(h)
    if isinstance(layer, nn.ReLU): print("conv block ->", tuple(h.shape[1:]))
print("flattened:", h.flatten(1).shape[1], "| Q-values for 9 actions:", tuple(net(x).shape), "| params: %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
# Expected output:
# conv block -> (32, 31, 31)
# conv block -> (64, 14, 14)
# conv block -> (64, 12, 12)
# conv block -> (128, 10, 10)
# flattened: 12800 | Q-values for 9 actions: (1, 9) | params: 6.8M
```

Four conv blocks (each conv → batch norm → ReLU) shrink 128×128 to 10×10×128 = 12,800 features, then 512 → 256 → 9 Q-values. Training changes from the lander: minibatch 64, a 10,000-step buffer, learning every step, and a self-chosen "solved" bar of 500 points. About 2 hours on a Colab V100 reached an average of 508.6 by episode 733; a demo episode scored 790.

**One frame isn't enough to see motion.** A single screenshot doesn't show which way a ghost is moving. Standard Atari DQNs stack the last 4 grayscale frames as the input; the course feeds single RGB frames. "Deterministic" in the environment name means a fixed frame-skip of 4 (each action repeats for 4 frames), not scripted ghosts.

**Eligibility traces / n-step returns.** The course's intuition lectures describe learning from several steps of reward at once instead of one — $y = r_t + \gamma r_{t+1} + \dots + \gamma^{n} \max Q(s_{t+n})$ — which propagates reward faster. They aren't implemented in the course code; [actor-critic-a3c](actor-critic-a3c.md) uses the same idea for advantages.

## 7. Improvements worth knowing

- **Double DQN** — pick the next action with the online network but evaluate it with the target network; reduces the systematic over-estimation that $\max$ causes.
- **Prioritised experience replay** — sample surprising transitions (large TD error) more often (cited by the course as reading).
- **Dueling networks, noisy nets, distributional heads** — combined in "Rainbow" DQN.
- **Adaptive ε** — the course also cites value-difference-based exploration (Tokic, 2010).

## 8. The same algorithm in *Deep RL in Python* (Breakout)

Section 6 of *Advanced AI: Deep Reinforcement Learning in Python* builds a DQN for Atari Breakout in TF1 and Theano. It follows the 2015 Nature recipe more closely than the AI A-Z version:

| Piece | Course setting (from its public repo, `rl2/atari`) |
|---|---|
| Preprocessing | grayscale, crop the play area, resize to 84×84; stack the last **4 frames** as the state |
| Actions | 4 (the environment reports 6; two duplicates are dropped) |
| Network | conv 32 filters 8×8 stride 4 → 64 4×4 stride 2 → 64 3×3 stride 1 → dense 512 → 4 Q-values |
| Loss / optimiser | Huber loss (TF) or squared error (Theano); Adam, learning rate 1e-5 |
| Replay buffer | 500,000 steps, pre-filled with 50,000 steps of random play |
| Target network | **hard copy** every 10,000 steps (not a soft τ update) |
| Minibatch / γ | 32 / 0.99 |
| ε | 1.0 → 0.1, linearly over 500,000 steps |
| Budget | 3,500 episodes |

**First on CartPole.** Before Atari, the section builds the same DQN for CartPole (an MLP, a replay buffer kept as lists, a target network copied every `copy_period` steps) — the setup §4 reproduces. The instructor also prices the Atari run: days to a week of GPU time per training run, so a hyperparameter sweep is out of reach for most students; his advice is to run on CPU, look for *some* improvement, and focus on the ideas.

**Memory-efficient replay.** Storing 4-frame states would hold every frame four times. The course stores **single frames** in a pre-allocated circular array and rebuilds a state from frames t−3…t when sampling, rejecting samples whose four frames straddle an episode boundary or the write pointer. Standard Atari wrappers (frame skipping is built into the env; reward clipping and treating a lost life as terminal) aren't used. The transcript reports no scores, only that the agent learns to play well; the code uses `Breakout-v0`, `scipy.misc.imresize` and TF1 sessions, all gone — `ale-py`'s `ALE/Breakout-v5` with Gymnasium's `AtariPreprocessing` and `FrameStackObservation` wrappers is the modern equivalent.

**Partially observable MDPs.** When one observation shows only part of the state, stacking frames is a fixed-length fix; a recurrent Q-network (DRQN) is the general one. The catch the course points out: random replay sampling destroys the order an RNN needs, so either store fixed-length sequences or sample an *end* index and replay the k frames before it (tracking episode boundaries).

## Trade-offs

- **Buffer size.** Larger = more diverse, less correlated batches vs more memory and staler data.
- **Soft (τ) vs hard target updates.** Smooth targets vs simpler, the original recipe; both work if tuned.
- **ε decay speed.** Faster exploitation vs risk of locking into a poor strategy.
- **MSE vs Huber loss.** Standard vs robust to the large TD errors early in training.
- **Pixels vs engineered state.** General, no feature work vs far more compute and data.

## Common Pitfalls

- **Training on consecutive steps.** Without replay, learning is unstable; sample at random.
- **No target network.** Targets move with every update; training oscillates or diverges.
- **Bootstrapping through a time-limit truncation.** Only true terminations should zero the future value.
- **Unseeded action sampling.** `env.action_space.sample()` uses its own RNG; seed it for repeatable runs — and even then, results only repeat on the same platform and library versions; compare across several seeds, not one run.
- **Judging too early.** DQN often looks flat for the first hundreds of episodes.
- **One frame for a moving world.** Stack frames (or add velocity) so the state is Markov.
- **Version drift.** `LunarLander-v2`, the old Gym API and `Deterministic-v0` Atari ids need updating for Gymnasium 1.x / ALE.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Vector state, few discrete actions | DQN (MLP) | Q-table | Generalises across states |
| Image state, discrete actions | Conv DQN with stacked frames | MLP on raw pixels | Learns visual features, sees motion |
| Q-values overestimated, unstable | Double DQN, Huber loss | Bigger network | Targets the bias directly |
| Rare important events | Prioritised replay | Uniform replay | Learns more from surprises |
| Continuous actions | PPO / SAC / DDPG | DQN | max over actions is intractable |

## References

- [Mnih et al. — Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602) · [Human-level control through deep RL (Nature, 2015)](https://www.nature.com/articles/nature14236)
- [van Hasselt et al. — Double DQN (2015)](https://arxiv.org/abs/1509.06461) · [Schaul et al. — Prioritized Experience Replay (2015)](https://arxiv.org/abs/1511.05952)
- [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) · [ALE Ms. Pac-Man](https://ale.farama.org/environments/ms_pacman/)
- [PyTorch — Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) — fixed seeds don't guarantee identical results across platforms or releases.
- [Sutton & Barto, 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html) — ch. 7 (n-step) and ch. 12 (eligibility traces).

## See Also

- [q-learning](q-learning.md) — the Bellman equation and TD update this approximates.
- [actor-critic-a3c](actor-critic-a3c.md) — learning a policy directly, with parallel environments.
- [convolutional-neural-networks](../deep-learning/convolutional-neural-networks.md) — the conv blocks in front of the Pac-Man Q-network.
- [artificial-neural-networks](../deep-learning/artificial-neural-networks.md) — the MLP, Adam and backprop under the hood.
