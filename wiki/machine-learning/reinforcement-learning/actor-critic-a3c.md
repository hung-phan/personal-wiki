---
name: "actor-critic-a3c"
description: "Actor-critic reinforcement learning and A3C explained with runnable PyTorch code — policy (actor) and value (critic) heads on one shared network, the advantage A = r + gamma·V(s') − V(s), policy-gradient loss −log π(a|s)·A, critic loss, entropy bonus, n-step returns, asynchronous (A3C) vs synchronous batched (A2C) workers, Gymnasium vector environments and autoreset; the course's Kung Fu Master agent (4 stacked 42x42 grayscale frames, 3 conv layers, actor 14 actions, critic 1 value, 10 environments, Adam 1e-4, entropy 0.001, reward scaling 0.01); an A2C that learns CartPole in seconds; PPO's clipped objective and SAC's entropy-regularised value with worked numbers, and the AWS DeepRacer workshop; the asynchronous TF1 A3C for Breakout from Deep RL in Python (threads per CPU, 5-step rollouts, entropy 0.01, RMSProp, gradient clipping at norm 5). Study guide for AI A-Z Parts 7–8 (sections 14–17) and Deep Reinforcement Learning in Python section 7. Also covers the professional A2C of Cutting-Edge AI: Deep Reinforcement Learning in Python (synchronous workers needing no network copies, the Atari wrapper stack, n_steps 5, entropy 0.01, gradient clipping at global norm 0.5, and a Breakout agent scoring about 850)."
created: 2026-11-13
updated: 2026-11-13
tags: [cutting-edge-ai, reinforcement-learning, actor-critic, deep-rl-python, a3c, a2c, policy-gradient, ppo, sac, gymnasium, pytorch, study-guide, ai-a-z]
---
# Actor-Critic, A3C, PPO and SAC

> **How to read this note.** Study guide for *AI A-Z*, Part 7 (A3C, sections 15–16: playing Kung Fu Master) and Part 8 (PPO and SAC, section 17). Part 8 is a two-minute motivation video plus a link to an external one-hour AWS DeepRacer workshop that trains PPO and SAC for a self-driving car; that workshop isn't part of the course text, so §5 covers the two algorithms' core ideas on its own. Read [q-learning](q-learning.md) and [deep-q-learning](deep-q-learning.md) first; [policy-gradients](policy-gradients.md) derives the actor's loss. §6 adds the threaded A3C from *Deep Reinforcement Learning in Python* (section 7). The snippets need `torch` and `gymnasium`; the CartPole actor-critic trains in about 10 seconds.

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
    logps, values, ents, rewards, cont, tails = [], [], [], [], [], []
    for _ in range(STEPS):                                 # collect a short rollout from every env
        logits, v = net(torch.tensor(s)); dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
        s, r, term, trunc, info = envs.step(a.numpy())
        tail = torch.zeros(N)                              # time-limit cut: bootstrap from V(final_obs), not from the reset state s
        if trunc.any():
            with torch.no_grad(): tail[trunc] = net(torch.tensor(np.stack(info["final_obs"][trunc])))[1]
        logps.append(dist.log_prob(a)); values.append(v); ents.append(dist.entropy()); tails.append(tail)
        done = term | trunc; rewards.append(torch.tensor(r, dtype=torch.float32)); cont.append(torch.tensor(1.0 - done, dtype=torch.float32))
        ep_ret += r; finished += list(ep_ret[done]); ep_ret[done] = 0
    with torch.no_grad(): R = net(torch.tensor(s))[1]      # bootstrap from the critic after the last step
    returns = []
    for r, c, t in zip(reversed(rewards), reversed(cont), reversed(tails)):   # n-step return: r_t + γ r_t+1 + ... + γ^n V(s_t+n)
        R = r + gamma * (R * c + t); returns.insert(0, R)  # terminated: 0 future value; truncated: V(final_obs)
    adv = torch.stack(returns) - torch.stack(values)       # advantage: better or worse than the critic expected?
    actor_loss = -(torch.stack(logps) * adv.detach()).mean()   # raise probability of better-than-expected actions
    critic_loss = adv.pow(2).mean()                        # critic learns to predict the return
    loss = actor_loss + 0.5 * critic_loss - beta * torch.stack(ents).mean()   # entropy bonus keeps it exploring
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 500 == 0: print(f"update {it + 1}: mean return of last 20 episodes {np.mean(finished[-20:]):6.1f}")
# Expected output:
# update 500: mean return of last 20 episodes   35.4
# update 1000: mean return of last 20 episodes   81.3
# update 1500: mean return of last 20 episodes  132.2
# update 2000: mean return of last 20 episodes  154.9
```

Returns climb from ~35 to ~155 in 2,000 small updates, about 10 seconds of CPU. That is one seed: over seeds 0–4 the mean return at update 2,000 ranged from 124 to 210 (torch 2.15 CPU, gymnasium 1.4), and the curve is not monotone (seeds 1 and 4 dipped between updates 1,500 and 2,000). The course uses one-step advantages instead (§4). Three Gymnasium 1.x details matter: vector environments **reset finished episodes automatically**, and the default mode does it on the *next* call — mixing a bogus transition into training. `AutoresetMode.SAME_STEP` avoids that, but it means `s` after a finished step is already the *new* episode's first observation. So a **terminated** step (the pole fell) gets zero future value, while a **truncated** step (CartPole's 500-step time limit) is bootstrapped from $V$ of `info["final_obs"]`, the state the episode was cut at — the same terminated-vs-truncated rule as [deep-q-learning](deep-q-learning.md).

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
The course's Part 8 link trains both on **AWS DeepRacer** (a 1/18-scale autonomous car in a simulator): PPO with discrete and continuous steering, SAC with continuous. You write a reward function (e.g. stay near the centre line), pick an algorithm and train. Note: as of 15 December 2025 AWS DeepRacer is no longer available as a service in the AWS Console and the cars are no longer sold (AWS DeepRacer FAQ); to reproduce the workshop you deploy the open-source *DeepRacer on AWS* solution into your own AWS account.

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

## 7. Dissecting a professional A2C (*Cutting-Edge AI: Deep Reinforcement Learning in Python*)

Section 3 of *Cutting-Edge AI: Deep Reinforcement Learning in Python* takes the opposite route to §4 and §6: rather than writing an agent, it reads OpenAI Baselines' A2C line by line on Atari Breakout. What that buys is the list of details a from-scratch A2C quietly omits — and an honest look at how much of "the algorithm" is actually wrapper code.

**Synchrony removes the weight copies entirely.** §2 says A2C steps its environments together; this course makes the structural consequence explicit. In A3C (§6) each worker holds its *own* copy of the networks, computes gradients against those weights, ships the **gradients** to a master and gets fresh weights back — so every update is computed by parameters that have already moved on. A2C's workers are not learners at all: they are bare environments in child processes, and the single network in the parent process is the only weight set in existence. There is nothing to copy and nothing to go stale. The only variability left is that the environments are in different states — the ball falling left in one, right in another — so a batch is one policy evaluated across many partial episodes of the same game.

**"Synchronous" was a design choice, and an ambiguous one.** The course flags that the word could have meant at least two other things. Each worker could play a *whole episode* before the update, which would hand you exact Monte-Carlo returns instead of bootstrapped 5-step estimates — and would need different hyperparameters. Or each worker could take its own 5 steps at its own pace and then sync, which is numerically identical to what Baselines does. What Baselines actually does is lockstep every *individual* step: the parent sends one action per environment down a `multiprocessing` pipe, then blocks on all the replies (processes, not threads, because of the GIL). Worth knowing because "A2C" names the result, not the mechanism, and the per-step lockstep is the least parallel of the three readings. The encapsulation is good enough that the training loop reads as if there were one environment with vector-valued observations and actions.

**There is no A2C paper.** Only an OpenAI blog post amounting to "we made A3C synchronous" — and none of the implementation detail below appears in it. The method is citable through that post and as a baseline in Wu et al.'s ACKTR paper (NeurIPS 2017), but there is no primary source to check the recipe against. That absence is the course's argument for reading the code.

### The Atari recipe, as the course states it

| Piece | Stated setting |
|---|---|
| Game | Breakout, `BreakoutNoFrameskip-v4` (you do the frame-skipping yourself) |
| Rollout | `n_steps = 5` per environment per update; returns bootstrapped from $V$ of the final state unless it is terminal |
| Loss | `policy_loss + 0.5 * value_loss - 0.01 * entropy`, with `value_loss = mean((V - R)^2) / 2` |
| Gradients | clipped to **global norm 0.5**, applied with **RMSProp** (decay and epsilon not read out) |
| Observation | grayscale, resized to **84×84**, **4 frames stacked** (stacking is for memory: consecutive states share 3 of 4 frames) |
| Action repeat | each action held for **4 frames**, observation = element-wise **max over the last 2** |
| Episode starts | up to **30 random no-ops** on reset; FIRE pressed on reset; each **life lost** ends an episode |
| Reward | clipped to its sign: **−1 / 0 / +1** |
| Network | **3 conv layers → flatten → 1 dense layer → two linear heads** (policy logits and a scalar value, both with no activation), **orthogonal initialisation**, `uint8` input **divided by 255** |
| Parallel environments | **stated inconsistently — 16 in one lecture, 8 in another** |
| Learning rate, γ, total timesteps | **never stated** |

The three loss weights are the ones to remember: policy 1, value 0.5, entropy 0.01. The course points out a fourth that is easy to miss — the value loss already carries a factor of ½ *inside* it, so the effective weight on the squared error is 0.25, not 0.5. He calls it a hidden hyperparameter, and it is the kind of thing that makes a reimplementation train differently for no visible reason.

Two honesty notes on the loss. The course narrates the entropy term as *added*; as §1 says, it is added to the objective and therefore **subtracted** from a minimised loss — his own bonus exercise ("check the signs") is the tell. And the policy loss is built from `sparse_softmax_cross_entropy_with_logits`, which looks like a classification call but with integer labels computes exactly $-\log \pi(a \mid s)$, because the actions *are* integers from 0.

**Nothing else is normalised.** Pixels are divided by 255 and rewards are reduced to their sign. There is no running mean/std on observations, no reward scaling, no learning-rate schedule discussed, no replay buffer and no target network. If you have met `VecNormalize` or a `RunningMeanStd` in reinforcement-learning code, it belongs to the continuous-control settings, not here.

### Results

Training takes **"a bit less than a day" on the instructor's laptop GPU** (model not stated). The trained agent then plays faster than real time and gets about as far as Breakout allows: across its five lives it **clears every block twice** — the screen refills once after the first clear — for a score of roughly **850**.

The scores printed *during* training are far lower, for three compounding reasons. Rewards are clipped to ±1, so a brick worth 7 counts as 1. Four frames collapse into one step, so several real rewards inside those frames collapse into a single +1. And each **life** is an episode, so a printed "episode" reward is a fifth of a game. The course names the first two and builds its own `Monitor` wrapper to recover the true per-game score, but never connects it to the third. No loss values, learning curves or total-timestep counts are given anywhere, so the ~850 is not reproducible from the lectures alone — you do not even know the batch size.

For the arithmetic itself, §3's snippet is this algorithm: `N` environments stepped together, 5-step rollouts, `returns - values` for the advantage, and `actor + 0.5 * critic - beta * entropy` summed into one scalar. The structural differences are the conv trunk, the extra ½ inside the value loss, and RMSProp in place of Adam.

### Reading this code in 2026

- **It is TensorFlow 1 throughout** — `tf.placeholder`, `tf.Session`, `variable_scope(reuse=...)`, `tf.gradients` plus a manual `apply_gradients`. None of it runs on TF2 without `tf.compat.v1` and `disable_eager_execution()`. In particular, the implementation builds **two network instances** — a step-model with `nsteps=1` for acting and a train-model with `nsteps=5` for learning, sharing one weight set via `reuse=True`. That split is an artifact of placeholders having a **fixed batch dimension**, not anything A2C requires: in TF2 or PyTorch you call one module with whichever batch size you have, exactly as §3 does.
- **OpenAI Baselines is archived.** Stable-Baselines3 and CleanRL are the live equivalents, and SB3's `AtariWrapper` bundles no-op reset, max-and-skip, episodic life, fire reset, frame warping and reward clipping — an entire lecture's worth of wrapper code is now one constructor argument. The pedagogy survives; the code should not be copied.
- **`ALE/Breakout-v5` now defaults to sticky actions ON** (`repeat_action_probability = 0.25`) with `frameskip = 4` — the opposite of the deterministic `-v4` behaviour this code assumes and re-implements by hand. Set it deliberately or you have silently changed the problem. Atari ROMs are also no longer bundled; you need `ale-py` with ROMs installed.
- **Gymnasium's `step()` returns five values** (`obs, reward, terminated, truncated, info`), and every wrapper here assumes a single `done`. This is not cosmetic: the n-step return multiplies future value by `(1 - done)`, and with the split you must **bootstrap on truncation but not on termination**. Collapsing the two tells the critic that a state cut off by a time limit is worth nothing — a real correctness bug if the code is ported naively. Same rule as §3's `info["final_obs"]` handling.
- **A2C is no longer the choice.** PPO (§5) superseded it almost immediately with the same skeleton plus a clipped surrogate, and IMPALA/R2D2 superseded both for Atari throughput. A day of laptop GPU for one Breakout agent is weak by current standards. The reason to read this section is the hidden details, not the algorithm.

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
- **Treating time-limit truncation as termination.** Zeroing the bootstrap at a time limit tells the critic the state is worth nothing when the episode was merely cut off; bootstrap truncated steps from $V$ of the final observation (with `SAME_STEP` autoreset, `info["final_obs"]`).
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
- [ALE Kung Fu Master](https://ale.farama.org/environments/kung_fu_master/) · [AWS DeepRacer FAQ (service ended 2025-12-15)](https://aws.amazon.com/deepracer/faqs/)

## See Also

- [policy-gradients](policy-gradients.md) — REINFORCE, baselines and Gaussian policies: where the actor loss comes from.
- [deep-q-learning](deep-q-learning.md) — value-based deep RL, replay buffers and target networks.
- [q-learning](q-learning.md) — Bellman, TD and the discount factor.
- [qlora-fine-tuning](../llm-engineering/qlora-fine-tuning.md) — fine-tuning LLMs; RLHF adds a PPO-style step on top.
- [convolutional-neural-networks](../deep-learning/convolutional-neural-networks.md) — the conv layers reading the game screen.
