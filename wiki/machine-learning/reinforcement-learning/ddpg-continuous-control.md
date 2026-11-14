---
name: "ddpg-continuous-control"
description: "DDPG (Deep Deterministic Policy Gradient) for continuous control explained with runnable PyTorch code — why a Q-network with one output per action cannot handle a continuous action space, why discretising actions is a bad fix (3 bins per joint on a 7-joint arm is 3^7 = 2187 actions), the deterministic policy mu(s) as a network trained to output argmax_a Q(s, a), the policy objective J = E[Q(s, mu(s))] and its actor loss -Q(s, mu(s)), the critic taking state AND action as input, the generator/discriminator analogy for freezing the critic while updating the actor, Gaussian exploration noise instead of unnamed Ornstein-Uhlenbeck correlated noise, soft target updates and the tau = 0.999 retention-vs-literature 0.001 convention trap, a 60-episode DDPG that swings up Pendulum-v1 in about 11 seconds, and why TD3 and SAC replaced it. Study guide for Cutting-Edge AI: Deep Reinforcement Learning in Python section 4."
created: 2026-11-13
updated: 2026-11-13
tags: [reinforcement-learning, ddpg, continuous-control, deterministic-policy-gradient, off-policy, actor-critic, pendulum, mujoco, td3, sac, pytorch, study-guide]
---
# DDPG: Deterministic Policy Gradients for Continuous Control

> **How to read this note.** Study guide for *Cutting-Edge AI: Deep Reinforcement Learning in Python* (Lazy Programmer), section 4 (DDPG). DDPG is DQN plus a policy network, so read [deep-q-learning](deep-q-learning.md) first — replay buffers, target networks, soft updates and ε-greedy are all explained there and this note does not repeat them. [policy-gradients](policy-gradients.md) covers the *stochastic* Gaussian policy that DDPG replaces with a deterministic one; [actor-critic-a3c](actor-critic-a3c.md) covers the actor-critic family DDPG belongs to, including PPO and SAC in its §5; [q-learning](q-learning.md) has the Bellman equation. The course's section 5 (evolution strategies) is a separate, independent topic and has its own note. The snippet needs `torch` and `gymnasium` and trains Pendulum in about 11 seconds. The shared Lazy Programmer appendix is covered in [rl-foundations-mc-td](rl-foundations-mc-td.md) §5.

## Why This Exists

**Problem.** Every algorithm up to this point either picks from a short menu of discrete actions (DQN: one output node per action) or samples from a distribution it has to reason about in log-probability space (REINFORCE, A2C). But a robot joint takes a torque, not a menu item. If the action is a real number — or a vector of eight real numbers — "one output per action" is infinite output nodes, and `argmax_a Q(s, a)` is an optimisation problem you would have to solve from scratch at every environment step.

**Key insight.** Train a second network to *output the argmax*. A deterministic policy $\mu(s)$ is a learned shortcut for "the best action here", so acting is one forward pass, and the critic $Q(s, a)$ can then safely take the action as an *input* instead of enumerating actions as outputs.

**Reach for this when:**
- Working through section 4 of *Cutting-Edge AI: Deep RL in Python*.
- You need the simplest algorithm that makes the continuous-action case click — DDPG is the bridge from DQN to TD3 and SAC.
- You are reading TD3 or SAC and need the parent algorithm they both patch.
- Reviewing code that mixes an actor, a critic, two target networks and a soft update, and you want to know which line does what.

**Don't reach for this when:**
- The action space is discrete — DDPG cannot do discrete actions at all; use [deep-q-learning](deep-q-learning.md).
- You are starting a real continuous-control project in 2026 — use SAC or TD3 (§8). DDPG's value now is pedagogical.
- You need a stochastic policy (for entropy-driven exploration, or because the task has no single best action) — see [policy-gradients](policy-gradients.md) and SAC.

## Diagrams

![DDPG as a composition of two networks: state into the actor mu, its action into the critic Q together with the state, producing the scalar Q(s, mu(s)); the critic update freezes the target networks, the actor update freezes the critic; next to it the GAN generator/discriminator analogy; and the soft target update as a slow exponential blend](assets/ddpg.svg)

<!-- DIAGRAM SPEC: name=ddpg | panels: (1) "The composition" - left-to-right chain: box "s" -> box "actor mu(s)" -> box "a = mu(s)" -> box "critic Q(s,a)" -> box "scalar Q(s, mu(s))", with a second arrow from "s" skipping forward directly into the critic box (the critic sees BOTH s and a). Below the chain two bars labelled "critic update: minimise (Q(s,a) - y)^2, y from target nets (frozen)" and "actor update: maximise Q(s, mu(s)), critic weights FROZEN (snowflake icon on the critic box)". (2) "Same shape as a GAN" - chain: box "z" -> box "generator G" -> box "fake sample" -> box "discriminator D" -> "real/fake score", with a snowflake on D and caption "train G by maximising D(G(z)) with D frozen -> train mu by maximising Q(s, mu(s)) with Q frozen". (3) "Soft vs hard target update" - small plot, x axis "training steps", y axis "target weight": a staircase line labelled "DQN: hard copy every N steps" and a smooth curve lagging below an upper dashed line labelled "online weights", the smooth curve labelled "DDPG: theta_targ <- tau*theta + (1-tau)*theta_targ, tau = 0.005". | labels: as quoted above; use "->" for arrows, no superscripts | the one idea it must land: DDPG is two networks glued into one differentiable chain, and each update freezes the other end of the chain. -->

## 1. Why a continuous action space breaks DQN

DQN's $Q$ network is a function of the state alone: feed in $s$, get $|A|$ numbers out, one per action. Acting is `argmax` over the output vector; the Bellman target is `max` over the output vector. Both operations are a loop over output nodes.

A continuous action space has infinitely many actions, so this architecture needs infinitely many output nodes. Two obvious fixes, both of which the course rejects:

**Fix 1 — discretise (binning).** Chop the action range into bins: $[-1, 0) \to$ action 0, $[0, 1) \to$ action 1, and so on. Two problems.

- *It throws away the geometry of the action space.* Action $0.9$ and action $1.1$ should do almost the same thing, and both should differ from $-0.9$. Separate output nodes know nothing about this: to the network, neighbouring bins are as unrelated as any other pair. Every output head has to learn its own value from scratch, with no sharing.
- *Curse of dimensionality.* The actions multiply across dimensions. The course's example: a robot arm with **7 degrees of freedom** and an absurdly coarse 3-way grid per joint (rotate one way / rotate the other / do nothing) already gives $3^7 = 2187$ output nodes — and that grid is far too coarse to control anything smoothly. Refine it to 10 bins per joint and you are at $10^7$.

**Fix 2 — feed the action in as an input.** Write the critic the textbook way, $Q(s, a) \to$ one scalar, with $a$ as part of the input. The architecture is fine; *acting* is the problem. Choosing an action now means $\arg\max_a Q(s, a)$ where $a$ is a network **input** — a continuous optimisation over the input space, to be solved freshly at every single environment step. Nobody can afford that in a control loop.

The nice twist: DDPG is fix 2, made to work.

## 2. The deterministic policy: a network that outputs the argmax

Add a second network, the **actor** $\mu(s)$, which returns an action directly — a scalar for Pendulum, a vector for a robot. Conceptually it is defined as

$$\mu(s) = \arg\max_a Q(s, a),$$

but instead of *solving* that optimisation at each step we **train a network to output the answer**. After training, acting is one forward pass, and fix 2's architecture becomes usable: the critic takes the action as an input, and $\mu$ supplies the maximising action.

**The objective.** We want future reward, and $Q$ *is* predicted future reward, so the policy's objective is simply to make the critic's number big:

$$J(\theta_\mu) = \mathbb{E}\big[\, Q(s, \mu(s)) \,\big], \qquad \text{loss} = -\frac{1}{B}\sum_{\text{batch}} Q(s, \mu(s)).$$

That is the whole actor loss: minus the critic's output. The expectation becomes a sample mean over a replay minibatch.

**Why there is no log-probability.** In REINFORCE and A2C the objective depends on the policy only through the *probability* of a sampled action, which is why you need the score-function trick, $\nabla \mathbb{E}[\cdot] = \mathbb{E}[\nabla \log \pi \cdot G]$. Here $J$ depends on $\mu$ **directly** — the action is the network's output, not a sample from it — so plain backpropagation through the composition $Q(s, \mu(s))$ is all you need. Writing out the chain rule:

$$\nabla_{\theta_\mu} J = \underbrace{\nabla_a Q(s, a)\big|_{a = \mu(s)}}_{\text{which way to move the action}} \cdot \underbrace{\nabla_{\theta_\mu} \mu(s)}_{\text{how to move the weights to do that}}$$

The left factor is the gradient of the critic *with respect to its own input* — a quantity DQN never needs and never computes. In code autodiff does both factors; you never type this formula. This is the **deterministic policy gradient** of Silver et al. (2014), which DDPG puts a deep network on.

**Deterministic vs Gaussian.** [policy-gradients](policy-gradients.md) §3 handles continuous actions by having the network emit a mean and a standard deviation and sampling $a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$. DDPG's $\mu$ is **not** a distribution mean — the course is emphatic that $\mu$ is just a symbol for the deterministic policy, nothing to do with the usual statistical $\mu$. One number comes out and that is the action. The consequences run through the rest of the note: no entropy term, no log-probability, exploration has to be bolted on from outside (§4), and because the gradient flows through the critic rather than through a sampled log-probability, it is much lower variance — and much more dependent on the critic being right.

## 3. Freezing the critic: the GAN analogy

Draw the architecture and you get a *composition* of two networks: state → actor → action → critic → scalar. The course's observation is that this is structurally a **GAN**:

| GAN | DDPG |
|---|---|
| $z \to G \to$ sample $\to D \to$ score | $s \to \mu \to$ action $\to Q \to$ value |
| Train $G$ by maximising $\log D(G(z))$ w.r.t. $\theta_G$, **holding $\theta_D$ fixed** | Train $\mu$ by maximising $Q(s, \mu(s))$ w.r.t. $\theta_\mu$, **holding $\theta_Q$ fixed** |

The freezing is the load-bearing part. Gradients from the actor loss necessarily flow *through* the critic to reach the actor — that is the only path — but the critic's weights must not move on that step. If they did, the critic would learn to simply report large numbers instead of accurate ones, and the actor would be chasing an inflated target. In practice you give each optimiser an explicit parameter list: the critic optimiser owns the critic's weights, the actor optimiser owns only the actor's.

**Two losses, handled by alternation.** A2C adds the actor and critic losses into one scalar and differentiates the sum ([actor-critic-a3c](actor-critic-a3c.md) §1). DDPG takes the other option: update the critic against the TD error, then update the actor against $-Q(s, \mu(s))$, then back to the critic, alternating forever.

## 4. Exploration when the policy is deterministic

A deterministic policy in a deterministic environment visits exactly one trajectory, learns nothing new and never discovers that the pendulum could be swung the other way. So exploration is injected as **additive noise** on the action, then clipped back into the legal range:

$$a = \mathrm{clip}\big(\mu(s) + \sigma \cdot \mathcal{N}(0, I),\; -a_{\max},\; +a_{\max}\big)$$

Two practical details the course adds:

- **A random warm-up.** For the first `start_steps` steps, ignore $\mu$ entirely and sample uniformly from the action space. This fills the replay buffer with diverse data before the networks have any opinion worth following.
- **Zero noise at test time.** When evaluating, call the policy with noise scale 0. Noise exists only to explore; the policy itself is deterministic, so that *is* its behaviour.

**On the noise distribution.** The original DDPG paper used a correlated noise process — specifically an **Ornstein-Uhlenbeck** process, a random walk that drifts back toward zero, so successive noise samples push in a consistent direction instead of jittering. (The course describes it only as "a complicated method of generating correlated noise" and never names it, which leaves a reader unable to recognise OU noise when they meet it in a paper or codebase. The name is worth knowing; `OrnsteinUhlenbeckActionNoise` appears in plenty of implementations.) The course's substantive claim — that plain uncorrelated Gaussian noise works just as well — is **correct**, and was confirmed by TD3, which dropped OU noise entirely. Its stated moral, experimentation over philosophy, holds up.

## 5. What carries over from DQN — and the `tau` trap

Everything stabilising about DQN is reused unchanged, so [deep-q-learning](deep-q-learning.md) §2 is the reference. What changes is only that there are now **two** networks to copy:

**Replay buffer.** Identical: store $(s, a, r, s', \text{done})$, sample random minibatches. DDPG is **off-policy** — the critic's target uses $\mu_{\text{targ}}(s')$ rather than the action the behaviour policy actually took, exactly as DQN's target uses $\max_{a'}$ regardless of what happens next — so old data from a worse policy is still valid training data. That is the whole reason a buffer is allowed here and not in A2C/PPO.

**Target networks, now a pair.** The critic's target is built from *both* target networks:

$$y = r + \gamma\,(1 - \text{done})\,Q_{\text{targ}}\big(s',\, \mu_{\text{targ}}(s')\big)$$

Read it against DQN's $r + \gamma (1 - \text{done}) \max_{a'} Q_{\text{targ}}(s', a')$: the `max` has been replaced by the target *actor*, which is the single structural change in the critic loss. $y$ must be treated as a constant (`torch.no_grad()`, or TF1's `stop_gradient`) — target weights only ever move by averaging, never by backprop. The $(1 - \text{done})$ factor zeroes the bootstrap at a true terminal state, whose value is zero by definition.

**Soft updates instead of hard copies.** Where DQN copies weights wholesale every N steps, DDPG blends a small fraction every step — exponential smoothing, the same mechanic as a momentum term or a running average:

$$\theta_{\text{targ}} \leftarrow \tau\,\theta_{\text{online}} + (1 - \tau)\,\theta_{\text{targ}}, \qquad \tau \ll 1$$

applied to both target networks.

> ### ⚠️ The `tau = 0.999` trap
>
> The course writes the soft update the other way round — $\theta_{\text{targ}} \leftarrow \tau\,\theta_{\text{targ}} + (1-\tau)\,\theta_{\text{online}}$ — and then says "typically we use a value of `tau` very close to one such as **0.999**". With *its* equation that is right: `tau` is the **retention** factor, the share of the old target kept.
>
> In the DDPG paper and essentially every implementation, `tau` is the opposite: the fraction of the **new** weights mixed in, and it is **small** — roughly 0.001 to 0.005. The two conventions are related by $\tau_{\text{course}} = 1 - \tau_{\text{literature}}$, so the course's 0.999 is the literature's **0.001**.
>
> Copy `tau = 0.999` into a codebase that uses the standard convention and you get target networks that track the online networks at 99.9% per step — i.e. effectively no target network at all, the dog chasing its own tail, and training that diverges while every line of code looks correct. The course's own code dodges this by naming the argument `decay` rather than `tau`, which is honest, but the lecture's number is the one you remember. **Always check which side of the equation your `tau` multiplies.** The snippet below uses the standard convention: `TAU = 0.005` means "move the target 0.5% of the way toward the online network, every step".

## 6. A DDPG you can run

Pendulum-v1 is the free environment the course leans on, and it is a good one: state is the 3-vector $[\cos\theta, \sin\theta, \dot\theta]$ with $\cos,\sin \in [-1,1]$ and $\dot\theta \in [-8, 8]$; the action is a single torque ("joint effort") in $[-2, 2]$; and the reward is fully documented as

$$r = -\big(\theta^2 + 0.1\,\dot\theta^2 + 0.001\,a^2\big)$$

whose coefficients read as a priority ordering — angle matters most, angular velocity some, effort a little. The best possible step reward is $0$ (upright, motionless, no torque applied) and the worst is about $-16.27$ (hanging straight down at $\theta = \pm\pi$, spinning at $\dot\theta = \pm 8$, full torque). Continuous state *and* continuous action, which is exactly what DQN cannot do.

The question this snippet answers: **does the whole thing actually work, and how few lines does it take?** Everything from §2–§5 is here — a deterministic actor, a critic that eats the action, Gaussian noise, a buffer, two target networks, the $-Q(s, \mu(s))$ loss and the soft update — in about twenty lines.

```python
import numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.manual_seed(0); torch.set_num_threads(1)
rng = np.random.default_rng(0)                                         # seed the noise and the replay sampler too
env = gym.make("Pendulum-v1"); env.action_space.seed(0)
S, A, A_MAX = 3, 1, 2.0
mlp = lambda i, o, tail: nn.Sequential(nn.Linear(i, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, o), tail)
actor, critic = mlp(S, A, nn.Tanh()), mlp(S + A, 1, nn.Identity())     # actor outputs an action, not a distribution
at, ct = mlp(S, A, nn.Tanh()), mlp(S + A, 1, nn.Identity())            # slow-moving target copies
at.load_state_dict(actor.state_dict()); ct.load_state_dict(critic.state_dict())
oa, oc = torch.optim.Adam(actor.parameters(), 1e-3), torch.optim.Adam(critic.parameters(), 1e-3)
buf, GAMMA, TAU, N = [], 0.99, 0.005, 100_000                          # TAU here is the standard "how fast targets move"
def act(s, noise):
    with torch.no_grad(): a = A_MAX * actor(torch.tensor(s)).numpy()
    return np.clip(a + noise * rng.standard_normal(A), -A_MAX, A_MAX)  # Gaussian exploration noise
returns = []
for ep in range(60):
    s, _ = env.reset(seed=ep); total, done = 0.0, False
    while not done:
        a = env.action_space.sample() if ep < 5 else act(s, 0.1 * A_MAX)   # random warm-up fills the buffer
        s2, r, term, trunc, _ = env.step(a); done = term or trunc
        buf.append((s, a, r, s2, float(term))); buf[:] = buf[-N:]; total += r; s = s2
        if len(buf) < 1000: continue
        idx = rng.integers(len(buf), size=64)
        bs, ba, br, bs2, bd = [torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*[buf[i] for i in idx])]
        with torch.no_grad():                                           # critic target uses the TARGET actor and critic
            y = br[:, None] + GAMMA * (1 - bd[:, None]) * ct(torch.cat([bs2, A_MAX * at(bs2)], 1))
        oc.zero_grad(); ((critic(torch.cat([bs, ba], 1)) - y) ** 2).mean().backward(); oc.step()
        oa.zero_grad(); (-critic(torch.cat([bs, A_MAX * actor(bs)], 1)).mean()).backward(); oa.step()
        with torch.no_grad():                                           # soft update: target = tau*online + (1-tau)*target
            for net, tgt in ((actor, at), (critic, ct)):
                for p, q in zip(net.parameters(), tgt.parameters()): q.mul_(1 - TAU).add_(TAU * p)
    returns.append(total)
print("mean return, episodes  1-10 (random warm-up):", round(np.mean(returns[:10])))
print("mean return, episodes 51-60 (trained):       ", round(np.mean(returns[-10:])))
print("a Pendulum return near -150 is good; -1200 is a pendulum that never gets up")
# Expected output:
# mean return, episodes  1-10 (random warm-up): -1402
# mean return, episodes 51-60 (trained):        -154
# a Pendulum return near -150 is good; -1200 is a pendulum that never gets up
```

Walking the parts, in the order they appear:

- **The actor** is an MLP ending in `nn.Tanh()`, and its output is multiplied by `A_MAX` at every use site. `tanh` lands in $[-1, 1]$, so `A_MAX * tanh(...)` lands in $[-a_{\max}, +a_{\max}]$ — the action bound is satisfied by construction, not by clipping. This squash-and-scale is the standard actor head; the course builds the multiplication into the network itself, which is equivalent. Note the assumption baked in: a **single scalar** `A_MAX`, symmetric about zero and identical across all action dimensions. True for Pendulum and for the MuJoCo locomotion envs; on an environment with asymmetric or per-dimension bounds this silently scales into the wrong range without raising anything, so check `env.action_space.low`/`.high` before reusing it.
- **The critic** takes `S + A` inputs — `torch.cat([state, action], 1)` — and ends in `nn.Identity()`. No output activation, because $Q$ is an unbounded real-valued return, not a probability. This is the "action as an input" architecture of §1, now legal.
- **The target pair** `at, ct` are fresh networks initialised by copying the online weights (`load_state_dict`), the one-off hard copy that every implementation does before the first step.
- **The buffer** is a plain list truncated to the last `N` transitions. Fine at this scale; the course's version preallocates NumPy arrays with a circular write pointer, which keeps RAM constant and avoids the quadratic cost (and, as the course notes, the memory leak) of popping from the front of a Python list in a long run.
- **Exploration** is `act(s, 0.1 * A_MAX)`: Gaussian noise at 10% of the action bound, clipped. For the first five episodes the code ignores the actor and calls `env.action_space.sample()` — the random warm-up of §4 — and no gradient step happens at all until the buffer holds 1000 transitions.
- **The critic target** is the one block under `torch.no_grad()`: the *target* actor proposes `A_MAX * at(s2)`, the *target* critic scores it, and the result is $y$. Two easy mistakes live here, both flagged by the course: feed the **next** state (not the current one) into the target networks, and keep only the target's $Q$-of-$\mu$ output — the target actor's action is never used for anything else.
- **The two gradient steps** are the alternation of §3: critic on `(Q(s,a) - y)^2`, then actor on `-critic(cat([s, A_MAX*actor(s)])).mean()`. The actor's gradient flows *through* the critic — it has to — but only `oa` (the actor's optimiser) calls `.step()`, so the critic's weights don't budge.
- **The soft update** is the last three lines: `q.mul_(1 - TAU).add_(TAU * p)` with `TAU = 0.005`, standard convention, applied to both pairs, after every gradient step.
- **Termination vs truncation.** The stored done flag is `float(term)`, *not* `term or trunc`. Pendulum never terminates — it only hits its 200-step time limit — so a 4-tuple-era implementation that stored `done = term or trunc` would wrongly tell the critic that the episode's value ends there. Bootstrap on truncation; zero the bootstrap only on real termination. (The course does this bookkeeping by hand, with a separate stored flag, because its gym version collapsed both into one `done`; Gymnasium now hands you the two separately.)

**What it prints.** Averaged over the first ten episodes — five of pure random actions plus the first few barely-trained ones — the return is about **−1401**: a pendulum flailing around the bottom, never getting up. Over the last ten of sixty episodes it is about **−156**, within a whisker of the roughly −150 that counts as a solved swing-up-and-hold. For scale, the no-op/never-up regime sits near −1200, so the gap between the two printed numbers is most of the distance from "random" to "optimal". Twelve thousand environment steps, about eleven seconds on a CPU, from nothing to competent control — which is also a fair illustration of why off-policy methods with a replay buffer are prized for expensive environments.

## 7. The course's own code and environments

The course's `ddpg.py` is **TensorFlow 1**, graph-and-session style, adapted from OpenAI Spinning Up. Its shape maps onto the snippet above one-to-one:

| Course (TF1) | Modern equivalent |
|---|---|
| `tf.placeholder` for `X`, `A`, `X2`, `R`, `D` | function arguments / tensors |
| `create_networks(...)` called twice, under `variable_scope('main')` and `'target'` | two sets of modules, `load_state_dict` to sync |
| `variable_scope('q', reuse=True)` to score `concat([s, mu])` with the same critic weights | just call the critic module again |
| `get_vars('main')` — substring match on `tf.global_variables()` names | `model.parameters()` |
| `tf.stop_gradient` on the target | `with torch.no_grad():` |
| explicit `var_list` per optimiser | one optimiser per module |
| `tf.assign` soft-update ops + a one-off `target_init` op | in-place `mul_`/`add_` on the target parameters |

None of the TF1 API survives in TF2 (`placeholder`, `Session`, `variable_scope(..., reuse=True)`, `global_variables` are all gone); running the file needs `tf.compat.v1` plus `disable_eager_execution()`, or a rewrite. The `get_vars(scope)` substring trick in particular has no TF2 analogue.

**One deliberate deviation worth knowing.** The pseudocode in the lecture trains after every environment step. The code does not: it plays a whole episode, *then* runs a training loop for exactly `episode_length` iterations (sample a batch → both gradient steps → soft updates). Same total number of updates, just batched at the episode boundary. The course attributes the restructuring to TD3, "the sequel to DDPG". The snippet in §6 uses the per-step form, which is the textbook version.

**Hyperparameters: the course states almost none.** It names every argument of `ddpg()` — `replay_size`, `gamma`, `decay`, `mu_lr`, `q_lr`, `batch_size`, `start_steps`, `action_noise`, `max_episode_length`, `num_train_episodes`, `hidden_sizes` — and reads out a numeric value for essentially none of them. The only two numbers that surface near the code are `test_agent_every` ≈ **25 episodes** and the `tau` ≈ 0.999 from the theory lecture (see §5); the 1,000,000-entry buffer and batch size 32 mentioned in passing belong to the DQN *review*, not to its DDPG settings. **Results are likewise qualitative only** — HalfCheetah "learns to run correctly" with a plausible quadruped gait, Pendulum goes from spinning endlessly to a near-instant swing-up and hold. No episode returns, no reward curves. If you want concrete starting values, take them from Spinning Up's published DDPG defaults or from the snippet above, not from the lecture.

**MuJoCo: the install lecture is entirely obsolete.** The course devotes a lecture to MuJoCo ("Multi-Joint dynamics with Contact") licensing — a licence key in `~/.mujoco/mjkey.txt`, a 30-day trial, a price list running from free-for-students to five figures for institutions, plus `mujoco-py` and a string of install bugs. **DeepMind acquired MuJoCo in 2021 and open-sourced it under Apache 2.0.** There is no key, no trial, no price list, and no Mac/Linux-only restriction. `mujoco-py` is deprecated and unmaintained; the official `mujoco` Python bindings replace it. The 2026 version of that entire lecture is `pip install mujoco "gymnasium[mujoco]"`, and the advice to "get through this section inside your 30-day trial" is meaningless.

Three more things have moved since recording:

- **Environment IDs.** `Pendulum-v0` → `Pendulum-v1`; `HalfCheetah-v2` → `v4`/`v5`; likewise `Hopper`, `Walker2d`, `Ant`, `Humanoid`.
- **gym → gymnasium.** `env.step()` returns **five** values (`obs, reward, terminated, truncated, info`) instead of four, which is what makes the termination/truncation handling in §6 straightforward. `gym.wrappers.Monitor`, which the course wraps its test environment in to record videos, was removed; use `RecordVideo` (plus `RecordEpisodeStatistics`).
- **HalfCheetah is documented now.** The course says it could find no documentation for HalfCheetah's state, action or reward and falls back to guessing from the Humanoid docs, conceding "we don't know the reward structure". That was true in 2019. Gymnasium now documents the 17-dimensional observation, the 6-dimensional action space and the exact reward (a forward-velocity term minus a control cost, both weights configurable). Don't carry the "unknown reward" line forward.

## 8. Where DDPG sits now

DDPG against A2C, as the course lays it out:

| | A2C | DDPG |
|---|---|---|
| Action spaces | discrete **and** continuous | continuous **only** |
| Actor | $\pi$ — stochastic policy | $\mu$ — deterministic policy |
| Critic | $V(s)$ — state value | $Q(s, a)$ — action value |
| On/off-policy | on-policy | **off-policy** (replay buffer) |
| Stabilisation trick | parallel workers | replay buffer + slow target networks |

The course closes by noting that continuous-control benchmarks are scarce precisely because the famous RL results (Atari, chess, Go) are all discrete — while continuous control is arguably the more important case, since all physical motion is continuous.

What it could not tell you in 2019: DDPG is now **mostly of pedagogical interest**. Its known failure mode is the same $\max$-style overestimation bias that Double DQN fixes for DQN — the actor is trained to maximise the critic, so it reliably finds and exploits whatever states the critic over-values, and the two reinforce each other. DDPG is also notoriously brittle to hyperparameters and seeds. Two successors fix this and are what you should actually run:

- **TD3** (Twin Delayed DDPG, 2018) — keeps the deterministic policy and adds three things: **twin critics** with the target taking the *minimum* of the two (kills overestimation), a **delayed** actor update (every other critic update), and **target policy smoothing** (noise added to the target action, so the critic cannot exploit a sharp peak). TD3 is also where the course's train-after-episode restructuring and its "plain Gaussian noise is fine" claim come from.
- **SAC** (Soft Actor-Critic, 2018) — the stochastic successor: a Gaussian policy with an entropy bonus in the objective, so exploration comes from the policy itself rather than from bolted-on noise, with an automatically tuned temperature. Usually the first thing to try on a new continuous-control problem. [actor-critic-a3c](actor-critic-a3c.md) §5 covers it alongside PPO.

Read DDPG as the algorithm that establishes the idea — *learn the argmax, then differentiate the critic through it* — that both of them inherit.

## Trade-offs

- **Deterministic vs stochastic policy.** Low-variance gradients and one forward pass to act vs exploration must be injected from outside, and the policy can't represent "two equally good options".
- **Off-policy with a buffer vs on-policy A2C/PPO.** Far better sample efficiency — old data stays usable — vs less stable, since the critic is trained on a distribution the current policy no longer generates.
- **Critic as input-taking $Q(s,a)$ vs one output per action.** Scales to any action dimensionality vs you lose the free `argmax`, and must trust the actor to find it.
- **Soft updates vs hard copies.** Smoother, no sudden jump in the target vs one more hyperparameter, and a wrong `tau` (§5) fails silently rather than loudly.
- **DDPG vs TD3/SAC.** Simplest code, easiest to understand end to end vs measurably worse and far more seed-sensitive; the extra 30 lines of TD3 pay for themselves immediately.

## Common Pitfalls

- **Copying `tau = 0.999` from the course into standard-convention code**, because that convention mixes in 99.9% of the online weights every step — the target network stops being slow, the target chases the prediction, and training diverges. See §5.
- **Letting the actor loss update the critic**, because then the critic learns to output big numbers rather than accurate ones. Each optimiser gets an explicit parameter list.
- **Forgetting `no_grad`/`stop_gradient` on the target**, because backprop into a target network destroys the only thing that makes it a stable reference.
- **Feeding the current state into the target networks**, because the target is $Q_{\text{targ}}(s', \mu_{\text{targ}}(s'))$ — a bug that trains quietly and wrongly.
- **Zeroing the bootstrap on truncation**, because a time limit is not a terminal state; its future value is not zero. Store `terminated`, not `terminated or truncated`.
- **Leaving exploration noise on at evaluation time**, because the policy is deterministic — the noise is scaffolding and makes the agent look worse than it is.
- **Reusing the scalar `action_max` trick on an env with asymmetric or per-dimension bounds**, because `tanh * action_max` then scales into the wrong range without any error. Read `action_space.low`/`.high`.
- **Trying DDPG on a discrete action space**, because $\mu(s)$ must be differentiable in the action and `argmax` over a discrete set is not. Use DQN.
- **Judging DDPG from one seed**, because it is genuinely unstable run to run — this is the brittleness TD3 was written to fix, not something you did wrong.
- **Following the course's MuJoCo install lecture**, because every step of it (licence key, `mjkey.txt`, trial window, `mujoco-py`) is dead. One `pip install` replaces it.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Continuous actions, learning the algorithm | DDPG | DQN with binned actions | Binning loses action geometry and blows up as $b^d$ |
| Continuous actions, a real project | SAC (or TD3) | DDPG | Same family, fixes overestimation and seed brittleness |
| Discrete actions | [deep-q-learning](deep-q-learning.md) | DDPG | A deterministic actor needs a differentiable action |
| Environment steps are expensive | DDPG/TD3/SAC (off-policy + buffer) | A2C/PPO | Replay reuses every transition many times |
| Need a stochastic policy or tunable exploration | SAC, or a Gaussian [policy gradient](policy-gradients.md) | DDPG + noise | Entropy is in the objective, not bolted on |
| Writing the soft update | $\theta_{\text{targ}} \leftarrow \tau\theta + (1-\tau)\theta_{\text{targ}}$, $\tau \approx 0.005$ | $\tau = 0.999$ in that equation | Mind which side `tau` multiplies (§5) |
| Picking exploration noise | Uncorrelated Gaussian | Ornstein-Uhlenbeck | Simpler and works as well (TD3 confirmed it) |
| Need DDPG hyperparameters | Spinning Up defaults or §6's snippet | The course's lectures | It states essentially none |

## References

- [Lillicrap et al. — Continuous control with deep reinforcement learning (DDPG, 2015)](https://arxiv.org/abs/1509.02971)
- [Silver et al. — Deterministic Policy Gradient Algorithms (2014)](http://proceedings.mlr.press/v32/silver14.html) — the theorem DDPG puts a deep network on.
- [Fujimoto et al. — Addressing Function Approximation Error in Actor-Critic Methods (TD3, 2018)](https://arxiv.org/abs/1802.09477) · [Haarnoja et al. — Soft Actor-Critic (2018)](https://arxiv.org/abs/1801.01290)
- [Goodfellow et al. — Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661) — the analogy in §3.
- [OpenAI Spinning Up — DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) — the pseudocode and the published default hyperparameters; the course's code is adapted from it.
- [Gymnasium — Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) · [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) · [gym→gymnasium migration guide](https://gymnasium.farama.org/introduction/migration_guide/)
- [MuJoCo (open source, Apache 2.0)](https://github.com/google-deepmind/mujoco) · [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html) — replaces the course's licensing lecture and `mujoco-py`.

## See Also

- [deep-q-learning](deep-q-learning.md) — the replay buffer, target network and soft update DDPG inherits wholesale.
- [policy-gradients](policy-gradients.md) — the stochastic Gaussian policy that DDPG's deterministic $\mu$ replaces, and the log-probability trick it avoids.
- [actor-critic-a3c](actor-critic-a3c.md) — the actor-critic family; §5 covers PPO and SAC.
- [q-learning](q-learning.md) — the Bellman equation behind the critic target.
- [rl-foundations-mc-td](rl-foundations-mc-td.md) — §5 covers the shared Lazy Programmer appendix and course logistics.
