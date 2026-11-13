# Reinforcement Learning

Study guides for learning from rewards: own explanations, original diagrams (`assets/`, drawn with `scripts/reinforcement_learning_diagrams.py`), and snippets that train in seconds on a CPU (gridworlds, CartPole) next to the course's own settings. Neural-network basics live in [deep-learning](../deep-learning/INDEX.md); the probability behind MDPs in [math-for-ml](../math-for-ml/INDEX.md). Up: [machine-learning](../INDEX.md).

## AI A-Z (Udemy)

- [q-learning](q-learning.md) — Part 4: agent/environment/reward, Bellman equation and discounting, MDPs and stochastic moves, plan vs policy, living penalty and how it reshapes the policy, Q-values, temporal difference and the Q-learning update, ε-greedy, value iteration vs learning from experience.
- [deep-q-learning](deep-q-learning.md) — Parts 5–6: Q-network, experience replay, target network with soft updates, ε-greedy vs softmax, DQN on CartPole vs a random baseline, the course's LunarLander and Ms. Pac-Man (conv) agents, n-step returns, Double DQN and prioritised replay.
- [actor-critic-a3c](actor-critic-a3c.md) — Parts 7–8: actor and critic heads, advantage, entropy bonus, A3C vs A2C, n-step A2C on CartPole, the course's Kung Fu Master agent, PPO clipping and SAC entropy with worked numbers, DeepRacer.

## Advanced AI: Deep Reinforcement Learning in Python (Udemy)

- [rl-foundations-mc-td](rl-foundations-mc-td.md) — Sections 1–2 and 8–13: reward design, returns, the Bellman equation as a linear system, Monte Carlo vs TD(0) on a random walk, bandits and ε-greedy, off-policy Q-learning; Theano/TF1 mapped to PyTorch; appendix advice.
- [function-approximation-td-lambda](function-approximation-td-lambda.md) — Sections 3–4: Gymnasium CartPole and MountainCar, random search, bins, RBF features with a linear Q per action, optimistic initial values, why plain neural nets fail, n-step returns, λ-return and eligibility traces.
- [policy-gradients](policy-gradients.md) — Section 5: policy gradient theorem, REINFORCE with and without a baseline, Gaussian policies for continuous actions, hill climbing on MountainCarContinuous.
- Section 6 (DQN on Breakout) → [deep-q-learning](deep-q-learning.md) §8 · Section 7 (threaded A3C on Breakout) → [actor-critic-a3c](actor-critic-a3c.md) §6.

## Machine Learning A-Z (Udemy)

- [multi-armed-bandits](multi-armed-bandits.md) — Part 6: explore vs exploit, UCB, Thompson sampling with Beta posteriors, random vs UCB vs Thompson on 10 ads, delayed feedback.
