# Machine Learning

Everything about machine learning, from the maths up to LLM systems. Each subfolder has its own `INDEX.md` with one line per note; search across all of them with `python3 scripts/extract-manifest.py <keyword>`.

| Folder | What's inside | Notes |
|--------|---------------|-------|
| [math-for-ml](math-for-ml/INDEX.md) | Foundations: statistics and probability, linear algebra, calculus, Markov chains, the transformer forward pass. | 5 |
| [classical-ml](classical-ml/INDEX.md) | Machine Learning A-Z: preprocessing, regression, classification, clustering, association rules, bag-of-words NLP, dimensionality reduction, model selection and XGBoost, ML on AWS. | 11 |
| [deep-learning](deep-learning/INDEX.md) | Deep Learning A-Z, Advanced NLP & RNNs, GANs & VAEs, Advanced Computer Vision and PyTorch: ANNs, CNNs, RNNs/LSTMs, SOMs, Boltzmann machines, autoencoders, text classification, seq2seq, attention, memory networks, VAEs and GANs, transfer learning and CNN architectures, object detection, style transfer, a PyTorch cheatsheet, and a nine-step numpy build of a neural network from scratch, with the modern defaults derived from it (in Simplified Technical English). | 17 |
| [reinforcement-learning](reinforcement-learning/INDEX.md) | AI A-Z, Deep RL in Python, Cutting-Edge AI and ML A-Z Part 6: bandits, MC and TD, Q-learning, function approximation and TD(λ), policy gradients, DQN, actor-critic/A3C/A2C, PPO, SAC, DDPG and evolution strategies. | 9 |
| [llm-engineering](llm-engineering/INDEX.md) | LLM Engineering and AI A-Z generative parts: APIs and tools, open models, choosing models, RAG, fine-tuning and QLoRA, agents, prompting, responsible AI. | 10 |
| [ml-systems](ml-systems/INDEX.md) | Applied systems: embedding training, behavioural embeddings, and recommenders from ranking formulas to matrix factorization at cluster scale. | 4 |

## Suggested reading order

1. [math-for-ml](math-for-ml/INDEX.md) — as needed, not up front.
2. [classical-ml](classical-ml/INDEX.md) — preprocessing → regression → classification → model selection.
3. [deep-learning](deep-learning/INDEX.md) — artificial neural networks → CNNs → RNNs → attention.
4. [reinforcement-learning](reinforcement-learning/INDEX.md) — bandits → Q-learning → deep RL.
5. [llm-engineering](llm-engineering/INDEX.md) and [ml-systems](ml-systems/INDEX.md) — building real systems.
