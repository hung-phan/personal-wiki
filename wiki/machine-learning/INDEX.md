# Machine Learning

Everything about machine learning, from the maths up to LLM systems. Each subfolder has its own `INDEX.md` with one line per note; search across all of them with `python3 scripts/extract-manifest.py <keyword>`.

| Folder | What's inside | Notes |
|--------|---------------|-------|
| [math-for-ml](math-for-ml/INDEX.md) | Foundations: statistics and probability, linear algebra, calculus, Markov chains, the transformer forward pass. | 5 |
| [classical-ml](classical-ml/INDEX.md) | Machine Learning A-Z: preprocessing, regression, classification, clustering, association rules, bag-of-words NLP, dimensionality reduction, model selection and XGBoost, ML on AWS. | 11 |
| [deep-learning](deep-learning/INDEX.md) | Deep Learning A-Z and Advanced NLP & RNNs: ANNs, CNNs, RNNs/LSTMs, SOMs, Boltzmann machines, autoencoders, text classification, seq2seq, attention, memory networks. | 10 |
| [reinforcement-learning](reinforcement-learning/INDEX.md) | AI A-Z, Deep RL in Python and ML A-Z Part 6: bandits, MC and TD, Q-learning, function approximation and TD(λ), policy gradients, DQN, actor-critic/A3C, PPO, SAC. | 7 |
| [llm-engineering](llm-engineering/INDEX.md) | LLM Engineering and AI A-Z generative parts: APIs and tools, open models, choosing models, RAG, fine-tuning and QLoRA, agents, prompting, responsible AI. | 10 |
| [ml-systems](ml-systems/INDEX.md) | Applied systems: embedding training and behavioural embeddings. | 2 |

## Suggested reading order

1. [math-for-ml](math-for-ml/INDEX.md) — as needed, not up front.
2. [classical-ml](classical-ml/INDEX.md) — preprocessing → regression → classification → model selection.
3. [deep-learning](deep-learning/INDEX.md) — artificial neural networks → CNNs → RNNs → attention.
4. [reinforcement-learning](reinforcement-learning/INDEX.md) — bandits → Q-learning → deep RL.
5. [llm-engineering](llm-engineering/INDEX.md) and [ml-systems](ml-systems/INDEX.md) — building real systems.
