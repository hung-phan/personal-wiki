# ML Systems

Applied machine learning: how models are trained and used in real systems (embeddings, retrieval, recommendations). Math foundations live in [math-for-ml](../math-for-ml/INDEX.md). Up: [machine-learning](../INDEX.md).

- [embedding-training](embedding-training.md) — how embeddings are trained: an `Embedding` layer as one-hot × W trained by backprop, categorical columns in tabular nets (one-hot vs label vs hashing vs target encoding vs embedding, train-only vocabulary, a rare/unknown row, vector size), then training an encoder: why an autoencoder's bottleneck gives only surface similarity, the pretrain → contrastive fine-tune recipe, InfoNCE with in-batch / hard / false negatives, temperature, pooling (mean, `[CLS]`, last-token), SimCSE, Matryoshka, recall@k evaluation, CLIP/DINO/MAE.
- [behavioral-embeddings](behavioral-embeddings.md) — user/item embeddings from behavior: implicit feedback (clicks, purchases; weighted ALS, BPR, two-tower with logQ correction), explicit preferences (ratings, Bradley–Terry pairs), sessions (item2vec, SASRec), graphs (DeepWalk, LightGCN, PinSage), cold start, time-split evaluation and the popularity baseline.

## Recommender Systems and Deep Learning in Python (Udemy)

- [recommender-systems](recommender-systems.md) — the arithmetic layer before any model: popularity and its failure modes, lift, the Hacker News and Reddit ranking formulas, why a mean rating is the wrong score and the three fixes (smoothing, a pessimistic lower bound, sampling), proof that the damped mean is a conjugate posterior mean, then user-user and item-item collaborative filtering beating the baselines threefold.
- [matrix-factorization-recommenders](matrix-factorization-recommenders.md) — one shared latent space instead of an N-squared similarity table: the biased MF model, the alternating-least-squares derivation and its block convexity, why it is not SVD, the probabilistic reading (MLE = least squares, Gaussian priors = L2), embeddings and deep/residual/NCF/AutoRec variants, and Spark ALS at cluster scale. K matters: MSE 1.014 baseline, 0.105 at the true rank.
- Star ratings as softmax RBM units → [boltzmann-machines](../deep-learning/boltzmann-machines.md) §5.
