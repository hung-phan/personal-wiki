# ML Systems

Applied machine learning: how models are trained and used in real systems (embeddings, retrieval, recommendations). Math foundations live in [math-for-ml](../math-for-ml/INDEX.md).

- [embedding-training](embedding-training.md) — how to train an encoder for embeddings: why an autoencoder's bottleneck gives only surface similarity, the pretrain → contrastive fine-tune recipe, InfoNCE with in-batch / hard / false negatives, temperature, pooling (mean, `[CLS]`, last-token), SimCSE, Matryoshka, recall@k evaluation, CLIP/DINO/MAE.
- [behavioral-embeddings](behavioral-embeddings.md) — user/item embeddings from behavior: implicit feedback (clicks, purchases; weighted ALS, BPR, two-tower with logQ correction), explicit preferences (ratings, Bradley–Terry pairs), sessions (item2vec, SASRec), graphs (DeepWalk, LightGCN, PinSage), cold start, time-split evaluation and the popularity baseline.
