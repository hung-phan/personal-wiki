# Deep Learning

Study guides for deep-learning courses: own explanations, original diagrams (`assets/`, drawn with `scripts/deep_learning_diagrams.py`), and runnable PyTorch + Keras snippets with expected output. Math foundations live in [math-for-ml](../math-for-ml/INDEX.md); applied systems in [ml-systems](../ml-systems/INDEX.md).

## Deep Learning A-Z (Udemy)

- [artificial-neural-networks](artificial-neural-networks.md) — Part 1 + ML-basics annex: neuron, activations, gradient descent and learning rate, batch vs mini-batch vs SGD, backprop checked by hand, leakage-free preprocessing, churn classifier in Keras and PyTorch with baseline, threshold and confusion matrix, logistic-regression comparison.
- convolutional-neural-networks — Part 2 (planned): convolution, ReLU, pooling, flattening, full connection, softmax and cross-entropy, cats-vs-dogs style CNN.
- recurrent-neural-networks — Part 3 (planned): unrolled RNN, vanishing gradients, LSTM gates, sliding-window time series with a naive baseline.
- self-organizing-maps — Part 4 (planned): best matching unit, shrinking neighbourhood, U-matrix outliers, SOM → ANN hybrid.
- [boltzmann-machines](boltzmann-machines.md) — Part 5: energy, why "restricted", Gibbs sampling and contrastive divergence, RBM like/dislike recommender in PyTorch and `keras.ops` with an always-like baseline, DBN vs DBM.
- [autoencoders](autoencoders.md) — Part 6: 4→2→4 toy, masked-loss movie recommender in PyTorch and Keras vs a movie-average baseline, overcomplete copy shortcut, sparse, denoising, contractive, stacked vs deep.
