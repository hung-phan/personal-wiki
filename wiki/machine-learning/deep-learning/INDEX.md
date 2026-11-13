# Deep Learning

Study guides for deep-learning courses: own explanations, original diagrams (`assets/`, drawn with `scripts/deep_learning_diagrams.py`), and runnable PyTorch + Keras snippets with expected output. Math foundations live in [math-for-ml](../math-for-ml/INDEX.md); applied systems in [ml-systems](../ml-systems/INDEX.md). Up: [machine-learning](../INDEX.md).

## Deep Learning A-Z (Udemy)

- [artificial-neural-networks](artificial-neural-networks.md) — Part 1 + ML-basics annex: neuron, activations, gradient descent and learning rate, batch vs mini-batch vs SGD, backprop checked by hand, leakage-free preprocessing, churn classifier in Keras and PyTorch with baseline, threshold and confusion matrix, logistic-regression comparison.
- [convolutional-neural-networks](convolutional-neural-networks.md) — Part 2: pixels as arrays, 3×3 convolution worked by hand, stride and output size, ReLU, max pooling, flatten, softmax + cross-entropy vs MSE gradients, cats-vs-dogs CNN in PyTorch and Keras with baseline, modern folder loading + preprocessing layers, the missing-1/255 prediction bug, Rekognition/Textract.
- [recurrent-neural-networks](recurrent-neural-networks.md) — Part 3: unrolled RNN checked by hand, vanishing/exploding gradients measured, forget-bias trick, LSTM equations verified against torch, GRU and peepholes, stacked-LSTM stock forecaster vs a naive yesterday baseline (it loses, and learns lag), extrapolation failure.
- [self-organizing-maps](self-organizing-maps.md) — Part 4: K-Means elbow and init trap vs k-means++/n_init, numpy SOM (weights as positions, BMU, shrinking neighbourhood, topology), U-matrix vs quantisation-error fraud flags scored against planted truth, why the SOM → ANN hybrid only learns pseudo-labels.
- [boltzmann-machines](boltzmann-machines.md) — Part 5: energy, why "restricted", Gibbs sampling and contrastive divergence, RBM like/dislike recommender in PyTorch and `keras.ops` with an always-like baseline, DBN vs DBM.
- [autoencoders](autoencoders.md) — Part 6: 4→2→4 toy, masked-loss movie recommender in PyTorch and Keras vs a movie-average baseline, overcomplete copy shortcut, sparse, denoising, contractive, stacked vs deep.

## Deep Learning: Advanced NLP and RNNs (Udemy)

- [text-classification-embeddings](text-classification-embeddings.md) — Sections 1–2, 7–13: embeddings as row lookups, tokenizing and padding, Conv1D + global max pool, multi-label sigmoid + BCE and per-label AUC vs misleading accuracy, RNN weight sharing, GRU checked against torch, LSTM return shapes, task shapes, Keras Sequential vs functional, appendix.
- [seq2seq-bidirectional-rnns](seq2seq-bidirectional-rnns.md) — Sections 3–4: bidirectional RNNs and when they leak the future, BiLSTM over image rows and columns, encoder-decoder and thought vector, teacher forcing, inference loop, reversal seq2seq vs copy baseline, translation and poetry settings.
- [rnn-attention](rnn-attention.md) — Section 5: alignment scores, softmax over time, context vectors, seq2seq with vs without attention (0.52 vs 0.995), attention maps, the course's Keras unrolled loop and chatbot, the step to Transformers.
- [memory-networks](memory-networks.md) — Section 6: bAbI, sentences as summed embeddings, attention over sentences, two hops, why order needs a time encoding (0.69 vs 1.00), the course's line-number trick.

## Machine Learning A-Z (Udemy)

- Part 8 → "Also in Machine Learning A-Z" sections of [artificial-neural-networks](artificial-neural-networks.md) (real churn results, R h2o) and [convolutional-neural-networks](convolutional-neural-networks.md) (training log, loss-comparison table). Other parts → [classical-ml](../classical-ml/INDEX.md).
