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

## Deep Learning: GANs and Variational Autoencoders (Udemy)

- [variational-autoencoders](variational-autoencoders.md) — the bottleneck holds a distribution, not a code: latent-variable models, the ELBO and its two terms, the closed-form Gaussian KL, the reparameterisation trick, Bernoulli vs Gaussian decoders, beta = 0 vs 1 measured by how close generated samples land to real digits, and the course's softplus-sigma and dead TF1 `contrib` APIs.
- [generative-adversarial-networks](generative-adversarial-networks.md) — one discriminator, two losses: the minimax game as actually implemented (non-saturating flipped-label generator loss), Keras's frozen-discriminator compile-order trick, DCGAN's transposed convolutions and batch norm, and a runnable 1-D mode collapse (52/42% healthy vs 100/0% collapsed).

## Deep Learning: Advanced Computer Vision (Udemy)

- [transfer-learning-and-cnn-architectures](transfer-learning-and-cnn-architectures.md) — borrow the body, train the head: freeze vs fine-tune and when to cache features, VGG and `preprocess_input`, ResNet's identity and convolutional blocks, 1x1 convolutions and Inception bottlenecks, global pooling, and class activation maps (with proof that CAM needs average, not max, pooling).
- [object-detection](object-detection.md) — a fixed tensor cannot hold a variable number of objects: box offsets, anchor boxes, multi-scale feature maps (SSD300's 8,732 predictions), IoU and non-max suppression checked against torchvision, one-stage vs two-stage, mAP, and the localisation project. Flags that the course's "SSD / RetinaNet" section trains neither.
- [neural-style-transfer](neural-style-transfer.md) — freeze the network, make the image the variable: why pixel MSE has the trivial copy solution, content loss on one activation, the Gram matrix as the part of a feature map that survives shuffling the pixels (runnable proof), L-BFGS on the image, and what breaks under TF2 eager.

## Foundations

- [pytorch-cheatsheet](pytorch-cheatsheet.md) — task-indexed PyTorch 2.x API reference, in ASD-STE100 Simplified Technical English: tensors and the numpy bridge, autograd and why gradients add up, the canonical training loop, conv and pool output shapes with the floor, recurrent layer shapes and the `batch_first` asymmetry, losses that want logits against losses that want probabilities, and `state_dict` save and reload. The transformer section covers `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.TransformerEncoderLayer` and four measured traps: `batch_first=False` gives a wrong answer with no error, `attn_mask` against `key_padding_mask`, the mask sign that inverts between the module and the functional call, and where the parameters sit between attention and the feed-forward block. Current calls only, with no legacy spellings.

- [neural-networks-from-scratch](neural-networks-from-scratch.md) — write a working network in numpy in nine steps, then replace it with PyTorch. One neuron, then the gradient measured by finite differences (so the three parts — error x response x signal — are derived, not asserted), then the matrix multiply, why two layers need an activation, softmax and cross-entropy (the gradient is one subtraction, `p - y`, verified), the backward pass derived in six parts (the easy last-layer case, then the sum over paths that a hidden unit causes, then why that sum is exactly W transposed, then delta as the reusable piece, then a check against autograd, then the batch form), the complete 64-32-10 program with mini-batches and shuffling (0.9741 against a 0.1018 baseline), then initialisation, ReLU against sigmoid (491,943x against 5.8x gradient loss over 8 layers) and momentum. Ends with how to check your own gradient. Written in ASD-STE100 Simplified Technical English. Build order follows lionelmessi6410/Neural-Networks-from-Scratch.
