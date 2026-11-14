---
name: neural-style-transfer
description: "Neural style transfer (Gatys) explained end to end — repaint a photo in a painting's style by optimising the image instead of the weights; why raw-pixel MSE has the trivial copy solution so you need a frozen pretrained VGG, content loss as MSE on one intermediate activation and how tap depth trades sharpness for abstraction, the Gram matrix as uncentred channel correlation that discards spatial position (runnable shuffle proof), style loss summed over five block*_conv1 layers with per-layer weights, total loss weighting, average pooling instead of max pooling, and the scipy.optimize.fmin_l_bfgs_b loop that returns loss and gradient together in maxfun=20 bursts with manual pixel clipping. Flags what breaks in 2026: K.gradients / K.function under TF2 eager, layer.get_output_at(1), the bogus plus/minus 127.5 pixel bound, and the move to fast feed-forward (Johnson) and AdaIN style transfer. Study guide for Deep Learning: Advanced Computer Vision (Lazy Programmer) section 09."
created: 2026-11-13
updated: 2026-11-13
tags: [deep-learning, computer-vision, style-transfer, gram-matrix, vgg, lbfgs, optimisation, study-guide, advanced-computer-vision]
---
# Neural Style Transfer

> **How to read this note.** Study guide for *Deep Learning: Advanced Computer Vision* (Lazy Programmer), section 09 — seven lectures, roughly 44 minutes, split into theory, the content-only script, the style-only script, and the combined script. §1–§5 go with the theory lecture, §6–§7 with the optimisation lecture, §8 with the three code lectures, §9 is what a 2026 reader has to change before any of it runs. Convolution and feature maps are assumed — see [convolutional-neural-networks](convolutional-neural-networks.md); VGG and pretrained backbones live in [transfer-learning-and-cnn-architectures](transfer-learning-and-cnn-architectures.md). The one snippet here needs only `torch`.

## Why This Exists

**Problem.** You have a photo of a city skyline and a photo of Van Gogh's *Starry Night*, and you want the skyline — same buildings, same layout — painted in swirling thick-stroke blue. Nothing in ordinary supervised learning fits: there is no training set of (photo, painted photo) pairs, there is no label, and the thing you want to change is a single image rather than a model. Every tutorial jumps straight to "Gram matrix" without saying what breaks if you try the obvious thing instead.

**Key insight.** Freeze a pretrained CNN and make **the image itself the variable**. The network is no longer something you train; it is a fixed measuring instrument. One of its intermediate activations measures *content* ("are the same things in the same places?"), and the correlations *between its channels*, averaged over all positions, measure *style* ("what textures and colour pairings occur, anywhere?"). Add the two mismatches into one loss and run gradient descent on the pixels.

**Reach for this when:**
- Working through section 09 of *Advanced Computer Vision*, or any first read of Gatys et al.
- You want to understand why a Gram matrix is "style" and not just a matrix that happens to work.
- You need the general trick — optimise the input with frozen weights — for feature visualisation, adversarial examples, deep dream, or activation maximisation. Style transfer is the friendliest instance of it.
- You want one artistic image and don't mind waiting seconds to minutes for it.

**Don't reach for this when:**
- You need real-time or video styling. Per-image optimisation is far too slow; use a feed-forward style network (§10).
- You need to *generate* new images from nothing. Style transfer edits one image and learns no distribution — that's a [GAN or a diffusion model](generative-adversarial-networks.md).
- You want a quantitative result. There is no accuracy here, and the course reports no numbers at all — only "does it look right".

## Diagrams

![Neural style transfer architecture: content image and style image feed a frozen pretrained VGG alongside the image being optimised; one tap gives content loss, five taps give Gram matrices and style loss, and the total loss gradient flows back into the pixels](assets/style-transfer-architecture.svg)

<!-- DIAGRAM SPEC: name=style-transfer-architecture | panels: single wide panel, three image thumbnails stacked on the left labelled "content image x" (a photo), "style image s" (a painting), "generated image x' (THE VARIABLE)" drawn with a dashed highlight border; three arrows carry them rightwards into ONE box in the middle labelled "pretrained VGG16, weights FROZEN (avg pooling)" drawn as a stack of conv blocks block1..block5 | from the VGG box: an upper branch taps a single intermediate conv output -> small box "MSE over activations" -> "J_content"; a lower branch taps FIVE conv outputs (block1_conv1 .. block5_conv1), each -> small box "Gram (C x C)" -> "MSE" -> five boxes "J_style_1 .. J_style_5" -> weighted-sum node "sum w_L * J_style_L" -> "J_style" | J_content and J_style both enter a node "J = J_content + J_style"; one thick feedback arrow labelled "dJ/dx' (L-BFGS updates PIXELS, not weights)" curves from J all the way back to the dashed "generated image x'" thumbnail ONLY -- not to the VGG box | annotate the content tap with "deeper tap -> blurrier, more abstract" and the Gram boxes with "spatial axis cancels: H x W x C -> C x C" | the one idea: the gradient arrow returns to the image and never touches the network. | ASCII-only labels, write "->" for arrows and "x^2" for squares. -->

## 1. The setup: two inputs, one output, no dataset

Three images are in play. The **content image** `x` supplies the layout: what is where. The **style image** `s` supplies texture, brushwork and palette. The **generated image** `x'` is the answer, and it starts as random noise. There is no training set, no data generator, no augmentation — the course's "dataset" is one photo (an elephant, picked incidentally from Caltech 101) and one painting.

Both reference images are pushed through the network once to produce fixed numeric **targets**. After that, every optimisation step only needs a forward and backward pass on `x'`.

## 2. Why a pretrained CNN at all?

**The question.** If the goal is for `x'` to have the same content as `x`, why not just minimise the pixel-wise difference?

Write it down: `argmin_x' mean((x - x')^2)`. This is perfectly solvable, and the solution is `x' = x`. You get the content image back, bit for bit, with no room left for any style. A content measure that pins down every pixel is useless — it over-specifies the answer.

What you want is a measure that says "same objects, same arrangement" while staying *indifferent* to the fine detail, so the style term has freedom to fill that detail in. A pretrained CNN hands you exactly that for free. Each pooling layer deliberately forgets precise position — a pooled value could have come from any of several input locations — and each conv block trades pixel detail for more abstract features. So comparing activations deep in the network constrains the *structure* of `x'` while leaving its texture undetermined.

This is also why the network must be **pretrained**: the layers only encode meaningful features ("eye", "wheel", "stripe") because they were fitted on ImageNet. A randomly initialised CNN gives you a random projection, not a content measure. The course uses VGG16 — see [transfer-learning-and-cnn-architectures](transfer-learning-and-cnn-architectures.md) for what VGG is and why these early architectures remain the default backbone for this task.

## 3. Content loss: one activation, compared elementwise

Pick one intermediate convolution, call its output `f_L(·)`. Push both images through the same frozen network and average the squared differences:

```
J_content = mean( (f_L(x) - f_L(x'))^2 )
```

That is it — an MSE, just computed on activations instead of pixels.

**Which layer?** This is the one genuinely aesthetic choice in the whole method, and the course is blunt that there is no correct answer:

- **Zero convolutions** is the degenerate case of §2: comparing raw pixels, so `x' = x`.
- **Early layers** still carry near-pixel information, so the reconstruction comes out very sharp and almost identical to the original — little room for style.
- **Late layers** have pooled away position several times, so the reconstruction is blurry but structurally faithful: the elephant is unmistakably an elephant in the right pose, drawn loosely.

The course shows a strip of the same photo reconstructed from successively deeper taps, sharp on the left and blurry on the right, and sets reproducing it as an exercise. Its content-only run is described as "capturing the main features of the original" — no loss values or timings are given.

## 4. Style loss: the Gram matrix

**The question.** What makes two images "the same style"? Not matching objects in matching places — a textile has the same pattern wherever you look at it. So whatever we measure must survive moving every stroke somewhere else.

That requirement is almost a recipe. Take a conv output `A` of shape `(H, W, C)`: `C` feature maps, each `H x W`. The channels are the "what" (this one fires on short curved strokes, that one on deep blue); the spatial axes are the "where". If *where* must not matter, **sum over the spatial axes and keep only relationships between channels**:

```
A        : conv output, shape (H, W, C)
A_flat   : permute to (C, H, W), then flatten the spatial dims -> (C, H*W)
G(A)     = A_flat @ A_flat.T / (number of elements)      # shape (C, C)
```

Entry `G[i, j]` is large when channels `i` and `j` tend to fire at the same positions. *Starry Night* has thick curved strokes co-occurring with deep blue, so its Gram matrix has a large entry wherever "curved stroke" meets "blue" — and that statement is true of the whole canvas, not of one corner. The course calls `G` an autocorrelation matrix because it correlates the features with themselves; take that as an analogy, not a definition — a real correlation subtracts means and divides by standard deviations, whereas `G` is an **uncentred second moment** over channels.

Compare Gram matrices with the same MSE, and because different depths capture patterns at different scales (small local strokes early, large global forms late), do it at **five** layers rather than one — matching the original paper. The course taps every layer whose name ends in `conv1`, i.e. `block1_conv1` through `block5_conv1`:

```
J_style_L = mean( (G(f_L(s)) - G(f_L(x')))^2 )
J_style   = sum_L w_L * J_style_L
```

### The Gram matrix really does throw position away

The cleanest way to believe this is to break content on purpose and watch the style measure refuse to notice. The snippet below takes one layer's feature maps, applies the *same* random shuffle of rows and columns to every channel (so every pair of channel values that co-occurred at a pixel still co-occurs — just somewhere else), and then computes both losses between the original and the shuffled version.

```python
import torch
torch.manual_seed(0)
f = torch.randn(1, 8, 12, 12)                                  # one layer's feature maps: 8 channels, 12x12
def gram(x):                                                   # style = correlations between channels, position dropped
    n, c, h, w = x.shape
    flat = x.reshape(c, h * w)
    return flat @ flat.T / (c * h * w)                          # normalise so the scale doesn't depend on size
shuffled = f[:, :, torch.randperm(12)][:, :, :, torch.randperm(12)]   # scramble WHERE things are, keep WHAT
print("content changed:", not torch.allclose(f, shuffled))
print("style (Gram) unchanged:", torch.allclose(gram(f), gram(shuffled), atol=1e-5))
content_loss = ((f - shuffled) ** 2).mean()                     # content loss compares feature maps elementwise
style_loss = ((gram(f) - gram(shuffled)) ** 2).mean()           # style loss compares Gram matrices
print(f"content loss {content_loss:.3f}   style loss {style_loss:.2e}")
# Expected output:
# content changed: True
# style (Gram) unchanged: True
# content loss 2.057   style loss 2.38e-17
```

The content loss is about **2.06**. That is no accident: for unit-variance features, shuffling positions makes the two tensors effectively unrelated at each pixel, and `E[(a - b)^2] = 2 var = 2`. Elementwise comparison sees two completely different images. The style loss is about **2.4e-17** — pure float round-off, i.e. exactly zero. Same style, totally different content, from one tensor and a permutation. If you only remember one thing about the Gram matrix, remember this: *it is the part of a feature map that survives rearranging the pixels.*

A caveat the demonstration also exposes: invariance to shuffling is a double-edged property. Nothing in `J_style` prevents the optimiser from putting the right texture in the wrong arrangement, which is precisely why the style-only run in the course "captures the style of *Starry Night* without any global detail" — none of the painting's structures survive, only its texture.

## 5. Total loss

```
J = w_content * J_content + sum_L w_L * J_style_L
```

Two practical notes from the course. First, `w_content` is strictly superfluous: only the *ratio* of content to style weight changes the picture, so one of the weights can always be folded into the others. Second, the useful magnitudes depend on which layers you tapped — a Gram matrix from `block1_conv1` and one from `block5_conv1` have wildly different scales, so a weight vector tuned for one set of taps means nothing for another. The course states no weight values; "experiment" is the whole guidance, and it frames the choice as art rather than tuning.

That scale sensitivity is partly self-inflicted. The lecture says dividing the Gram matrix by its element count "isn't really necessary" since the loss is weighted anyway — true, but the normalisation is exactly what would make one weight vector transfer across layers of different spatial size. Skipping it is why the weights have to be re-tuned per layer choice.

## 6. The twist: the image is the variable

Ordinary training is `argmin_W J(W)`: data fixed, weights free. Style transfer is the mirror image:

```
argmin_x'  J(x')        weights frozen, image free
```

The course frames this as a descendant of the old feature-visualisation question "what is this hidden unit looking for?", answered by `argmax_x activation(x)` — find the input that excites a unit most. Same machinery, different unknown.

Mechanically this is easy: autodiff computes `dJ/dx'` exactly as readily as `dJ/dW`, since the input is just another node in the graph. In the course's Keras-1/2 idiom that is `grads = K.gradients(loss, model.input)`, wrapped into a callable with `K.function`. The real obstacle is plumbing: `model.fit` has no way to update an input, so the training loop must be written by hand.

Worth stating plainly because it is the cheapest thing about the method: **no training data, no training run, no weight updates, no checkpoint.** Compare [generative-adversarial-networks](generative-adversarial-networks.md), where you train a generator on a dataset and end up with a reusable model — style transfer trains nothing and the "result" is a single image you cannot reuse.

## 7. Optimising with L-BFGS

With the gradient in hand you could run plain gradient descent (or Adam — see [artificial-neural-networks](artificial-neural-networks.md) for the optimiser family). The course instead reaches for `scipy.optimize.fmin_l_bfgs_b`, and the reasoning is sound even where the details have aged.

**Why L-BFGS fits this problem.** Stochastic optimisers are built for *noisy* gradients: you see a different minibatch each step, so you take small steps and average out the noise. Here there is no minibatch. The objective is one fixed, deterministic, smooth function of the pixels, and every gradient is exact. That is the regime classical optimisation was designed for, and L-BFGS exploits it: it keeps a short history of recent gradients and steps, uses it to approximate curvature, and runs a line search so you never pick a learning rate. On a noise-free objective that converges in far fewer function evaluations than fixed-step gradient descent.

The lecture's gloss "L-BFGS only needs the gradient, so no second derivatives" is half the story. It is a **quasi-Newton** method — it builds an implicit inverse-Hessian approximation *from* the gradient history. It genuinely uses curvature information; what it avoids is ever forming or inverting a Hessian.

**Four implementation details, each with a reason:**

1. **Return loss and gradient from one call.** SciPy's `func` may return the `(loss, grad)` tuple together, which suits graph frameworks perfectly: one compiled function produces both outputs from a single forward/backward pass. The alternative, `approx_grad=True`, is rejected outright — finite differences would be slower *and* worse when the exact gradient is available.
2. **Flatten, and cast to float64.** SciPy treats the unknown as a 1-D vector, but the image is `(1, H, W, 3)`. So the objective is wrapped: reshape the incoming vector into an image, evaluate, flatten the returned gradient, and cast **both** loss and gradient to `float64`. The lecture calls the cast a weird SciPy quirk; it is not a bug — SciPy's L-BFGS-B core is Fortran and expects double precision, and handing it float32 misbehaves silently. Keep this step.
3. **Short bursts plus manual clipping, instead of `bounds`.** Pixels have a legal range, and the principled fix is L-BFGS-B's own `bounds` argument — but box constraints slow the optimiser down a lot. So the code calls `fmin_l_bfgs_b(..., maxfun=20)` in a loop, clips `x` back into range between bursts, and records the loss each time. The course says only "a few" outer iterations and gives no number.
4. **A random initial image.** `x0` is a random vector — the lecture doesn't state the distribution. (You can equally initialise from the content image, which converges faster and is what most modern implementations do.)

## 8. What the course's code does differently from stock VGG

- **Average pooling replaces max pooling.** `VGG16_AvgPool` walks the original VGG16 layer by layer into a fresh `Sequential`, substituting `AveragePooling2D` for every `MaxPooling2D` and copying everything else. The stated reason is empirical — it simply looks better — with the plausible explanation that max pooling discards all but one value per window while averaging keeps a contribution from every pixel. For a method whose gradient flows all the way back to the pixels, that matters: through a max pool, 3 of every 4 pixels receive no gradient at all on that step.
- **A cut-off variant for the content script.** `VGG16_AvgPool_CutOff(..., num_convs)` stops after `num_convs` convolutions, asserting `1 <= num_convs <= 13` because VGG16 has 13 convolutions. The course calls it "not strictly needed" — a multi-output model does the same job.
- **A multi-output model for the five style taps**, built from the layers whose names end in `conv1`, with `predict(style_img)` giving the numeric Gram targets.
- **Preprocessing has to be undone for display.** `preprocess_input` converts to VGG's BGR, mean-subtracted format; `unpreprocess` inverts it and `scale_img` rescales into `[0, 1]`, because matplotlib wants RGB in range.
- **The style image is resized to the content image's shape**, not the other way round: the MSE needs matching tensor shapes, resizing the content would warp the thing you care about, and a texture doesn't mind being stretched.

The three scripts build up in order — content only, style only, then both — with the second importing helpers from the first and the third from both. Results are reported qualitatively only; the course reads out no losses, timings or scores at any point, even though timing code is present.

## 9. What breaks if you run the course code in 2026

The section predates TensorFlow 2, and four claims should not be copied forward.

- **`K.gradients` and `K.function` do not work under eager execution.** They need a symbolic graph; under TF2's default eager mode `K.gradients` raises. Either call `tf.compat.v1.disable_eager_execution()` to restore the lecture's world, or — better — rewrite the loop with a `tf.Variable` image inside a `tf.GradientTape`, which is what the official TensorFlow tutorial does. This is not a small detail: it is the difference between the code running and not running.
- **`layer.get_output_at(1)` is a dead end.** The lecture spends real time on it: because `VGG16_AvgPool` reuses the *same layer objects* in a second model, Keras sees two models sharing layers, each layer then has two outputs, and index 1 is the avg-pool one. That inbound-node indexing is a Keras 1/2 internal, brittle then and effectively gone in Keras 3. Build the avg-pool network as a fresh architecture and copy the weights across instead.
- **The pixel bound "±127.5" is wrong on its own terms.** VGG's `preprocess_input` subtracts *per-channel* ImageNet means (roughly 103.9 / 116.8 / 123.7 in BGR), so the legal range after preprocessing is about −124 to +152 and is **not symmetric**. Clipping to ±127.5 happens to produce acceptable pictures, which is why the error goes unnoticed, but don't repeat it as a principled bound — derive the clip from the actual means, per channel.
- **`fmin_l_bfgs_b` still exists, but the documented entry point moved.** Prefer `scipy.optimize.minimize(..., method='L-BFGS-B', jac=True)`; `jac=True` is the same "return loss and gradient together" contract.

One more caveat that is about hardware rather than API: the lecture's "L-BFGS converges faster than gradient descent" was measured in a CPU-era script. SciPy's optimiser lives on the CPU, so every function evaluation forces a GPU-to-CPU round trip of the whole image and gradient. On a modern GPU, Adam on a `tf.Variable` usually wins on wall-clock despite needing more steps, because nothing ever leaves the device.

## 10. Where the field went

Optimising every output image from scratch is the expensive way to do this. Johnson et al. (2016) instead **trained a feed-forward network** to produce the styled image in one forward pass, using exactly the content and Gram losses above as its training objective — hundreds of times faster at inference, at the cost of one trained network per style. Huang & Belongie's AdaIN (2017) removed that restriction too, conditioning a single network on an arbitrary style image at test time by matching channel means and variances. Both are strictly better production choices. Gatys-style per-image optimisation survives as the *teaching* version — and as the clearest demonstration of the frozen-network, free-input trick that also powers feature visualisation and adversarial examples.

## Trade-offs

- **No training versus no reuse.** You skip datasets, labels and training runs entirely; in exchange you pay the full optimisation cost for every single image and keep nothing reusable afterwards.
- **L-BFGS versus Adam.** L-BFGS needs no learning rate and converges in fewer evaluations on this noise-free objective; it also forces CPU round-trips, cannot be batched naturally, needs a flatten/float64 wrapper, and handles box constraints badly enough that the course abandons them for manual clipping.
- **Content tap depth.** Shallow taps keep detail but crowd out style; deep taps give style room but lose sharpness. There is no setting that gives both.
- **Gram matrices as "style".** Position invariance is what makes them a style measure, and also what makes the method blind to the arrangement of its own textures. Later work (histogram and distribution losses, AdaIN's mean/variance matching) exists partly to patch that.
- **Average pooling.** Better gradients and better pictures, but you are now running a network whose pooling differs from the one the weights were trained with — defensible here because you only need the features as a *measure*, not as a classifier.

## Common Pitfalls

- **Don't use a randomly initialised network** because untrained filters encode nothing; "content" and "style" would both be random projections and the output would be noise with no structure.
- **Don't take the content loss on raw pixels** (or at zero convolutions) because the minimiser is `x' = x` exactly, which leaves the style term with nothing to change.
- **Don't reuse style weights across a different set of taps** because Gram magnitudes scale with layer size, so a weight vector tuned for one layer set silently over- or under-weights another. Normalising the Gram matrix mitigates this; the course skips the normalisation and then has to re-tune.
- **Don't quote "the Gram matrix is the autocorrelation matrix" as a definition** because it is uncentred and unnormalised — a convenient analogy, not the thing itself.
- **Don't drop the float64 cast in the SciPy wrapper** because the Fortran L-BFGS-B core expects doubles and failure is silent rather than loud.
- **Don't clip pixels to ±127.5** because VGG mean-subtraction is per channel and asymmetric; the correct bounds are `-mean_c` to `255 - mean_c`.
- **Don't expect numbers from this section** because the course reports results purely by eye — no losses, no timings, no metrics. Any figure you see quoted elsewhere as "the course's" is invented.
- **Don't copy TF1 symbolic calls into a TF2 file** because `K.gradients` raises under eager execution; port to `tf.GradientTape` or explicitly disable eager first.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| One artistic image, minutes to spare | Per-image optimisation (this note) | A feed-forward style net | Zero setup, zero training, any style image immediately |
| Styling video or anything interactive | Johnson et al. feed-forward net | Per-image L-BFGS | One forward pass per frame instead of hundreds of gradient steps |
| Arbitrary style, still fast | AdaIN | One network per style | A single network conditions on the style image at test time |
| Reimplementing the course in 2026 | `tf.Variable` + `tf.GradientTape` + Adam | `K.gradients` / `K.function` | Eager-compatible, and keeps image and gradient on the GPU |
| Output should stay sharp and photo-like | Shallower content tap, higher content weight | Deepest conv tap | Early activations still carry pixel detail |
| Output should be loose and painterly | Deeper content tap, higher style weights | Early tap | Pooling has discarded position, so texture has room |
| Generating *new* images of a class | GAN or diffusion model | Style transfer | Style transfer edits one image; it learns no distribution to sample from |
| Explaining which pixels drove a prediction | Class activation maps / Grad-CAM | Style transfer | Different question: attribution, not synthesis |

## References

- [Gatys, Ecker & Bethge, *A Neural Algorithm of Artistic Style* (2015) — the paper this section teaches, including the five-layer style setup](https://arxiv.org/abs/1508.06576)
- [Johnson, Alahi & Fei-Fei, *Perceptual Losses for Real-Time Style Transfer and Super-Resolution* (2016) — the feed-forward replacement](https://arxiv.org/abs/1603.08155)
- [Huang & Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (2017)](https://arxiv.org/abs/1703.06868)
- [Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (2014) — VGG](https://arxiv.org/abs/1409.1556)
- [SciPy, `fmin_l_bfgs_b` (the course's call)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html)
- [SciPy, `minimize` with `method='L-BFGS-B', jac=True` (the current entry point)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [TensorFlow, *Neural style transfer* tutorial — the eager/`GradientTape` version](https://www.tensorflow.org/tutorials/generative/style_transfer)
- [TensorFlow, `tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
- Course: *Deep Learning: Advanced Computer Vision*, Lazy Programmer (Udemy), section 09 — "Neural Style Transfer".

## See Also

- [convolutional-neural-networks](convolutional-neural-networks.md) — convolution, feature maps and pooling, assumed throughout here.
- [transfer-learning-and-cnn-architectures](transfer-learning-and-cnn-architectures.md) — VGG and pretrained backbones; why ImageNet features are reusable.
- [generative-adversarial-networks](generative-adversarial-networks.md) — the contrast: a GAN trains a generator on a dataset, while style transfer optimises one image against a frozen network with no training set at all.
- [artificial-neural-networks](artificial-neural-networks.md) — gradient descent and the optimiser family that L-BFGS is being compared against.
- The shared Lazy Programmer appendix bundled with this course is covered in [rl-foundations-mc-td](../reinforcement-learning/rl-foundations-mc-td.md) §5.
