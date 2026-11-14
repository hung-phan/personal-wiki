---
name: object-detection
description: "Object detection and localization explained — why a fixed-size tensor can't hold a variable number of objects, normalized box coordinates (top-left origin, row/col/height/width vs corners), why detectors predict offsets relative to anchor/default boxes instead of absolute coordinates, sliding windows re-expressed as convolution (the 'single shot' in SSD) and its hidden stride, multi-scale feature maps and which map catches which object size, 8732 default boxes of an SSD300 counted, confidence thresholding, IoU/Jaccard and non-max suppression from scratch checked against torchvision.ops.nms, the foreground/background class imbalance that hard negative mining and focal loss (RetinaNet) attack, one-stage vs two-stage detectors, mAP and COCO AP@[.50:.95]; plus the multi-output localization project (VGG + box head + presence head + class head, the presence gate on the box loss, loss weights, sigmoid+BCE on coordinates and why its loss floor isn't zero, alpha compositing of sprites onto backgrounds). Study guide for Deep Learning: Advanced Computer Vision sections 8 and 12."
created: 2026-11-13
updated: 2026-11-13
tags: [deep-learning, computer-vision, object-detection, ssd, anchors, non-max-suppression, iou, map, study-guide, advanced-computer-vision]
---
# Object Detection and Localization

> **How to read this note.** Study guide for *Deep Learning: Advanced Computer Vision* (Lazy Programmer): section 8 ("Object Detection (SSD / RetinaNet)") and section 12 (the object-localization project). §1–§9 are the ideas; §10 is what the course's notebooks actually run; §11 is the project. **Know this before you start.** The section is titled "Object Detection (SSD / RetinaNet)" and delivers neither architecture. SSD is explained and **never implemented or trained**; RetinaNet is **never mentioned at all**, and with it go focal loss, feature pyramid networks and any discussion of foreground/background class imbalance. What the course actually runs is inference with somebody else's pretrained checkpoint from the TensorFlow Object Detection API, at roughly 1 FPS rising above 3 FPS on a Colab GPU, and it **reports no mAP, precision, recall or accuracy number anywhere in either section**. §7 below is background the course omits, flagged as such. Convolution, strides and feature maps are assumed: see [convolutional-neural-networks](convolutional-neural-networks.md). Backbones, ResNet and the "deeper layers see bigger things" fact come from [transfer-learning-and-cnn-architectures](transfer-learning-and-cnn-architectures.md). Precision and recall come from [classification-models](../classical-ml/classification-models.md). The instructor's shared appendix is covered once in [rl-foundations-mc-td](../reinforcement-learning/rl-foundations-mc-td.md) §5. The snippets need `torch` and `torchvision`: `VIRTUAL_ENV=/Users/phahng/Desktop/workarea/.venv uv pip install torch torchvision`.

## Why This Exists

**Problem.** A classifier answers "what is this picture of?" with one label. A street scene contains eleven cars, three pedestrians, a bicycle and two traffic lights — and the next frame contains none of them. A neural network outputs a tensor of fixed shape, so there is no obvious place to put "however many objects happen to be here, each with a box". Every detector is an answer to that one structural problem, and the answer is not intuitive: you predict a *fixed, huge* number of candidate boxes and then throw almost all of them away.

**Key insight.** Fix the candidates in advance. Tile the image with thousands of **default boxes** (anchors) at several positions, scales and shapes; for each one, predict (a) a small correction to its coordinates and (b) a class score. A convolutional layer already evaluates every position in parallel, so one forward pass scores all of them at once — the "single shot". Then two pieces of ordinary non-ML code reduce thousands of boxes to a handful: drop everything below a confidence threshold, and among the survivors keep only the highest-scoring box of each overlapping cluster (**non-max suppression**). The variable-length output never comes out of the network; it comes out of the filter after it.

**Reach for this when:**
- You need *what and where*, for zero-or-more objects: counting, tracking, cropping, driving, inventory, "flag the frames that contain X".
- You are reading detector code or a detection paper and the words anchor, default box, offset encoding, IoU, NMS, AP50 or mAP are being used as if you already knew them.
- You have one object per image and only need its box — that is **localization**, and it is a four-number regression bolted onto a classifier (§11). Much simpler than detection; use it if it fits.
- You are evaluating a detector someone handed you and need to know what its mAP number does and does not promise.

**Don't reach for this when:**
- One label per image is enough. A classifier is smaller, faster, easier to train and easier to evaluate.
- You need per-pixel outlines rather than rectangles — that is instance segmentation (Mask R-CNN, SAM), a different head and a different loss.
- You are starting a detection project today and were about to type SSD code from this note. Don't: use a maintained implementation (§10) and spend your time on data.

## Diagrams

![SSD pipeline: a backbone CNN shrinking left to right, detection heads tapped off six feature maps of different sizes, each cell carrying several default boxes, all 8732 predictions pooled, then a confidence threshold and non-max suppression reducing them to three final boxes](assets/ssd-pipeline.svg)

<!-- DIAGRAM SPEC: name=ssd-pipeline | panels: (1) left-to-right backbone drawn as 6 shrinking slabs labelled "38x38", "19x19", "10x10", "5x5", "3x3", "1x1" with "conv + pool" arrows between them; under each slab a small label "sees ~7px", "~15px", "~30px", "~60px", "~100px", "~300px"; (2) above each slab a thin "head" box labelled "3x3 conv -> k*(4+21)" with a dashed arrow from slab to head; next to the 38x38 head a magnified single grid cell showing 4 overlaid default boxes (small square, large square, tall rectangle, wide rectangle) in 4 distinct colours; next to the 10x10 head the same magnified cell with 6 boxes; (3) all head outputs converge with arrows into one tall thin column labelled "8732 predictions x (4 offsets + 21 scores)"; (4) that column -> box labelled "score > threshold" -> box labelled "per-class NMS, IoU > 0.5 suppressed" -> a small photo-frame rectangle containing 3 clean boxes labelled "dog 0.91", "car 0.88", "person 0.76". Annotate between (3) and (4): "the only variable-length thing happens HERE, outside the network". labels: use "->" for arrows, no superscripts | the one idea it must land: the network always emits the same fixed-size tensor; the variable number of objects is produced by thresholding and NMS afterwards, and early/large feature maps carry the small objects while late/small maps carry the big ones. -->

![Sliding window as convolution: a 32x32 window network with its dense layers rewritten as convolutions produces a 1x1x3 output, while the identical weights applied to a 36x36 image produce a 2x2x3 grid of four class predictions, one per window, with the four input windows colour-matched to the four outputs](assets/sliding-window-as-convolution.svg)

<!-- DIAGRAM SPEC: name=sliding-window-as-convolution | panels: two side-by-side columns sharing one legend. Left column header "window: 32x32x3", shape chain drawn as stacked slabs with labels "32x32x3 -> conv 5x5 -> 28x28x32 -> pool/2 -> 14x14x32 -> conv 5x5 -> 10x10x64 -> pool/2 -> 5x5x64 -> conv 5x5 (was Dense 128) -> 1x1x128 -> conv 1x1 -> 1x1x128 -> conv 1x1 -> 1x1x3"; final slab annotated "one softmax". Right column header "image: 36x36x3", same chain with labels "36x36x3 -> 32x32x32 -> 16x16x32 -> 12x12x64 -> 6x6x64 -> 2x2x128 -> 2x2x128 -> 2x2x3"; final slab drawn as a 2x2 grid of 4 small coloured squares. Above the right column draw the 36x36 input as a square with 4 overlapping 32x32 window outlines in the SAME 4 colours as the output grid, offset by 4 pixels each. Caption strip along the bottom: "same weights, same parameter count; one pass instead of 4 crops. two 2x pools -> window stride = 4 input pixels, not 1". labels: ASCII only, use "->" | the one idea it must land: a dense layer re-expressed as a convolution turns a per-window classifier into a per-position classifier for free, and the window stride is set by the pooling, which is exactly why anchors are needed to cover the positions and shapes in between. -->

## 1. Three tasks, one ladder

**The question.** What changes, mechanically, as you go from "what is this?" to "where is everything?"

- **Classification.** Input image → one label. Output shape: $K$.
- **Localization.** One object, assumed present. Input image → one label *and* four numbers for its box. Output shape: $K + 4$. This is the same transfer-learning setup as a classifier with four extra regression units stapled on, and it trains with ordinary backprop. Relatives: predicting an ellipse for a face, or a list of facial keypoints (**landmark detection**) — same idea, more regression outputs.
- **Detection.** Zero, one or many objects, each with a label and a box. Output shape: **unknown at compile time**.

That last line is the whole difficulty. The course is blunt about it: a tensor has a fixed size, one image has no objects and another has fifty, and no choice of output width is correct. It offers a deliberate strawman first — emit the single highest-confidence object, erase that region from the image, run the network again, repeat until confidence drops below a threshold: "some kind of while loop". It works in principle and is hopeless in practice (many passes per image, and erasing destroys context for whatever was behind).

The real answer inverts the problem. Instead of asking the network for a variable number of answers, ask it for a *fixed, generous* number of answers about fixed, pre-chosen candidate boxes, and let non-neural code select. A useful framing the course repeats: detection moves you "from inside the CNN to outside" — the CNN becomes one component of a system whose other parts (anchor generation, decoding, thresholding, NMS) are plain programming with no learned parameters at all.

## 2. Writing a box down

**The question.** Four numbers — but which four, in what units, and relative to what?

**Normalize to [0, 1], never pixels.** A box stored as pixel coordinates is tied to one input resolution; divide by width and height and the same target works for any resize. It also keeps the regression targets in the range a sigmoid can produce.

**The origin is the top-left corner and the first index is the row**, which grows *downward*. This disagrees with school maths and with Matplotlib (horizontal first), and the resulting index swaps are a steady source of bugs — the course's plotting helper has to write `Rectangle((col*S, row*S), width*S, height*S)` to undo the convention twice over.

**Two parameterizations:**

| Form | Numbers | Problem |
|---|---|---|
| Corners | $x_1, y_1, x_2, y_2$ | Nothing enforces $x_2 > x_1$. A sigmoid keeps each number in [0,1] but cannot stop the network emitting an inside-out box. |
| Origin + extent | $\text{row}_0, \text{col}_0, h, w$ | $h, w$ out of a sigmoid are automatically positive, so the box is always valid. |

The course uses the second and is right to. Its remaining failure mode, which it names: the box can still spill past the image edge, since $\text{row}_0 + h$ can exceed 1. In practice a trained model mostly doesn't, and production decoders clip.

**Why detectors predict offsets, not coordinates.** This is the part section 8 skips, and it matters more than anything else in this section. A detection head is a convolution: the *same* weights are applied at every cell of the feature map. If the head predicted absolute coordinates, the weights would have to encode "I am the cell at row 17, column 4" — but a convolution is translation-invariant by construction and has no idea where it is. So the head predicts a **correction relative to its own default box**:

$$\hat{c}_x = \frac{c_x - c_x^{a}}{w^{a}}, \qquad \hat{c}_y = \frac{c_y - c_y^{a}}{h^{a}}, \qquad \hat{w} = \log\frac{w}{w^{a}}, \qquad \hat{h} = \log\frac{h}{h^{a}}$$

where the $a$ superscript is the anchor. Three things fall out. Centre offsets are measured *in units of the anchor's size*, so the same "shift right by 10%" means different pixel counts for a small and a large anchor — one set of weights serves both. Width and height are predicted in **log space**, so the regression target is symmetric (halving and doubling are $\pm\log 2$) and the decoded size $w = w^a e^{\hat{w}}$ is positive for free. And the correct output for a well-matched anchor is close to **zero**, which is a far easier target to learn than an arbitrary coordinate. The course's framing of default boxes as "sort of the opposite of regression — you're given the box and only decide whether something is in it" is the right intuition, and it correctly adds that "some regression is still done, easier because it's constrained by the box"; the formula above is what that constrained regression actually is.

## 3. A sliding window *is* a convolution

**The question.** The localization network of §1 eats a fixed-size crop. Do we have to run it on every crop of the image?

The naive answer: yes — loop over every window position, crop, call `model.predict`, and write plumbing to track where the window was and map the normalized coordinates back into the full image. That is $O(N^2)$ forward passes for an $N \times N$ image and it is unusable.

The insight (this is the best lecture in the section, and it is also the original OverFeat argument): sliding a small template over an image and recording a response at each position is the *definition* of convolution, and convolution is one GPU-parallel kernel. So re-express the *dense* layers as convolutions and the whole window-classifier becomes position-agnostic.

The course's worked example, on a 32×32×3 window: conv $3\times5\times5\times32$ → 28×28×32, pool → 14×14×32, conv $32\times5\times5\times64$ → 10×10×64, pool → 5×5×64, then Dense(128), Dense(128), Dense(3). Now rewrite the head:

- `Dense(128)` on a flattened 5×5×64 map ≡ a convolution with a $64\times5\times5\times128$ kernel. Map and kernel are both 5×5, so the output is **1×1×128** — numerically the identical vector, with the identical parameter count.
- `Dense(128)` on that ≡ a $128\times1\times1\times128$ convolution, which ignoring the 1×1 is just a $128\times128$ matrix.
- `Dense(3)` ≡ a $128\times1\times1\times3$ convolution.

Same parameters, same arithmetic — here convolution *is* matrix multiplication. The payoff: feed the identical network a **36×36×3** image instead (the 32×32 was only ever the window size) and the shapes become 32×32×32 → 16×16×32 → 12×12×64 → 6×6×64 → conv 5×5 → **2×2×128** → 1×1 convs preserve the grid → **2×2×3**. Four softmax outputs, one per window, from one pass. That is the "single shot" in SSD, and the second stated win is that there is no region-proposal stage at all: convolution checks everywhere simultaneously.

**What the course leaves out, and you should not miss.** Those four windows are not at stride 1. The two 2× pools make the effective window stride **4 input pixels** — which is exactly why 36×36 gives precisely a 2×2 grid ($(36-32)/4 + 1 = 2$). The lecture says "we set it up so that there are exactly four windows" without naming the stride, which can leave the impression that the trick gives you dense stride-1 windows for free. It does not, and that gap in coverage — a coarse grid of positions, each of one fixed shape — is the entire reason anchors exist.

## 4. The problem of scale: several feature maps

**The question.** A sliding window assumes you know how big the object is. You don't: the car two blocks away is 12 pixels wide and the bicycle in front of you fills the frame, and both must be caught.

The course's two motivating crops are good ones. Zoom in too far and a patch of orange-and-black stripes is unidentifiable — it is part of a **tiger**, but no window at that scale can say so. Zoom out too far and the object is a few specks.

The fix reuses a fact established earlier in the course and in [transfer-learning-and-cnn-architectures](transfer-learning-and-cnn-architectures.md): as you go deeper, feature maps get smaller and each unit's receptive field gets larger, so deep units respond to big, abstract structures and shallow units to small, local ones. So don't pick one layer — treat **every stage as a feature extractor** and attach a small detection head to several of them. Each head detects at its own scale: the fine 38×38 map finds small objects, the 1×1 map finds objects that fill the image. (Same multi-branch spirit as Inception, and SSD is backbone-agnostic: VGG, ResNet and Inception all drop in.)

An SSD300 taps six maps. Counting what that buys:

```python
# SSD-style detection: every cell of several feature maps carries a few default ("anchor") boxes.
maps = [(38, 38, 4), (19, 19, 6), (10, 10, 6), (5, 5, 6), (3, 3, 4), (1, 1, 4)]   # SSD300: size, size, boxes/cell
total = sum(h * w * k for h, w, k in maps)
for h, w, k in maps:
    print(f"{h:2d}x{w:<2d} grid x {k} boxes = {h*w*k:5d}   each cell sees ~{300//h:3d}px of the 300px input")
print("total predictions per image:", total)
n_classes = 21                                                 # 20 VOC classes + background
print("outputs per box: 4 offsets + 21 class scores -> tensor of", total, "x 25 =", total * 25)
print("after score threshold 0.5 and NMS, typically a handful survive")
# Expected output:
# 38x38 grid x 4 boxes =  5776   each cell sees ~  7px of the 300px input
# 19x19 grid x 6 boxes =  2166   each cell sees ~ 15px of the 300px input
# 10x10 grid x 6 boxes =   600   each cell sees ~ 30px of the 300px input
#  5x5  grid x 6 boxes =   150   each cell sees ~ 60px of the 300px input
#  3x3  grid x 4 boxes =    36   each cell sees ~100px of the 300px input
#  1x1  grid x 4 boxes =     4   each cell sees ~300px of the 300px input
# total predictions per image: 8732
# outputs per box: 4 offsets + 21 class scores -> tensor of 8732 x 25 = 218300
# after score threshold 0.5 and NMS, typically a handful survive
```

5776 + 2166 + 600 + 150 + 36 + 4 = **8732 predictions per image**, each carrying 4 box offsets plus 21 class scores (20 Pascal VOC classes plus background) — a 8732 × 25 tensor, about 218,000 numbers, out of every forward pass. Note how the stride column changes the character of each map: a cell of the 38×38 map covers roughly 7 input pixels and there are 5776 of its predictions, so the overwhelming majority of the budget is spent looking for small objects, which is where detectors are weakest. The 1×1 map contributes 4 boxes and is responsible for "the object is the whole picture". After thresholding and NMS, a handful survive — which is the point: the fixed-size tensor is enormous precisely so that the variable-size answer can be carved out of it.

One honest caveat about the course's version: tapping backbone layers directly is **SSD's** design, not a feature pyramid. A **Feature Pyramid Network** adds a top-down path with lateral connections so that the fine, high-resolution map also carries the deep semantic features, and it is a large part of why later detectors (RetinaNet, most YOLO variants, Faster R-CNN v2) detect small objects far better. FPN is not mentioned in the course at all.

## 5. The problem of shape: default (anchor) boxes

**The question.** One window position, one window shape. But cars and buses are wide, people are tall, and two objects can share a window with one occluding the other.

The course's motivating failure is memorable: build a "people finder" with a tall, narrow window, and it fails on a person **lying down**. A single box shape per position is a bet on the aspect ratio of the thing you're looking for.

SSD's answer: at every cell of every feature map, place **several default boxes of different shapes**, and ask the head, for each one, "is there an object in roughly this box, and what correction does it need?" The course's slide shows four — small square, big square, tall rectangle, wide rectangle — and walks the lying-down person: the tall box says no, both squares say no, the **wide box says yes**. Crucially, default boxes exist at *every* scale, so the choice of shape and the choice of size are independent: the paper's figure catches a large dog on a coarse map with one shape and a small cat on a fine map with another.

The real SSD is a little richer than the four shapes on the slide. Aspect ratios are $\{1, 2, 3, \tfrac{1}{2}, \tfrac{1}{3}\}$ plus one extra square of size $\sqrt{s_k s_{k+1}}$, giving **4 or 6 boxes per location** depending on the map (the 4-box maps drop the 3 and 1/3 ratios) — which is where the per-cell counts in the snippet above come from. Anchor scales $s_k$ are interpolated roughly linearly from ~0.2 on the finest map to ~0.9 on the coarsest.

Two mechanisms the course doesn't cover but that you need to read any detector's training code:

- **Matching.** During training, each ground-truth box is assigned to its best-overlapping default box, *plus* every default box with IoU > 0.5 against it. Those become positives, with their offset targets computed by the formula in §2. Everything else is a negative, and its only target is "background".
- **Decoding.** At inference the predicted offsets are inverted against each anchor to recover absolute boxes. This is where the log-space width and height get exponentiated, and where a sign or ordering bug produces boxes that are plausible-looking but systematically wrong — a classic silent failure.

## 6. From 8732 boxes to a handful: IoU and NMS

**The question.** The network fires a dozen overlapping boxes on one dog. How do you decide which single box is "the detection", and how do you score it against the truth?

**IoU (= Jaccard = intersection over union).** The course warms up on sets of words: "The quick brown fox jumps over the lazy dog" versus "The quick red fox jumps over the lazy turtle" share *the, quick, fox, jumps, over, lazy* → intersection 6; each contributes 2 unique words → union 10; Jaccard = 0.6. (Small slip in the lecture: it says eight words per sentence; there are nine, eight of them unique. The arithmetic is right.) For boxes, swap word counts for areas:

$$\mathrm{IoU} = \frac{\mathrm{area}(B_{\text{gt}} \cap B_{\text{pred}})}{\mathrm{area}(B_{\text{gt}} \cup B_{\text{pred}})}$$

0 means disjoint, 1 means identical. Threshold it to decide whether a predicted box "found" a ground-truth box — the course suggests 0.6 or 0.7; the field's convention is 0.5 for a lenient match (see §9).

**Non-max suppression.** Many overlapping boxes on one object is a *good* sign — the object is recognizable from several of its parts — but only one may be output. The procedure: sort boxes by score; take the highest, keep it; discard every remaining box whose IoU against it exceeds a threshold, since high IoU means "same object"; repeat with the highest-scoring survivor until nothing is left. This is SSD's final block, and it has no learned parameters.

Both are about fifteen lines. The snippet implements IoU and NMS from scratch on four boxes — three piled on the same object plus one off on its own — and checks the result against `torchvision.ops.nms`:

```python
import numpy as np, torch
from torchvision.ops import nms                              # reference implementation to check against
def iou(a, b):                                               # boxes as (x1, y1, x2, y2)
    x1, y1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    x2, y2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)                 # 0 when the boxes miss each other
    area = lambda c: (c[2] - c[0]) * (c[3] - c[1])
    return inter / (area(a) + area(b) - inter)                # intersection over union
boxes = np.array([[10, 10, 60, 60], [12, 12, 62, 62], [14, 8, 58, 64], [200, 200, 260, 260]], float)
scores = np.array([0.95, 0.90, 0.60, 0.80])
print("IoU of the two near-duplicates:", round(iou(boxes[0], boxes[1]), 3))
def my_nms(boxes, scores, thr=0.5):                           # keep the best box, drop everything overlapping it
    keep, order = [], list(np.argsort(-scores))
    while order:
        i = int(order.pop(0)); keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return keep
mine = my_nms(boxes, scores)
ref = nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist()
print("my NMS keeps:", mine, " torchvision keeps:", ref, " match:", sorted(mine) == sorted(ref))
# Expected output:
# IoU of the two near-duplicates: 0.855
# my NMS keeps: [0, 3]  torchvision keeps: [0, 3]  match: True
```

The two near-duplicate boxes score **IoU 0.855**, comfortably above any sensible threshold, so NMS keeps box 0 (score 0.95) and suppresses boxes 1 and 2, which are the same object seen slightly differently. Box 3 is spatially disjoint, so its IoU against the winner is 0, and it survives. Final answer: boxes **0 and 3** — matching torchvision exactly. That match is the useful part: a hand-rolled NMS and the production kernel agree, so the algorithm really is this small.

**Two things the course's version gets incomplete.** First, it is presented as a single pass — take the best, delete the overlaps, done — when real NMS **iterates** over the survivors, as the loop in the snippet does; with three objects in a row, one pass keeps only one of them. Second, NMS is normally run **per class**. This is not a detail: the course's `jungle.jpeg` failure, where one animal ends up with two boxes labelled "bear" and "dog", is exactly what class-wise NMS does by design, and the lecture presents it as a model failure without making that connection. (The underlying problem there is different again — see §10.) Also unmentioned: the IoU threshold is a tunable knob (~0.5 is typical), **Soft-NMS** decays scores instead of deleting boxes, class-agnostic NMS exists for when you'd rather not double-label, and DETR-family detectors remove NMS altogether by training with a set-matching loss.

**Confidence thresholding** is the other half of the filter, and the course leans on it constantly — "two classes passed the threshold", "the score is pretty low" — but **never states a number**. Treat it as the knob it is: it trades precision against recall directly, and the right value depends entirely on what a false positive costs you. Note also that mAP (§9) is computed by *sweeping* this threshold, so the number you pick for display has no effect on the reported metric.

## 7. Class imbalance and focal loss — background the course omits

**Flagging provenance first.** Nothing in this section comes from the course. The section's title names RetinaNet, but RetinaNet, focal loss, feature pyramid networks and class imbalance are **absent from the lectures entirely** — there is no discussion of how a detector is trained at all, because no detector is trained. One short paragraph here, because it is the most load-bearing omission: without it, SSD looks like the end of the story rather than the step before the obvious fix.

**The question.** 8732 candidate boxes, and an image containing three objects. What are the other 8729 doing to your loss?

They are all negatives, and they dominate. Tens of thousands of easy background boxes each contribute a small loss — the model is already confident they are background — but their gradients sum to far more than the handful of positives contribute, so the detector learns "background" extremely well and barely learns to detect. SSD (Liu et al. 2016) handles this with **hard negative mining**: sort the negatives by confidence loss and keep only the worst, so negatives outnumber positives at most 3:1. RetinaNet (Lin et al. 2017) replaces that heuristic with a loss function. **Focal loss** scales cross-entropy by $(1 - p_t)^{\gamma}$, where $p_t$ is the predicted probability of the correct class:

$$\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log p_t$$

At $\gamma = 0$ it is ordinary cross-entropy; at the paper's $\gamma = 2$ a well-classified box at $p_t = 0.99$ has its loss scaled by $10^{-4}$ while a hard example at $p_t = 0.1$ is scaled by 0.81 and barely touched. The easy majority is suppressed smoothly rather than discarded by a sort, and no mining stage is needed — which is what let a one-stage detector match two-stage accuracy for the first time (§8). Details in the two papers, linked under References.

## 8. One-stage vs two-stage

**The question.** Why are there two families, and which do you reach for?

**Two-stage** (R-CNN → Fast R-CNN → **Faster R-CNN**): stage one proposes a few hundred or thousand regions that might contain *something*, class-agnostically (in Faster R-CNN, by a small Region Proposal Network sharing the backbone). Stage two crops features for each proposal and classifies and refines it. The second stage sees a balanced, already-plausible set of candidates, which historically made it the more accurate family — at the cost of a per-proposal loop that is slow and awkward to deploy.

**One-stage** (**SSD**, YOLO, RetinaNet, FCOS): one pass, dense predictions over all anchors or all positions, no proposal step. Faster, simpler to export, and — until focal loss — less accurate, for exactly the reason in §7.

The course's summary lecture claims SSD gives you both speed *and* accuracy rather than trading one for the other, with the self-driving motivation that a car detecting 90% of traffic lights is unacceptable. Against the baselines in the paper's own 2015–16 chart (YOLOv1, R-CNN, Fast/Faster R-CNN) that was true. **Stated undated, it reads as a claim about the present, and it has not held for years.** RetinaNet, FCOS, EfficientDet, the YOLOv5–v11 line and the DETR/DINO family all sit above and to the right of SSD on that same speed-accuracy plot. The distinction itself has also softened: modern one-stage detectors are often anchor-free (predict box extents directly from each position), and DETR-style detectors replace both anchors and NMS with a transformer decoder and a bipartite set-matching loss.

## 9. mAP: the metric the course plots but never defines

**The question.** The section opens with a speed-versus-accuracy chart whose y-axis is mAP. What is on that axis?

Start from precision and recall ([classification-models](../classical-ml/classification-models.md)). In detection, a predicted box is a **true positive** if its IoU with an unmatched ground-truth box of the same class clears a threshold; a second box on an already-matched object is a **false positive**, not a duplicate true positive (this is why NMS matters to the score, not just to the picture). Every ground-truth box with no match is a **false negative**.

Now sweep the confidence threshold from high to low. Each setting gives one (recall, precision) pair; together they trace a **precision-recall curve** per class. **Average Precision (AP)** is the area under that curve — one number summarising the whole trade-off for one class. **mAP** is AP averaged over classes.

The remaining question is which IoU threshold defines a true positive. Pascal VOC used 0.5 (`AP50`), which is lenient: a sloppy box counts. The **COCO** convention, which is what modern papers report as "mAP" or "AP", averages AP over ten thresholds, IoU = 0.50, 0.55, …, 0.95 (`AP@[.50:.95]`), so box tightness is part of the score. COCO also reports `AP50`, `AP75` and separate figures for small, medium and large objects — the last of which is the honest way to see that detectors are much worse on small objects than one aggregate number suggests.

Two consequences worth internalising. One: **mAP numbers are only comparable within the same protocol** — an AP50 of 0.74 and a COCO AP of 0.41 can be the same model. Two: **the course never computes any of this.** No mAP, no precision, no recall, no IoU measurement appears anywhere in section 8 or 12 — only eyeballed screenshots. Evaluation is the single biggest hole in both sections.

## 10. What the course actually runs (section 8)

**The question.** After seven conceptual lectures, what is in the notebook?

Someone else's pretrained detector, called for inference. Specifically: `git clone` of the TensorFlow `models` repository, a `protoc` run inside `models/research`, install via `object_detection`'s `setup.py`, download a checkpoint from the **TF2 Detection Model Zoo**, `tf.saved_model.load(...)`, and call it on four JPEGs and then on three MP4s frame by frame via `imageio`. The whole flow is two notebooks in Google Colab, TensorFlow 2 / `tf.keras` throughout (no PyTorch anywhere in either section).

What the course states, and only what it states:

- The zoo lists **CenterNet, EfficientDet, SSD and Faster R-CNN**; SSD comes with MobileNet or ResNet backbones. The instructor picks one **arbitrarily and never says which**, so the checkpoint, its input resolution and its published mAP are all unstated. All zoo models are trained on **COCO 2017**, so one label map serves them all; `create_category_index_from_label_map(..., use_display_name=True)` turns it into a dict from class id to readable name.
- Loading the saved model takes about **30 seconds**. The returned object is a *callable function*, not a model object — a detail that confuses people reading this code.
- The detection call returns `num_detections`, `detection_classes`, `detection_boxes`, `detection_scores`; you drop the batch dimension from each, cast classes to int, and hand them to `visualize_boxes_and_labels_on_image_array`, which draws in place.
- Video throughput: about **1 FPS on the first video** (model warm-up and variable initialization) and **over 3 FPS on the second and third**, on a Colab GPU. **No confidence threshold value, no mAP, no accuracy number, and no timing on CPU-versus-GPU is given.**
- Qualitative results, as reported: `intersection.jpeg` finds all the cars, the bicycle and the stop signs, explicitly including both large and small instances of each — multi-scale detection visibly working. `traffic.jpeg` finds cars, trucks, a bicycle, a person and some potted plants. `walkingdog.jpeg` finds both people and the dog. `jungle.jpeg` **fails**: each animal gets two boxes from two different classes, one labelled both "bear" and "dog", both wrong. The `safari` video labels a lion as dog, cow, horse, elephant and even person, but never lion, and misses a second smaller lion; the `catdog` video calls a box a cup and a floor toy a person, but correctly identifies a refrigerator from its bottom half alone.

Two honest readings of those failures. The jungle and safari errors are **not** mysterious model weakness: COCO's 80 classes contain bear, cat, dog, cow, horse, sheep, elephant, zebra and giraffe but **no lion and no tiger**. A closed-vocabulary classifier asked about an out-of-vocabulary animal must return something wrong — that is the expected behaviour, and the lecture sets "is lion in COCO?" as an exercise without answering it. The modern answer to that class of problem is an **open-vocabulary detector** (OWL-ViT, Grounding DINO), which takes the class names as text input at inference time. Separately, 1–3 FPS is not "real time", which undercuts the section's framing; the gap is per-frame Python and `imageio` overhead at batch size 1 (the instructor names batching as the fix), and a MobileNet-SSD exported to TensorRT or ONNX and run batched is orders of magnitude faster.

**The section ends by admitting SSD is never implemented.** The instructor's rule is that course code stays in "the low 100s of lines", and SSD is mostly non-ML plumbing — anchor generation, matching, encoding, decoding, NMS — which is a fair call, but it means you finish the section having read about SSD and executed only `model(image)`.

**Do not start from this code in 2026.** `models/research/object_detection` is effectively unmaintained, the `protoc` + `setup.py` install is brittle, and the TF2 zoo checkpoints date to 2020. The instructor's own warning — pin a working version rather than chasing the latest — is sound advice aimed at the wrong problem here, because the whole toolchain has been superseded. Current equivalents, in rough order of how little work they are:

- **Ultralytics YOLO** — `model = YOLO("yolo11n.pt"); model(img)`, plus training and export in the same package. The fastest path to a working detector; check the AGPL licence terms before shipping it.
- **`torchvision.models.detection`** — `ssdlite320_mobilenet_v3_large`, `retinanet_resnet50_fpn_v2`, `fasterrcnn_resnet50_fpn_v2`, all with pretrained COCO weights. Closest to the concepts in this note (it literally has an `AnchorGenerator` and `torchvision.ops.nms` you can read), and the best choice for *learning* the pipeline.
- **Hugging Face `transformers`** — DETR, RT-DETR for transformer detectors; OWL-ViT and Grounding DINO for open-vocabulary and text-prompted detection.
- **MMDetection / Detectron2** — when you need many architectures under one training harness.

The *concepts* in lectures 58–63 and 67 (localization, variable-length output, sliding-window-as-convolution, scale, shape, IoU, NMS) remain accurate and are the reason to watch the section.

## 11. The localization project (section 12)

**The question.** If detection is too much plumbing to implement, what *can* you build end to end? A single-object localizer — and build it in seven steps, each adding one difficulty.

The data is **generated**, not collected, by a Python generator that yields `(X, Y)` batches forever. That is the clever move: the ground-truth box is known exactly because the code *placed* the object, so there is no annotation work and effectively unlimited data.

| Exp. | Image | New difficulty | Outputs |
|---|---|---|---|
| 1 | White rectangle on black, 100×100 | the baseline | 4 (box) |
| 2 | Charmander sprite on black | a real object, fixed size; the alpha channel | 4 |
| 3 | + random scale, U(0.5, 1.5) | unknown object size | 4 |
| 4 | + random horizontal flip | appearance variation | 4 |
| 5 | + random 200×200 crop of a real photo | background clutter, compositing | 4 |
| 6 | object present only 50% of the time | "is there anything here at all" | 5 (box + presence) |
| 7 | three Pokemon, present 75% of the time | which object is it | 8 (box + 3 classes + presence) |

**The model, all seven times.** VGG16 with `input_shape=(100,100,3)` (three channels even for the greyscale experiments, because VGG demands colour), `include_top=False`, ImageNet weights → `Flatten()` → head. In experiment 1 the head is a single `Dense(4, activation='sigmoid')`: four numbers in [0,1], exactly the normalized $(\text{row}_0, \text{col}_0, h, w)$ of §2. By experiment 7 it is three parallel heads off the same flattened features, joined by `Concatenate`:

- `Dense(4, 'sigmoid')` → the box,
- `Dense(3, 'softmax')` → which Pokemon,
- `Dense(1, 'sigmoid')` → is anything there,

giving an 8-vector sliced as `[:4]`, `[4:7]`, `[7]`. The stated reason for parallel heads rather than one wide `Dense(8)` is simply that you cannot apply a sigmoid to five of a layer's outputs and a softmax to the other three — the activation is a property of the layer. This is the functional-API pattern you will meet again in any multi-task network.

### Why the three outputs need three different losses

This is the real lesson of the project, and it generalises far beyond boxes.

**Different output types have different natural losses.** A probability-like scalar wants binary cross-entropy. A one-of-$K$ choice wants categorical cross-entropy on a softmax. Continuous coordinates want a regression loss — squared error, or (the course notes, correctly) **Huber**, which behaves like squared error near zero and like absolute error far away, so one badly-placed box doesn't dominate the batch.

**Gradient descent needs one scalar, so you add them — with weights.** The course's experiment-6 loss, with predictions $P_1..P_5$ and targets $T_1..T_5$:

$$\mathcal{L} = w_{\text{box}} \cdot T_5 \cdot \mathrm{BCE}(T_{1:4}, P_{1:4}) \;+\; w_{\text{cls}} \cdot \mathrm{BCE}(T_5, P_5)$$

with **$w_{\text{box}} = 2$ and $w_{\text{cls}} = 0.5$** (the weights for the three-term experiment-7 loss are not stated). Those weights are hyperparameters like any other, found by trial and error: raise the box weight if localization lags, raise the classification weight if presence lags. The same weighted-sum-of-incommensurable-losses pattern shows up in neural style transfer (content versus style) and in essentially every multi-task model.

**The $T_5$ gate is the subtle part.** When no object is present, $T_1..T_4$ are *undefined* — there is no box to regress towards. Multiplying the box term by the presence target $T_5 \in \{0, 1\}$ makes that term contribute exactly zero loss and exactly zero gradient on empty images. Without the gate you would be training the box head towards whatever garbage the generator left in those four slots. In experiment 7 both the box term and the class term are gated this way (an absent object has no class either, unless you add an explicit background class — which the course notes is the alternative design: fold "no object" in as class $K+1$ and drop the presence output entirely). **Every detector does this**, and it is the same mechanism as SSD's "negatives only contribute a classification loss".

### The sigmoid + BCE choice, and why its loss value looks wrong

The project puts a **sigmoid on the four coordinates and trains them with binary cross-entropy**, rather than regressing them with MSE. The justification offered is that the targets live in [0,1] so, "stretching the imagination", you can treat them as probabilities and write $\mathrm{BCE} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$ — which is well-defined for fractional $y$: at $y = 0.6$ *both* terms are active and the formula still works. An optional aside connects cross-entropy to KL divergence, $H(P,Q) = D_{KL}(P\|Q) + H(P)$, so the two differ by a constant and share gradients.

Two corrections, because this is where a reader gets misled.

**The probability analogy is loose, not a derivation.** $(\text{row}_0, \text{col}_0, h, w)$ are not a distribution — nothing normalizes them — so the KL framing is a mnemonic. Sigmoid+BCE works here for a sounder reason: it is a well-conditioned surrogate for a *bounded* regression target (a per-coordinate Bernoulli log-likelihood), its gradient does not vanish when the prediction is badly wrong the way sigmoid+MSE does, and the sigmoid enforces the [0,1] constraint for free. MSE on the same sigmoid outputs also works, and the course says so.

**The loss value is not a quality metric, and that is why 0.6 is unexplainable.** The course reports a printed loss "around 0.6" and says it "isn't really relative to anything" — and sets as an exercise, never answered, "what is the optimal $\hat{y}$ when $y = 0.6$?" The answer: BCE over a fractional target is minimized at $\hat{y} = y$, but its minimum *value* is the entropy $H(y) = -[y\log y + (1-y)\log(1-y)]$, **not zero**. For targets around 0.3–0.5 that floor is roughly 0.6–0.69 nats, which is exactly the number being observed. A perfect model would still print ~0.6. Worse, mixing this offset loss with a genuine classification BCE in a weighted sum means the loss weights are partly absorbing a constant. **Report IoU** (§6) if you want to know how good the boxes are. And in 2026, prefer **Huber/smooth-L1 or an IoU-family loss (GIoU/DIoU/CIoU)** for box regression — an IoU loss optimizes the thing you actually measure and stays meaningful when the boxes don't overlap at all.

### Compositing: the part that is harder than the network

Experiment 5 is where the project earns its keep, and the lesson has nothing to do with neural networks. Pasting an RGBA sprite onto a background crop with `bg[rows, cols] = sprite[:, :, :3]` **punches a black rectangle**: the sprite's transparent pixels are zero in all four channels, and once you drop the alpha channel those zeros read as black. The course's fix:

1. `mask = (alpha == 0)` — a 2-D boolean, true where the background should survive.
2. `np.expand_dims(mask, -1)` → shape `H×W×1`, so broadcasting applies it to all three colour channels.
3. `bg_slice = bg_slice * mask` → zeros exactly in the object's silhouette.
4. `bg_slice += sprite[:, :, :3]` → the sprite's zeros and the background's zeros are complementary, so addition composites them.

That is a genuinely useful trick and the clearest diagram in the section. **It also has a bug once experiment 3's resizing is in play**, which the course does not flag: `skimage.transform.resize` interpolates and anti-aliases, so a resized alpha channel is no longer strictly $\{0, 255\}$. Fringe pixels fail `alpha == 0`, so the background is zeroed there *and* the sprite's semi-transparent RGB is added on top — a bright halo around every object. The correct operation is **alpha blending**: with $a = \alpha/255$, $\text{out} = \text{rgb} \cdot a + \text{bg} \cdot (1-a)$, which degrades gracefully for any fractional alpha. (Related trap in the same code: `resize` with the default `preserve_range=False` returns floats in [0,1] while a uint8 background is 0–255.) This is inferred from the code as described, not verified against the notebook — but it is worth checking before reusing the generator.

### Other specifics worth knowing, and what isn't stated

- **Pixel range is [0,1]**, not 0–255, and VGG's own `preprocess_input` is deliberately **skipped**: the instructor says using it is "technically correct" but "when I did this, the results actually got worse".
- **Freezing is never mentioned**, so all of VGG16 trains. Combined with [0,1] inputs instead of `preprocess_input`'s mean-subtracted BGR, and Adam at its 1e-3 default, the ImageNet weights are being substantially overwritten. These notebooks are **ImageNet-initialized fine-tuning, not frozen-extractor transfer learning** — which plausibly explains both the forced 10× learning-rate drop in experiment 3 and why correct preprocessing "made it worse". The usual recipe is the opposite order: `base.trainable = False`, train the head, then unfreeze and continue at ~1e-5.
- `n_batches_per_epoch = 50`, chosen "for no particular reason" — and legitimately arbitrary, because an infinite generator makes 50 batches × 1 epoch equivalent to 10 × 5. Batch size is 32 in the outline and 64 in experiment 7. **The epoch count is never stated for any experiment.**
- Experiment 7's presence probability is **75%**, reasoned as: with "no object" treated as a fourth outcome alongside three Pokemon, each outcome gets ~25%, so P(object) = 3/4. A sanity check on one 64-sample batch gave **33% no object, 23% Charmander, 25% Bulbasaur, 19% Squirtle** against a 25% target — the spread is just small-batch noise.
- **Stated results are qualitative only.** Experiments 1–6: boxes and presence "look good" / "pretty much always finds Charmander". Experiment 7: classification and presence correct on sampled runs, but **"the boxes are a little more off than before"**, with an invitation to tune the loss weights. **No accuracy, IoU or comparable loss number is reported for any experiment in section 12.** Given §6, measuring mean IoU on a generated validation batch would take about five lines and would make every one of these claims checkable.
- `model.fit_generator` is used throughout and is **dead API** — deprecated in TF 2.1 and removed in Keras 3. Use `model.fit(gen, steps_per_epoch=50, epochs=...)`, or better `tf.data.Dataset.from_generator(...).prefetch(AUTOTUNE)`. Likewise `tf.keras.preprocessing.image.load_img` is legacy (now `tf.keras.utils.load_img`), and `scipy.misc.imread/imsave` — which the course correctly flags as removed (SciPy 1.3) — is replaced by `imageio` / `scikit-image`.
- A noted **inconsistency across sections**: lecture 59 prescribes four linear regression outputs trained with squared error or Huber, while section 12 uses sigmoid + BCE. Both are presented as *the* answer with no cross-reference. Treat them as two valid options: Huber/IoU-loss for unbounded or production box regression, sigmoid+BCE when the targets are bounded in [0,1] and you want the constraint enforced by construction.
- The closing moral, "all data is the same", oversells what transfers. A localizer trained on sprites composited over 15 backgrounds will not work on real photos: hard mask edges, no shadows, no lighting match and no motion blur are a trivially learnable shortcut, and the model will take it. The *code structure* transfers; the weights do not.

## Trade-offs

- **Anchors vs anchor-free.** Default boxes give the network an easy, near-zero regression target and explicit control over the shapes you cover, at the cost of several hyperparameters (scales, ratios, matching IoU) and a huge imbalanced candidate set. Anchor-free heads (FCOS, CenterNet) and set-prediction heads (DETR) delete those hyperparameters and pay in slower convergence or more compute.
- **Dense predictions vs proposals.** One-stage is faster and simpler to export; two-stage sees a balanced candidate set and historically localized better. Focal loss largely closed the accuracy gap (§7).
- **More anchors.** Better coverage and recall, but prediction count — and the background:foreground ratio — grows with it.
- **Confidence threshold.** High threshold = precise, misses things; low = catches more, floods you with false positives. There is no neutral setting, only the one your application's cost ratio implies.
- **BCE vs Huber vs IoU loss on boxes.** Sigmoid+BCE enforces [0,1] by construction but has a non-zero loss floor you can't read; Huber is robust and interpretable but doesn't optimize IoU; GIoU/DIoU optimize the metric directly and need care when boxes are disjoint.
- **Generated vs collected data.** Composited sprites give exact labels and unlimited volume for free, and a domain gap you cannot close by training longer.
- **Implementing a detector vs using one.** Writing SSD teaches you anchors, matching and decoding; using `torchvision` or Ultralytics gets you a working model today. The course chooses the second and is honest about it — just don't mistake having watched section 8 for being able to build SSD.

## Common Pitfalls

- **Expecting the network to emit a variable number of boxes.** It never does. The output tensor is fixed-size; the variable-length result is produced by thresholding and NMS outside the model (§1).
- **Predicting absolute coordinates from a convolutional head.** A convolution doesn't know where it is, so shared weights cannot encode position. Predict offsets relative to each anchor, with width and height in log space (§2).
- **Swapping row and column.** Images and detectors are (row, col) with the origin top-left; Matplotlib, OpenCV and most geometry are (x, y). `Rectangle` takes `(x, y)` then `width, height` — the course's own helper has to reorder all four numbers. Plot your boxes on your data before you train anything.
- **Running NMS once, class-agnostically.** Real NMS iterates over survivors, and is normally applied per class. One pass with three objects in a row keeps one of them (§6).
- **Reading the loss value as model quality.** BCE on a fractional target bottoms out at $H(y) \neq 0$, so ~0.6 can be a perfect fit. Report IoU or mAP, never the training loss (§11).
- **Letting empty images train the box head.** Gate the regression term by the presence target (or add a background class) so absent objects produce no box gradient (§11).
- **Pasting an RGBA sprite with `=`.** Transparent pixels are zeros in all channels and read as black. Mask-and-add works for binary alpha; after any resize you need real alpha blending (§11).
- **Fine-tuning a pretrained backbone without preprocessing or freezing.** Feeding [0,1] pixels to a network that expects mean-subtracted BGR, with every layer trainable at Adam's default LR, overwrites the ImageNet features you came for (§11).
- **Comparing mAP numbers across protocols.** VOC AP50 and COCO AP@[.50:.95] are different scales. Always state which (§9).
- **Blaming the model for out-of-vocabulary classes.** A COCO detector has no lion class and no tiger class; it *must* answer wrong. Use an open-vocabulary detector or retrain with your classes (§10).
- **Reading "SSD is best on both speed and accuracy" as current.** It was true against 2015–16 baselines and has not been for years (§8).
- **Treating 1–3 FPS as real-time.** That figure is per-frame Python overhead at batch size 1, not the model's ceiling; batch the frames or export to ONNX/TensorRT (§10).
- **Starting new work from `models/research/object_detection`.** Unmaintained, brittle install, 2020-era checkpoints (§10).

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| One object per image, box needed | Localization: classifier + 4 sigmoid regression outputs | A full detector | A few extra units on a pretrained backbone; trains like a classifier |
| Zero-or-more objects | A real detector (anchors + NMS, or DETR-style) | Detect-erase-repeat loops | Many passes per image, and erasing destroys context |
| Objects of wildly different sizes | Heads on several feature maps, ideally with FPN | One head on the last layer | The deepest map has no resolution left for small objects (§4) |
| Objects of wildly different shapes | Several aspect ratios per cell | One square anchor | A tall anchor misses a person lying down (§5) |
| Training swamped by background boxes | Focal loss (or hard negative mining 3:1) | Plain cross-entropy over all anchors | 8729 negatives out-gradient 3 positives (§7) |
| Many boxes on one object | Per-class, iterative NMS | One suppression pass | One pass collapses distinct nearby objects (§6) |
| Reporting detector quality | COCO mAP, with AP50/AP75 and size breakdown | Training loss, or eyeballed screenshots | A loss of 0.6 can be optimal; screenshots don't aggregate (§9, §11) |
| Box regression loss, new code | Huber/smooth-L1, or GIoU/DIoU | Sigmoid + BCE on coordinates | Interpretable, and IoU losses optimize the reported metric (§11) |
| Need a detector working today | Ultralytics YOLO, or `torchvision.models.detection` | TF Object Detection API | Maintained, current weights, one-line inference (§10) |
| Learning the pipeline by reading code | `torchvision.models.detection` + `torchvision.ops` | Ultralytics, or writing SSD from scratch | Readable `AnchorGenerator`, `box_iou`, `nms` you can step through |
| Classes not in COCO, no labels to spare | Open-vocabulary detector (OWL-ViT, Grounding DINO) | Fine-tuning on 20 images | Takes class names as text at inference time (§10) |
| Fine-tuning a backbone on a small dataset | Freeze, train head, then unfreeze at ~1e-5, with the backbone's own preprocessing | Everything trainable at 1e-3 with [0,1] pixels | Otherwise the pretrained features are overwritten (§11) |

## References

- [Liu et al. — SSD: Single Shot MultiBox Detector (2015)](https://arxiv.org/abs/1512.02325) — the course's paper: default boxes, multi-scale maps, the matching strategy, hard negative mining, and the speed-accuracy chart shown in the intro.
- [Lin et al. — Focal Loss for Dense Object Detection (2017)](https://arxiv.org/abs/1708.02002) — RetinaNet. The section the course's title promises and never delivers (§7).
- [Lin et al. — Feature Pyramid Networks for Object Detection (2016)](https://arxiv.org/abs/1612.03144) — the top-down path that fixes small-object detection on multi-scale heads (§4).
- [Ren et al. — Faster R-CNN (2015)](https://arxiv.org/abs/1506.01497) — the two-stage reference point (§8).
- [Redmon et al. — You Only Look Once (2015)](https://arxiv.org/abs/1506.02640) — the other one-stage family.
- [Sermanet et al. — OverFeat (2013)](https://arxiv.org/abs/1312.6229) — the original "dense layers as convolutions" sliding-window argument of §3.
- [Carion et al. — End-to-End Object Detection with Transformers (2020)](https://arxiv.org/abs/2005.12872) — DETR: no anchors, no NMS, set-matching loss.
- [Zhao et al. — DETRs Beat YOLOs on Real-time Object Detection (2023)](https://arxiv.org/abs/2304.08069) — RT-DETR.
- [Tan et al. — EfficientDet (2019)](https://arxiv.org/abs/1911.09070) · [Rezatofighi et al. — Generalized IoU loss (2019)](https://arxiv.org/abs/1902.09630) · [Bodla et al. — Soft-NMS (2017)](https://arxiv.org/abs/1704.04503)
- [Minderer et al. — Simple Open-Vocabulary Object Detection with Vision Transformers (2022)](https://arxiv.org/abs/2205.06230) (OWL-ViT) · [Liu et al. — Grounding DINO (2023)](https://arxiv.org/abs/2303.05499)
- [COCO detection evaluation](https://cocodataset.org/#detection-eval) — the exact definition of AP@[.50:.95], AP50, AP75 and the small/medium/large breakdown (§9).
- [`torchvision.ops.nms`](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html) — the reference NMS the §6 snippet checks itself against · [torchvision detection models](https://docs.pytorch.org/vision/stable/models.html#object-detection)
- [Hugging Face — object detection task guide](https://huggingface.co/docs/transformers/tasks/object_detection) · [Ultralytics docs](https://docs.ultralytics.com/)
- [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) — what the course downloads from; kept here for reading the notebooks, not for new work.
- [lazyprogrammer/machine_learning_examples](https://github.com/lazyprogrammer/machine_learning_examples) — the instructor's public code.

## See Also

- [convolutional-neural-networks](convolutional-neural-networks.md) — convolution, stride, feature maps and output-size arithmetic, all assumed here.
- [transfer-learning-and-cnn-architectures](transfer-learning-and-cnn-architectures.md) — pretrained backbones, ResNet, and why deeper layers respond to bigger structures (the premise of §4).
- [classification-models](../classical-ml/classification-models.md) — precision, recall and the confusion matrix that AP and mAP are built from (§9).
- [artificial-neural-networks](artificial-neural-networks.md) — the dense heads, sigmoid/softmax outputs and cross-entropy that every detection head reuses.
- [rl-foundations-mc-td](../reinforcement-learning/rl-foundations-mc-td.md) §5 — the instructor's shared appendix and setup advice, covered once there.
