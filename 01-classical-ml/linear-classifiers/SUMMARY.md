# Linear Classifiers — What We Learned

## The common framework

All four models here compute a **linear score**: `score = X · W (+ b)`, a weighted
sum of the input features. What differs between them is entirely the **loss
function** used to train the weights, and (for Perceptron) whether there's a loss
function at all. None of them can draw a curved or blocky decision boundary —
only straight lines/hyperplanes — which is the shared limitation explored below.

## Status of the original coursework files

- **`Perceptron.py`** and **`Softmax.py`** were empty stubs (docstrings only, no
  actual logic — `predict()` even referenced an undefined `pred` variable). We
  wrote complete implementations from scratch to demonstrate these
  (`perceptron_demo.py`, `softmax_demo.py`), rather than editing the originals.
- **`SVM.py`** and **`LogisticRegression.py`** were fully implemented, but both had
  `3072` (the pixel count of a flattened 32×32×3 CIFAR-10 image) hardcoded as the
  input dimension inside `train()` — meaning they only worked on the exact
  CIFAR-10 image-classification assignment they were written for. We fixed both
  to use the actual input shape dynamically (`X_train.shape[1]` /
  `X_train.shape[0]`) so they run on any dataset. This is the one actual code
  edit made to original files in this folder.

**Important caveat:** switching from CIFAR-10 (the original hard image
classification target) to Iris (used here for clear, fast demonstration) doesn't
change whether the code is correct — the math is identical regardless of
dataset. But it completely changes what the accuracy number *means*. This exact
Logistic Regression code would likely only reach ~35-40% on the real CIFAR-10
task (a well-known limitation of linear models on raw pixels), vs. the 90-98% we
see here on Iris. The numbers below are only meaningful relative to each other
(same dataset, same features) — not as a claim about how well these
implementations would perform on the original assignment's actual data.

## Perceptron — mistake-driven learning

No loss function: loop through training points one at a time, and only update
weights when a prediction is wrong (`w += learning_rate * y * x`). Guaranteed to
converge to a perfect separator *if and only if* the data is linearly separable;
otherwise it never stops making mistakes.

**Experiment** (setosa vs. versicolor — a cleanly, widely separable pair):
```
Epoch 1: 9 mistakes
Epoch 2: 0 mistakes -> converged
Test accuracy: 1.0000
```
Convergence happened exactly as guaranteed, because this particular class pair
is trivially separable (confirmed visually in earlier decision-boundary plots).

## SVM — margin/hinge loss

Trains via mini-batch stochastic gradient descent (SGD) on the hinge loss: for
each example, the correct class's score must beat every other class's score by
a margin of at least 1, or a proportional penalty is added. Loss stops caring
once the margin is satisfied — "good enough" is good enough.

**Experiment** (all 3 Iris classes, 4 features):
```
train_acc=0.9667, test_acc=1.0000
```
Never reaches 100% train accuracy — a straight-line boundary structurally cannot
separate the overlapping versicolor/virginica pair as well as, e.g., the
unrestricted decision tree could. Visualized in `svm_decision_boundary.png`:
straight diagonal boundaries (any angle), unlike the tree's axis-aligned
rectangles, with visible misclassified points exactly where the two classes
overlap.

## Softmax — probability/cross-entropy loss

Generalizes logistic regression's sigmoid to multiple classes: converts scores
to probabilities (`softmax`), then minimizes cross-entropy loss
(`-log(probability of correct class)`), which — unlike hinge loss — keeps
pushing for more confidence even after a prediction is already correct.

**Experiment** (identical setup to SVM, for direct comparison):
```
train_acc=0.9667, test_acc=1.0000  (identical to SVM)
First loss: 1.0827, Final loss: 0.3674
```
Landing on the *exact same accuracy* as SVM, despite a completely different loss
function and training dynamic, is strong evidence that the limitation is
"linear boundary vs. genuinely overlapping classes," not a quirk of either loss
function. Loss curve (`softmax_loss_curve.png`) shows a smooth, monotonic
decrease (full-batch gradient descent — no mini-batch noise like SVM's training
would have) that plateaus well before the full 1000 epochs, meaning training
converged early and the remaining epochs were unnecessary.

## Logistic Regression — binary sigmoid + cross-entropy

Binary-only special case of softmax (one sigmoid output instead of a softmax
over multiple classes). Uses a transposed data convention
(`features × samples`, not `samples × features`) inherited from the original
CIFAR-10 assignment code.

**Experiment** (versicolor vs. virginica only — the single hardest pair,
isolated from the rest of the dataset):
```
train_acc=0.9875, test_acc=0.9000
```
The biggest train/test gap of any model in this folder. Isolating just this one
overlapping pair (removing the easy setosa class entirely) exposes how genuinely
difficult it is: near-perfect on training data, but a real generalization drop
on unseen examples.

## The recurring finding: versicolor/virginica overlap

Across every model in this repo so far — Decision Tree (one impure leaf,
`Gini=0.5`), SVM (96.67% ceiling), Softmax (identical 96.67% ceiling), and
Logistic Regression isolating the pair directly (90% test accuracy) — the same
specific boundary between versicolor and virginica has been the consistent
point of difficulty. This is strong evidence the difficulty is a property of
the **data** (real biological overlap in these measurements between the two
species), not a weakness specific to any one algorithm.

## Files in this folder

- `LinearClassifier.py` — actually a full CNN on CIFAR-10 (misleadingly named;
  unrelated to the linear models below), not used in these experiments
- `Perceptron.py`, `Softmax.py` — original empty stubs from coursework
- `SVM.py`, `LogisticRegression.py` — original implementations, each with one
  hardcoded dimension (`3072`) fixed to be dynamic
- `perceptron_demo.py`, `softmax_demo.py` — complete from-scratch
  implementations we wrote, demonstrating the same concepts
- `test_svm.py`, `visualize_svm.py`, `svm_decision_boundary.png` — SVM experiment + plot
- `test_logreg.py` — Logistic Regression experiment (versicolor vs. virginica)
- `softmax_loss_curve.png` — Softmax training loss curve
