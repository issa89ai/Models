# Naive Bayes — What We Learned

## Status of the original file

`naiveBayes.py` is genuine Python 2 code (`print "..."` without parentheses,
a `SyntaxError` in Python 3) from the UC Berkeley Pacman AI framework
(CS566), tightly coupled to that framework's data format (`util.Counter`,
`classificationMethod`) and not runnable standalone. The original folder also
contained MIRA and Perceptron-Pacman variants plus the surrounding framework
files (`classificationAgents.py`, `dataClassifier.py`, `classificationMethod.py`,
`mira.py`, `perceptron.py`, `perceptron_pacman.py`) — these were removed at the
user's request to keep this folder focused on Naive Bayes specifically.

## The concept

Built on Bayes' theorem: `P(label | features) ∝ P(label) × P(features | label)`.
Computing the true joint likelihood of all features together would need
enormous amounts of data, so Naive Bayes makes a simplifying ("naive")
assumption: treat every feature as independent given the class, turning the
likelihood into a simple product of per-feature probabilities. The original
file's math (Laplace smoothing with a `+k`/`+2k` term, log-joint-probability
classification) was reimplemented standalone in `naive_bayes_demo.py`, using
Iris data binarized around each feature's median to match the original's
binary-feature assumption.

## Experiment 1: from-scratch binarized Naive Bayes

```
train_acc=0.7333, test_acc=0.7667
```

The weakest result of any model tried in this repo so far (vs. 90-100% for
KNN, Decision Trees, SVM, Softmax on the same dataset).

## Experiment 2: sklearn GaussianNB (continuous features, no binarization)

```
train_acc=0.9500, test_acc=1.0000
```

Nearly matches every other model — using the *same* naive independence
assumption, but modeling each feature as a continuous Gaussian per class
instead of forcing it into a single active/inactive bit.

## The key finding

**Binarization, not the independence assumption, was the dominant cause of
the poor first result.** Petal length and petal width genuinely are
correlated in Iris (visible in every decision boundary plot in this repo),
technically violating Naive Bayes's core assumption — yet the algorithm still
performed nearly as well as everything else once given continuous features.
This is a broader, generalizable lesson: Naive Bayes is often blamed for
underperforming due to its "naive" independence assumption, when in practice
feature representation (how much information survives into the features you
feed it) is usually the bigger risk.

## Files in this folder

- `naiveBayes.py` — original CS566 coursework file (Python 2, Pacman-framework-coupled, reference only)
- `naive_bayes_demo.py` — standalone reimplementation of the same smoothing/log-joint math, run on binarized Iris
- `test_gaussian_nb.py` — sklearn GaussianNB comparison on continuous (non-binarized) Iris
