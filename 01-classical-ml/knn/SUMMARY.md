# KNN — What We Learned

## The mechanism

KNN has no real "training." `train()` just stores `X_train`/`y_train` in memory — all the actual work happens at prediction time ("lazy learning"). To classify a new point:

1. Compute the distance from the new point to every training point.
2. Sort and take the **k** closest.
3. Majority-vote their labels.

Our ported `KNN.py` computes all distances at once using the algebraic identity
`||a - b||² = ||a||² - 2(a·b) + ||b||²`, which turns an O(n·m) nested-loop distance
computation into a single matrix multiplication — same math, much faster.

We also spotted a harmless quirk in the vote-counting: each label's count starts at 0
instead of 1 (`labels_counts[label] = 0` on first sight, `+= 1` after). Every label ends
up systematically undercounted by exactly 1, but since *all* labels get the same -1
penalty, the ranking (which label has the most votes) is unaffected. It would only
matter if the code needed the true count (e.g., to report a confidence/probability),
not just the argmax.

## What k actually controls

k is a dial between sensitivity and stability — the bias-variance tradeoff:

- **Small k** (e.g. k=1): prediction depends on a single nearby point. Very reactive to
  local detail, but easily thrown off by noise/outliers ("high variance").
- **Large k**: prediction averages over a big neighborhood, smoothing out noise but
  also smoothing out real local structure ("high bias").
- **k = entire training set**: every prediction becomes identical — just "predict the
  globally most common class," ignoring the input entirely.

## Experiment: Iris dataset, k = 1, 5, 25, 75, 120 (120 training / 30 test samples)

```
k=1:   accuracy=1.0000
k=5:   accuracy=1.0000
k=25:  accuracy=1.0000
k=75:  accuracy=0.9333
k=120: accuracy=0.3000
```

Accuracy holds perfect through k=25 (Iris is a small, easy, near-linearly-separable
dataset). At k=75 (62% of the training set), accuracy starts dropping as distant,
irrelevant points dilute the real local signal. At k=120 (the *entire* training set),
every test point gets the same prediction regardless of its actual features — the
model has degenerated into "always predict the majority class," which is why accuracy
collapsed to ~0.30 (roughly the majority-class baseline for a balanced 3-class problem).

**Takeaway on scaling k:** the "right" k generally scales with dataset size. On a much
larger dataset (e.g. 10,000 samples), a bigger k can still stay meaningfully "local"
since points are packed more densely — whereas on a small dataset like this one, even a
moderately large k forces the search to reach far outside any real local neighborhood.

## Manual worked example (toy 6-point dataset)

Given `A(1,1)→🔴, B(1,2)→🔴, C(2,2)→🔴, D(8,8)→🔵, E(9,8)→🔵, F(9,9)→🔵` and a new
point `Y=(8,9)`: distances to D and F both equal 1.0 (nearest), E is ≈1.41. With k=1,
the nearest point (D or F, tied) is 🔵, so Y is classified 🔵. This confirmed the exact
mechanics of `find_dist()` + `predict()` by hand, separate from the k-tradeoff question.

## Where this shows up in the real world

Standalone KNN classifiers aren't common in production ML anymore, but the underlying
mechanism — "find the nearest vectors, use them" — is the same idea behind
**Retrieval-Augmented Generation (RAG)**: our own `06-nlp-llm-apps/pdf-rag-chatbot` and
the separate `doc-bot` project both embed text into vectors and retrieve the *nearest*
ones to a query — literally this algorithm, applied to text embeddings instead of
flower measurements. At massive scale, exact nearest-neighbor search (checking every
point, like our `find_dist()` does) is too slow, so production systems use
**Approximate Nearest Neighbor (ANN)** methods (HNSW, FAISS, ScaNN) — e.g. `ChromaDB`,
already used in `doc-bot`, is an approximate-KNN engine under the hood.

## Files in this folder

- `KNN.py` — the from-scratch implementation (loop-free distance computation)
- `knn_vectorized.py` — alternate no-loop version from the original coursework
- `assignment_notebook.ipynb` — original CS596 assignment notebook
- `test_knn.py` — our own experiment script (Iris, varying k)
