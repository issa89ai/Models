# PCA and NMF — What We Learned

## Status of the original files

The original notebooks used an external oil & gas well dataset
(`unconv_MV_v5.csv`) not available outside the original course environment
(distributed separately by the instructor, not saved alongside the
notebook). The core PCA workflow (`PCA().fit()`, `explained_variance_ratio_`,
cumulative variance, `.transform()`) is real and complete though --
substituted `sklearn`'s handwritten digits dataset (64 raw pixel features,
10 classes) instead, deliberately higher-dimensional than anything used
earlier in this repo (Iris=4, diabetes=10), since PCA's value becomes far
more obvious with more features to compress.

## The concept

PCA finds new axes (principal components) that are linear combinations of
the original features, ordered by how much variance each one captures: the
first component captures the most variance possible in a single direction,
the second captures the most *remaining* variance at a right angle to the
first, and so on. This compresses high-dimensional data into far fewer
dimensions while retaining most of the meaningful variation -- useful for
visualization (squeeze many features down to 2-3 for plotting) and as
preprocessing (remove redundant/noisy dimensions before another model).

## Experiment: digits dataset (1,797 images, 64 raw pixel features)

```
Components needed for 90% of variance: 21 (out of 64)
```

**The variance breakdown** follows the classic PCA "elbow" shape: the first
3 components alone capture ~40% of total variance (14.5% + 13.5% + 11.8%),
then a steep drop-off -- most real structure lives in a handful of
directions, with rapidly diminishing returns after. 21 of 64 components
(a 67% reduction) retain 90% of the total information.

**The 2-component scatter plot** is the most striking result: even
compressed all the way down to just 2 numbers per digit (a 32x compression),
the 10 digit classes visibly separate into distinct clusters -- despite PCA
never seeing the labels at all (purely unsupervised, just finding directions
of maximum variance in raw pixels). This demonstrates how much genuinely
redundant information is baked into raw pixel data.

**Reconstruction** (`pca_reconstruction.png`) makes "explained variance %"
concrete rather than abstract: at 2 components (28.5% variance) the
reconstructed digit is a blurry smear; by 10 components (73.8%) it's clearly
recognizable; by 20 components (89.4%) it's nearly indistinguishable from
the original. The variance percentage directly corresponds to visible image
quality.

## NMF vs. PCA: the non-negativity constraint

PCA's components can be negative, meaning components can represent patterns
that cancel/subtract, producing "holistic" representations. NMF forces
every value (data, components, weights) to be non-negative -- since nothing
can subtract, NMF is mathematically forced to build representations
additively, tending to isolate distinct, localized **parts** rather than
diffuse global patterns (the well-known effect originally demonstrated with
face images: PCA gives ghostly whole-face components, NMF gives
eye/nose/mouth-like parts).

**Experiment 1 -- downstream classification task** (`PCA_NMF.ipynb`'s
original approach, using the built-in breast cancer dataset, 30 features
-> 10 components each):
```
Logistic Regression accuracy, raw (30 features): 0.9825
Logistic Regression accuracy, PCA (10 components): 0.9825   <- identical to raw
Logistic Regression accuracy, NMF (10 components): 0.8860   <- ~10 points lower
```
PCA compressed 30 features to 10 with **zero loss** in downstream accuracy --
it's mathematically the optimal linear compression for a given number of
dimensions (directly maximizes captured variance). NMF, given the same
number of components, loses real accuracy: its non-negativity constraint is
a genuine restriction that trades some compression efficiency for
interpretability.

**A real bug worth noting**: `MinMaxScaler` is correctly fit on the training
set only (fitting on test data would be leakage), but the test set naturally
contains a few feature values slightly outside the training set's observed
range, producing a handful of small negative values after scaling -- which
NMF's strict non-negativity check rejects. Fixed by clipping the *scaled
test data* (not the raw data) to `>= 0` before the NMF transform. This isn't
a sign anything was wrong with the split or scaling -- it's expected
real-world behavior whenever a scaler's fitted range doesn't perfectly
cover held-out data.

**Experiment 2 -- visual component comparison** (digits dataset,
`pca_vs_nmf_components.png`): PCA's components are diffuse red/blue patterns
spanning the whole 8x8 grid -- hard to describe as "a stroke" or "a part."
NMF's components are visibly more localized -- sparse bright blobs
concentrated in specific regions (one is even recognizable as a ring/loop
shape) -- a direct visual confirmation of the parts-based vs. holistic
distinction, using real data rather than just theory.

**Overall conclusion**: NMF trades raw predictive/compression efficiency for
genuinely more interpretable, parts-based components -- a real, visible, and
numerically concrete interpretability-vs-performance tradeoff.

## Files in this folder

- `Dimenionality reduction.ipynb`, `PCA_Loading Matrix.ipynb`, `PCA_NMF.ipynb`
  -- original CS504 coursework notebooks (PCA_NMF.ipynb's breast-cancer +
  downstream-classifier approach was reused directly; the others reference
  the unavailable external CSV)
- `test_pca_digits.py` -- standalone PCA experiment on digits, described above
- `test_pca_vs_nmf.py` -- PCA vs. NMF comparison (breast cancer + digits), described above
- `pca_digits_analysis.png`, `pca_reconstruction.png`, `pca_vs_nmf_components.png` -- saved plots
