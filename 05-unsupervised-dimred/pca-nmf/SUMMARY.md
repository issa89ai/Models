# PCA — What We Learned

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

## Files in this folder

- `Dimenionality reduction.ipynb`, `PCA_Loading Matrix.ipynb`, `PCA_NMF.ipynb`
  -- original CS504 coursework notebooks (reference the unavailable external CSV)
- `test_pca_digits.py` -- our standalone PCA experiment on digits, described above
- `pca_digits_analysis.png`, `pca_reconstruction.png` -- saved plots
