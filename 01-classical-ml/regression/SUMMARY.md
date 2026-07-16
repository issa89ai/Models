# Regression — What We Learned

## Regression vs. classification

First model type in this repo that predicts a **continuous number** (disease
progression score) rather than a category. The loss function changes
accordingly: **Mean Squared Error (MSE)** — average squared distance between
predicted and actual values — instead of Gini impurity or cross-entropy.

## Three ways to solve linear regression, compared

Using `sklearn`'s diabetes dataset (10 health features -> disease progression
score after one year) instead of the original notebooks' bike-rental CSV
(loaded from a Google Drive path not available outside that environment):

```
Normal Equation:   MSE=2900.19, R2=0.4526
Gradient Descent:  MSE=2899.86, R2=0.4527
sklearn:           MSE=2900.19, R2=0.4526
```

- **Normal Equation** (`θ = (XᵀX)⁻¹Xᵀy`): an exact, closed-form solution via
  one matrix formula — no iteration, no learning rate to tune.
- **Gradient Descent**: the same iterative approach used for SVM/Softmax
  training, requires feature normalization first since raw features have very
  different scales.
- **sklearn's `LinearRegression`**: matches the Normal Equation to 4 decimal
  places (same underlying closed-form approach, more numerically robust
  solver). Gradient Descent lands almost exactly on the same answer (an
  iterative approximation converging toward the same true minimum) but isn't
  bit-for-bit identical.

Three fundamentally different computational approaches converging on
essentially the same answer is strong practical confirmation they're all
correctly solving the same underlying optimization problem.

**On R²=0.4526**: much lower than the 90-100% seen repeatedly on Iris — not a
failure, but a reflection that Iris is a small, artificially clean dataset,
while real disease-progression data is genuinely noisy and only partially
predictable from these 10 features.

## LASSO — regularized regression

Adds an L1 penalty (`α × Σ|θⱼ|`) to the MSE loss, which drives some
coefficients to *exactly* zero (unlike L2/Ridge, which only shrinks them) —
automatic feature selection.

```
Plain regression:         R2=0.4526 (0/10 coefficients zero)
alpha=0.01: 0/10 zeroed,   R2=0.4567
alpha=0.1:  3/10 zeroed,   R2=0.4719  <- best result of all, incl. plain regression
alpha=1:    7/10 zeroed,   R2=0.3576
alpha=5:    10/10 zeroed,  R2=-0.0120
alpha=10:   10/10 zeroed,  R2=-0.0120
```

**Key finding: `alpha=0.1` beats plain (unregularized) regression**, despite
dropping 3 features. Plain regression's huge raw coefficients (`s1: -931.49`,
`s5: 736.20`) were likely overfitting slightly, chasing noise; mild
regularization trades a little training fit for better generalization to
unseen data. Push regularization too far (`alpha=1`) and real signal starts
getting discarded too, not just noise (R² drops to 0.36). At `alpha=5` and
above, every coefficient is zero — the model has degenerated into always
predicting the average value, the same failure mode as KNN at k=120.

**Regularization path** (`lasso_regularization_path.png`) shows this
shrinkage feature-by-feature across a range of `alpha`. `age` and `s2`
collapse to zero fastest (least essential to the model); **`bmi` survives
longest**, remaining nonzero past `alpha≈1` — identifying it as the single
most robust predictor in this model, which matches real-world medical
knowledge that BMI is a major diabetes risk factor. A good sanity check that
the model's implicit notion of "importance" lines up with actual domain
knowledge.

## Files in this folder

- `linear_regression_*.ipynb`, `lasso_*.ipynb`, `logistic_regression_python.ipynb`
  — original CS504 lecture notebooks (R, basic Python, PyTorch, sklearn versions)
- `test_regression.py` — Normal Equation vs. Gradient Descent vs. sklearn comparison (diabetes dataset)
- `test_lasso.py` — LASSO coefficient/R² comparison + regularization path plot
- `lasso_regularization_path.png` — saved plot
