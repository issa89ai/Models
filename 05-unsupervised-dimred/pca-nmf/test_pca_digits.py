import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = digits.data
y = digits.target
print(f"Original data shape: {X.shape}  (64 raw pixel features per digit)")

pca_full = PCA()
pca_full.fit(X)
exp_var = pca_full.explained_variance_ratio_
cumul_var = np.cumsum(exp_var)

n_components_90 = np.argmax(cumul_var >= 0.90) + 1
print(f"Components needed for 90% of variance: {n_components_90} (out of 64)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(range(len(exp_var[:20])), exp_var[:20])
axes[0].set_title('Explained variance per component (first 20)')
axes[0].set_xlabel('Component')

axes[1].step(range(len(cumul_var)), cumul_var, where='mid')
axes[1].axhline(0.90, color='red', linestyle='--', label='90% threshold')
axes[1].set_title('Cumulative explained variance')
axes[1].set_xlabel('Number of components')
axes[1].legend()

# Reduce to 2 components and plot, colored by actual digit label
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)
scatter = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10)
axes[2].set_title('Digits compressed to 2 components')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[2], label='digit')

plt.tight_layout()
plt.savefig('pca_digits_analysis.png')
print("Saved pca_digits_analysis.png")

# Reconstruction: how much detail survives with only a few components?
fig2, axes2 = plt.subplots(1, 5, figsize=(12, 3))
original_digit = X[0].reshape(8, 8)
axes2[0].imshow(original_digit, cmap='gray')
axes2[0].set_title(f'Original\n(digit={y[0]})')
axes2[0].axis('off')

for i, n_comp in enumerate([2, 5, 10, 20]):
    pca_n = PCA(n_components=n_comp)
    X_transformed = pca_n.fit_transform(X)
    X_reconstructed = pca_n.inverse_transform(X_transformed)
    reconstructed_digit = X_reconstructed[0].reshape(8, 8)
    axes2[i + 1].imshow(reconstructed_digit, cmap='gray')
    axes2[i + 1].set_title(f'{n_comp} components\n({cumul_var[n_comp-1]*100:.1f}% var)')
    axes2[i + 1].axis('off')

plt.tight_layout()
plt.savefig('pca_reconstruction.png')
print("Saved pca_reconstruction.png")
plt.show()
