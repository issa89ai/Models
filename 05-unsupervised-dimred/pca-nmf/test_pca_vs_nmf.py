import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Part 1: downstream classification task, breast cancer dataset ---
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# NMF requires non-negative input -- scale to [0, 1] instead of standardizing to mean 0
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

N_COMPONENTS = 10

pca = PCA(n_components=N_COMPONENTS)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

nmf = NMF(n_components=N_COMPONENTS, random_state=42, max_iter=1000)
X_train_nmf = nmf.fit_transform(X_train_scaled)
# MinMaxScaler is fit on the training set only (correct practice), but the test
# set can contain values slightly outside that observed range, producing a few
# small negative values after scaling. NMF requires strict non-negativity, so
# clip them back to 0 -- these are a handful of borderline values, not a sign
# our scaling or split is wrong.
X_test_nmf = nmf.transform(np.clip(X_test_scaled, 0, None))

clf_pca = LogisticRegression(max_iter=5000).fit(X_train_pca, y_train)
clf_nmf = LogisticRegression(max_iter=5000).fit(X_train_nmf, y_train)
clf_raw = LogisticRegression(max_iter=5000).fit(X_train_scaled, y_train)

acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))
acc_nmf = accuracy_score(y_test, clf_nmf.predict(X_test_nmf))
acc_raw = accuracy_score(y_test, clf_raw.predict(X_test_scaled))

print(f"Breast cancer dataset: {X.shape[1]} original features -> {N_COMPONENTS} components")
print(f"Logistic Regression accuracy, raw (all {X.shape[1]} features): {acc_raw:.4f}")
print(f"Logistic Regression accuracy, PCA ({N_COMPONENTS} components):  {acc_pca:.4f}")
print(f"Logistic Regression accuracy, NMF ({N_COMPONENTS} components):  {acc_nmf:.4f}")

# --- Part 2: visual comparison of components, digits dataset ---
digits = load_digits()
X_digits = digits.data

pca_digits = PCA(n_components=8)
pca_digits.fit(X_digits)

nmf_digits = NMF(n_components=8, random_state=42, max_iter=1000)
nmf_digits.fit(X_digits)

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i in range(8):
    axes[0, i].imshow(pca_digits.components_[i].reshape(8, 8), cmap='RdBu')
    axes[0, i].axis('off')
    axes[1, i].imshow(nmf_digits.components_[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('PCA', fontsize=12)
axes[1, 0].set_ylabel('NMF', fontsize=12)
fig.suptitle('PCA components (can be negative, red/blue) vs. NMF components (non-negative, parts-based)')
plt.tight_layout()
plt.savefig('pca_vs_nmf_components.png')
print("\nSaved pca_vs_nmf_components.png")
plt.show()
