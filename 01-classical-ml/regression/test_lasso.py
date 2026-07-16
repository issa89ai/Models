import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plain linear regression, for comparison
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print("Plain Linear Regression coefficients:")
for name, coef in zip(feature_names, linreg.coef_):
    print(f"  {name}: {coef:.2f}")
print(f"R2: {r2_score(y_test, linreg.predict(X_test)):.4f}\n")

# LASSO at increasing alpha values
alphas = [0.01, 0.1, 1, 5, 10]
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    n_zero = np.sum(lasso.coef_ == 0)
    r2 = r2_score(y_test, lasso.predict(X_test))
    print(f"alpha={alpha}: {n_zero}/{len(feature_names)} coefficients zeroed, R2={r2:.4f}")

# Regularization path: how each coefficient changes as alpha increases
alphas_path = np.logspace(-2, 2, 50)
coefs_path = []
for alpha in alphas_path:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefs_path.append(lasso.coef_)
coefs_path = np.array(coefs_path)

plt.figure(figsize=(9, 6))
for i, name in enumerate(feature_names):
    plt.plot(alphas_path, coefs_path[:, i], label=name)
plt.xscale('log')
plt.xlabel('alpha (regularization strength)')
plt.ylabel('Coefficient value')
plt.title('LASSO Regularization Path')
plt.legend(loc='upper right', fontsize=8)
plt.axhline(0, color='black', linewidth=0.5)
plt.savefig('lasso_regularization_path.png')
print("\nSaved lasso_regularization_path.png")
plt.show()
