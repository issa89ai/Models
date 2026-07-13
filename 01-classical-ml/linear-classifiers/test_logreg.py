import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

iris = load_iris()
# Only versicolor (1) and virginica (2) -- the harder, overlapping pair
mask = iris.target != 0
X = iris.data[mask]
y = iris.target[mask] - 1  # remap to 0/1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This file expects shape (features, samples), not (samples, features) -- transpose
X_train_t = X_train.T
X_test_t = X_test.T
y_train_t = y_train.reshape(1, -1)
y_test_t = y_test.reshape(1, -1)

model = LogisticRegression()
model.train(X_train_t, y_train_t, num_iterations=1000, learning_rate=0.05)

y_pred_train = model.predict(X_train_t)
y_pred_test = model.predict(X_test_t)

train_acc = np.mean(y_pred_train == y_train_t)
test_acc = np.mean(y_pred_test == y_test_t)

print(f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
