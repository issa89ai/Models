import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from SVM import SVM

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVM(alpha_in=0.01, reg_const_in=0.01)
model.train(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = np.mean(y_pred_train == y_train)
test_acc = np.mean(y_pred_test == y_test)

print(f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
