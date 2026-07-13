import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from SVM import SVM

iris = load_iris()
X_2feat = iris.data[:, [2, 3]]  # petal length, petal width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_2feat, y, test_size=0.2, random_state=42)

model = SVM(alpha_in=0.01, reg_const_in=0.01)
model.train(X_train, y_train)

x_min, x_max = X_2feat[:, 0].min() - 1, X_2feat[:, 0].max() + 1
y_min, y_max = X_2feat[:, 1].min() - 1, X_2feat[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_2feat[:, 0], X_2feat[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('SVM Decision Boundary (linear, 2 features)')
plt.legend(handles=scatter.legend_elements()[0], labels=list(iris.target_names))
plt.savefig('svm_decision_boundary.png')
print("Saved svm_decision_boundary.png")
plt.show()
