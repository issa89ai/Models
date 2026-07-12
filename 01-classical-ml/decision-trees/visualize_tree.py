import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Plot 1: train vs test accuracy across depth ---
depths = [1, 2, 3, 4, 5, None]
depth_labels = [str(d) for d in depths]
train_accs = []
test_accs = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_accs.append(clf.score(X_train, y_train))
    test_accs.append(clf.score(X_test, y_test))

plt.figure(figsize=(7, 5))
plt.plot(depth_labels, train_accs, marker='o', label='Train accuracy')
plt.plot(depth_labels, test_accs, marker='o', label='Test accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy by Tree Depth')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_depth.png')
print("Saved accuracy_vs_depth.png")

# --- Plot 2: the actual tree structure (depth=3) ---
clf_depth3 = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_depth3.fit(X_train, y_train)

plt.figure(figsize=(14, 8))
plot_tree(clf_depth3, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree Structure (max_depth=3)')
plt.savefig('tree_structure.png')
print("Saved tree_structure.png")

# --- Plot 3: decision boundary using 2 features (petal length, petal width) ---
X_2feat = X[:, [2, 3]]  # petal length, petal width
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_2feat, y, test_size=0.2, random_state=42)

clf_2feat = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_2feat.fit(X_train2, y_train2)

x_min, x_max = X_2feat[:, 0].min() - 1, X_2feat[:, 0].max() + 1
y_min, y_max = X_2feat[:, 1].min() - 1, X_2feat[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = clf_2feat.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_2feat[:, 0], X_2feat[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Decision Boundary (max_depth=3, 2 features)')
plt.legend(handles=scatter.legend_elements()[0], labels=list(iris.target_names))
plt.savefig('decision_boundary.png')
print("Saved decision_boundary.png")

plt.show()
