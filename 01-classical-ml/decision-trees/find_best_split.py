import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names


def gini(labels):
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)


def weighted_gini(y_left, y_right):
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * gini(y_left) + (len(y_right) / n) * gini(y_right)


def find_best_split(X, y, feature_names):
    best_score = float('inf')
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]
    for feature_index in range(n_features):
        values = np.unique(X[:, feature_index])
        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            y_left = y[left_mask]
            y_right = y[~left_mask]

            score = weighted_gini(y_left, y_right)

            if score < best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold

    print(f"Best split: {feature_names[best_feature]} <= {best_threshold:.3f}")
    print(f"Weighted Gini achieved: {best_score:.4f}")
    return best_feature, best_threshold


find_best_split(X, y, feature_names)
