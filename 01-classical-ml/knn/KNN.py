import numpy as np
import scipy
import operator

def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN():
    def __init__(self, k):
        self.k = k
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def find_dist(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(np.multiply(np.dot(X_test, self.X_train.T), -2) + np.sum(self.X_train ** 2, axis=1) + np.sum(X_test ** 2, axis=1)[:, np.newaxis])
        return dists

    def predict(self, X_test):
        dists = self.find_dist(X_test)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            k_nearest_index = np.argsort(dists[i, :])[:self.k]
            closest_y = self.y_train[k_nearest_index]
            labels_counts = {}
            for label in closest_y:
                if label in labels_counts.keys():
                    labels_counts[label] += 1
                else:
                    labels_counts[label] = 0
            sorted_labels_counts = sorted(
                labels_counts.items(), key=operator.itemgetter(1), reverse=True)
            y_pred[i] = sorted_labels_counts[0][0]
        return y_pred