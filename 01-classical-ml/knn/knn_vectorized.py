import numpy as np
import operator

class KNN(object):

    def __init__(self):
        pass
        
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(np.multiply(np.dot(X, self.X_train.T), -2) + np.sum(self.X_train ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis])
        return dists
        
    def predict_labels(self, dists, k=1):
 
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            k_nearest_index = np.argsort(dists[i, :])[:k]
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