import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class NaiveBayesDemo:
    def __init__(self, k=1):
        self.k = k  # Laplace smoothing parameter, same role as in the original naiveBayes.py

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.labels = np.unique(y_train)
        self.prior = {}
        self.conditional_prob = {}

        for label in self.labels:
            X_label = X_train[y_train == label]
            self.prior[label] = X_label.shape[0] / n_samples

            for feat in range(n_features):
                count_active = np.sum(X_label[:, feat] > 0)
                count_total = X_label.shape[0]
                # Laplace smoothing -- identical formula to the original file:
                # (+k) on the numerator, (+2k) on the denominator (0 and 1 both smoothed)
                self.conditional_prob[(feat, label)] = (count_active + self.k) / (count_total + 2 * self.k)

    def predict(self, X_test):
        predictions = []
        for datum in X_test:
            log_joint = {}
            for label in self.labels:
                log_joint[label] = np.log(self.prior[label])
                for feat in range(len(datum)):
                    p = self.conditional_prob[(feat, label)]
                    if datum[feat] > 0:
                        log_joint[label] += np.log(p)
                    else:
                        log_joint[label] += np.log(1 - p)
            predictions.append(max(log_joint, key=log_joint.get))
        return np.array(predictions)


iris = load_iris()
X = iris.data
y = iris.target

# Binarize each feature around its median, to match the original file's
# binary (active/inactive) feature assumption
medians = np.median(X, axis=0)
X_binary = (X > medians).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=42)

model = NaiveBayesDemo(k=1)
model.train(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = np.mean(y_pred_train == y_train)
test_acc = np.mean(y_pred_test == y_test)
print(f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
