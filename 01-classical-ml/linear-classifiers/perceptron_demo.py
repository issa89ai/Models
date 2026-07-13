import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class PerceptronDemo:
    def __init__(self, learning_rate=0.1, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.mistakes_per_epoch = []

    def train(self, X_train, y_train):
        # y must be -1 / +1 for this update rule
        y_signed = np.where(y_train == 0, -1, 1)

        n_features = X_train.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            mistakes = 0
            for i in range(X_train.shape[0]):
                score = np.dot(self.w, X_train[i]) + self.b
                prediction = 1 if score >= 0 else -1

                if prediction != y_signed[i]:
                    self.w = self.w + self.learning_rate * y_signed[i] * X_train[i]
                    self.b = self.b + self.learning_rate * y_signed[i]
                    mistakes += 1

            self.mistakes_per_epoch.append(mistakes)
            if mistakes == 0:
                print(f"Converged after {epoch + 1} epochs (zero mistakes)")
                break

    def predict(self, X_test):
        scores = X_test.dot(self.w) + self.b
        return np.where(scores >= 0, 1, 0)


iris = load_iris()
# Only setosa (0) and versicolor (1) -- a linearly separable pair
mask = iris.target != 2
X = iris.data[mask]
y = iris.target[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = PerceptronDemo(learning_rate=0.1, epochs=50)
model.train(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Mistakes per epoch: {model.mistakes_per_epoch}")
print(f"Test accuracy: {accuracy:.4f}")
