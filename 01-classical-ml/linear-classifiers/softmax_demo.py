import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class SoftmaxDemo:
    def __init__(self, learning_rate=0.1, epochs=1000, reg_const=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_const = reg_const
        self.W = None
        self.loss_history = []

    def softmax(self, scores):
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_classes = np.max(y_train) + 1
        self.W = 0.01 * np.random.randn(n_features, n_classes)

        for epoch in range(self.epochs):
            scores = X_train.dot(self.W)
            probs = self.softmax(scores)

            correct_logprobs = -np.log(probs[np.arange(n_samples), y_train])
            data_loss = np.sum(correct_logprobs) / n_samples
            reg_loss = self.reg_const * np.sum(self.W * self.W)
            loss = data_loss + reg_loss
            self.loss_history.append(loss)

            dscores = probs.copy()
            dscores[np.arange(n_samples), y_train] -= 1
            dscores /= n_samples

            dW = X_train.T.dot(dscores)
            dW += 2 * self.reg_const * self.W

            self.W -= self.learning_rate * dW

    def predict(self, X_test):
        scores = X_test.dot(self.W)
        return np.argmax(scores, axis=1)


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SoftmaxDemo(learning_rate=0.1, epochs=1000, reg_const=0.01)
model.train(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_acc = np.mean(y_pred_train == y_train)
test_acc = np.mean(y_pred_test == y_test)
print(f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
print(f"First loss: {model.loss_history[0]:.4f}, Final loss: {model.loss_history[-1]:.4f}")

plt.plot(model.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.title('Softmax Training Loss Over Time')
plt.savefig('softmax_loss_curve.png')
print("Saved softmax_loss_curve.png")
plt.show()
