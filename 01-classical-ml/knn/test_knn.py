from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from KNN import KNN

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for k in [1, 5, 25, 75, 120]:
    model = KNN(k)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"k={k}: accuracy={accuracy:.4f}")
