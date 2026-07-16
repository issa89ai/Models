import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def normal_equation(X, y):
    X_with_bias = np.column_stack((np.ones(len(X)), X))
    theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
    return theta


def predict_normal_eq(X, theta):
    X_with_bias = np.column_stack((np.ones(len(X)), X))
    return X_with_bias.dot(theta)


def gradient_descent(X, y, learning_rate=0.1, iterations=5000):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_norm = (X - means) / stds
    X_with_bias = np.column_stack((np.ones(len(X_norm)), X_norm))

    theta = np.zeros(X_with_bias.shape[1])
    m = len(y)
    for i in range(iterations):
        gradient = (1 / m) * X_with_bias.T.dot(X_with_bias.dot(theta) - y)
        theta = theta - learning_rate * gradient
    return theta, means, stds


def predict_gradient_descent(X, theta, means, stds):
    X_norm = (X - means) / stds
    X_with_bias = np.column_stack((np.ones(len(X_norm)), X_norm))
    return X_with_bias.dot(theta)


# Method 1: Normal Equation (closed-form, from scratch)
theta_normal = normal_equation(X_train, y_train)
pred_normal = predict_normal_eq(X_test, theta_normal)
mse_normal = mean_squared_error(y_test, pred_normal)
r2_normal = r2_score(y_test, pred_normal)

# Method 2: Gradient Descent (iterative, from scratch)
theta_gd, means, stds = gradient_descent(X_train, y_train)
pred_gd = predict_gradient_descent(X_test, theta_gd, means, stds)
mse_gd = mean_squared_error(y_test, pred_gd)
r2_gd = r2_score(y_test, pred_gd)

# Method 3: sklearn LinearRegression (library reference)
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
pred_sklearn = sklearn_model.predict(X_test)
mse_sklearn = mean_squared_error(y_test, pred_sklearn)
r2_sklearn = r2_score(y_test, pred_sklearn)

print(f"Normal Equation:   MSE={mse_normal:.2f}, R2={r2_normal:.4f}")
print(f"Gradient Descent:  MSE={mse_gd:.2f}, R2={r2_gd:.4f}")
print(f"sklearn:           MSE={mse_sklearn:.2f}, R2={r2_sklearn:.4f}")
