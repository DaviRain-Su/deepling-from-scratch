import matplotlib.pyplot as plt
import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # Clip to prevent log(0)
    epsilon = 1e-7
    return -np.sum(t * np.log(y + epsilon)) / y.shape[0]


def cross_entropy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    epsilon = 1e-7
    return (
        -np.sum(t * np.log(y[np.arange(batch_size), t] + epsilon)) / batch_size
    )


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


def function_2(x):
    return np.sum(x**2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


x = np.arange(0.0, 20.0, 0.1)
# y = step_function(x)
# y = sigmoid(x)
# y = relu(x)
y = function_1(x)
# y = softmax(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))

init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x, lr=0.01, step_num=1000)
print(result)
