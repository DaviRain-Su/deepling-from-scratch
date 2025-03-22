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


x = np.arange(-5.0, 100.0, 0.1)
# y = step_function(x)
# y = sigmoid(x)
# y = relu(x)
y = softmax(x)
plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
plt.show()
