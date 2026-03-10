import numpy as np

def tanh(x):
    return np.tanh(x)

x = np.array([0.6, -0.3])

np.random.seed()

w1 = np.random.uniform(-0.5, 0.5, (2, 2))
w2 = np.random.uniform(-0.5, 0.5, (2, 1))

b1 = 0.5
b2 = 0.7

z1 = np.dot(x, w1) + b1
a1 = tanh(z1)

z2 = np.dot(a1, w2) + b2
output = tanh(z2)

print("Weights W1:\n", w1)
print("Weights W2:\n", w2)
print("Hidden layer output:\n", a1)
print("Final Network Output:\n", output)
