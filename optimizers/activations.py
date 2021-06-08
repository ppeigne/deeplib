import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def leaky_relu(z: np.ndarray, rate: float = .01) -> np.ndarray:
    return np.maximum(z * rate, z)

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# def relu(z):
#     return np.maximum(0, z)

# def leaky_relu(z, rate=.01):
#     return np.maximum(z * rate, z)

# def sigmoid_prime(z):
#     return 1 / (1 + np.exp(-z))

def relu_prime(z):
    return (z > 0) * 1

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# def leaky_relu(z, rate=.01):
#     return np.maximum(z * rate, z)
