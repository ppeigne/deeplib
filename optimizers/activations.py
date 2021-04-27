import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def leaky_relu(z: np.ndarray, rate: float = .01) -> np.ndarray:
    return np.maximum(z * rate, z)
