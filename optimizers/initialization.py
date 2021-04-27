import numpy as np

def initialize_relu(shape: (int, int)) -> np.ndarray:
    return np.random.random(shape) * np.sqrt(2 / shape[0])


def initialize_tanh(shape: (int, int)) -> np.ndarray:
    return np.random.random(shape) * np.sqrt(1 / shape[0])