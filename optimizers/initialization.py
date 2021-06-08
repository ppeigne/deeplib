import numpy as np
from typing import Tuple

def initialize_relu(shape: Tuple[int, int]) -> np.ndarray:
    return np.random.random(shape) * np.sqrt(2 / shape[0])


def initialize_tanh(shape: Tuple[int, int]) -> np.ndarray:
    return np.random.random(shape) * np.sqrt(1 / shape[0])
    

# def initialize_relu(shape):
#     return np.random.random(shape) * np.sqrt(2 / shape[0])


# def initialize_tanh(shape):
#     return np.random.random(shape) * np.sqrt(1 / shape[0])
 