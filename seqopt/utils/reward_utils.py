import numpy as np


def scaled_dist(dist: np.ndarray, scale: float = 0.8):
    return 1.0 - np.tanh(np.arctanh(np.sqrt(0.95)) / scale * dist)