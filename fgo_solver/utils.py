import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Convert 3D vector to skew-symmetric matrix."""
    if v.shape != (3,):
        raise ValueError("Input vector must be 3-dimensional.")
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
