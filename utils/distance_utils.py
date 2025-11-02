"""
Distance calculation utilities.

Provides functions for calculating distances between points in 2D space.
"""

import numpy as np
from typing import Tuple


def measure_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        point1: Array-like with [x, y] coordinates
        point2: Array-like with [x, y] coordinates

    Returns:
        Euclidean distance between the two points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def measure_xy_distance(point1: np.ndarray, point2: np.ndarray) -> Tuple[float, float]:
    """
    Calculate X and Y distance components between two 2D points.

    Args:
        point1: Array-like with [x, y] coordinates
        point2: Array-like with [x, y] coordinates

    Returns:
        Tuple of (x_distance, y_distance)
    """
    x_distance = point2[0] - point1[0]
    y_distance = point2[1] - point1[1]

    return x_distance, y_distance
