"""
Distance calculation utilities.

Provides functions for calculating distances between points in 2D space
and utilities for bounding box operations.
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


def get_center_of_bbox(bbox):
    """
    Get the center point of a bounding box.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]

    Returns:
        Tuple of (center_x, center_y) as integers
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """
    Get the width of a bounding box.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]

    Returns:
        Width of the bounding box (x2 - x1)
    """
    return bbox[2] - bbox[0]


def get_foot_position(bbox):
    """
    Get the foot position (bottom center) of a bounding box.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]

    Returns:
        Tuple of (center_x, bottom_y) as integers
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
