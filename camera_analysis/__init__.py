"""
Camera Analysis Module

Provides camera movement estimation and compensation for improved tracking accuracy.
Also provides camera shot boundary detection for scene segmentation.
"""

from camera_analysis.camera_movement_estimator import CameraMovementEstimator
from camera_analysis.shot_detector import ShotDetector

__all__ = ['CameraMovementEstimator', 'ShotDetector']
