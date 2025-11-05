"""Keypoint Detection Module for Soccer Analysis.

This module provides independent keypoint detection functionality for soccer field analysis,
following the modular architecture pattern of the Soccer Analysis project.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from keypoint_detection.detect_keypoints import (calculate_field_dimensions,
                                                 denormalize_keypoints,
                                                 detect_keypoints_in_frames,
                                                 extract_field_corners,
                                                 filter_visible_keypoints,
                                                 get_keypoint_detections,
                                                 load_keypoint_model,
                                                 normalize_keypoints)
from keypoint_detection.keypoint_constants import (CONFIDENCE_THRESHOLD,
                                                   FIELD_CORNERS,
                                                   KEYPOINT_CONNECTIONS,
                                                   KEYPOINT_NAMES,
                                                   NUM_KEYPOINTS)

__all__ = [
    # Core detection functions
    'load_keypoint_model',
    'detect_keypoints_in_frames',
    'get_keypoint_detections',
    'normalize_keypoints',
    'denormalize_keypoints',
    'filter_visible_keypoints',
    'extract_field_corners',
    'calculate_field_dimensions',

    # Constants
    'KEYPOINT_NAMES',
    'KEYPOINT_CONNECTIONS',
    'FIELD_CORNERS',
    'NUM_KEYPOINTS',
    'CONFIDENCE_THRESHOLD',
    "KEYPOINT_COLOR",
    "CONNECTION_COLOR",
    "FIELD_CORNER_COLOR",
    "TEXT_COLOR",
]