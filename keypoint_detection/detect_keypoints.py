"""Core Keypoint Detection Functions for Soccer Analysis.

This module provides core functionality for detecting soccer field keypoints
using YOLO pose estimation models and geometric calculations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import supervision as sv
import torch
from ultralytics import YOLO

# Global counter for throttling keypoint detection warnings
_keypoint_warning_counter = 0

# ================================================
# Core Detection Functions
# ================================================

def load_keypoint_model(model_path: str) -> YOLO:
    """Load and return a YOLO pose estimation model for keypoint detection.

    Args:
        model_path: Path to the YOLO pose model file

    Returns:
        YOLO model instance configured for pose estimation on GPU

    Raises:
        RuntimeError: If GPU is required but not available
    """
    from constants import GPU_DEVICE, REQUIRE_GPU

    # Check GPU availability
    if REQUIRE_GPU and not torch.cuda.is_available():
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "GPU REQUIRED BUT NOT AVAILABLE!\n"
            "="*60 + "\n"
            "This script is configured to use GPU only.\n"
            "Possible solutions:\n"
            "  1. Install CUDA-enabled PyTorch:\n"
            "     pip uninstall torch torchvision torchaudio\n"
            "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
            "  2. Install NVIDIA CUDA drivers from nvidia.com\n"
            "  3. Check GPU with: nvidia-smi\n"
            "\n"
            "To allow CPU usage (NOT RECOMMENDED), set REQUIRE_GPU=False in constants.py\n"
            "="*60
        )

    # Load model
    model = YOLO(model_path)

    # Force GPU usage if available
    if torch.cuda.is_available():
        device = f'cuda:{GPU_DEVICE}'
        model.to(device)
        print(f"✓ Keypoint model loaded on GPU: {torch.cuda.get_device_name(GPU_DEVICE)}")
    else:
        print("⚠️ WARNING: Running on CPU - This will be VERY slow!")

    return model


def detect_keypoints_in_frames(model: YOLO, frames) -> List:
    """Detect keypoints in video frames using YOLO pose model.

    Args:
        model: Loaded YOLO pose model
        frames: Video frames or single frame

    Returns:
        Detection results from YOLO pose model containing keypoints
    """
    return model(frames, verbose=False)


def get_keypoint_detections(keypoint_model: YOLO, frame: np.ndarray) -> Tuple[sv.Detections, sv.KeyPoints]:
    """Get keypoint detections using supervision KeyPoints (matches roboflow/sports).

    ROBOFLOW/SPORTS APPROACH:
    - Uses sv.KeyPoints.from_ultralytics() wrapper for consistent interface
    - Returns sv.KeyPoints object with .xy property for coordinates
    - Filtering happens later using mask: (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

    Args:
        keypoint_model: Loaded YOLO pose model
        frame: Input frame as numpy array

    Returns:
        Tuple of (detections, keypoints) where keypoints is sv.KeyPoints object
        with .xy property containing coordinates
    """
    results = detect_keypoints_in_frames(keypoint_model, frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # CRITICAL: Use sv.KeyPoints like roboflow/sports does
    # This gives us the .xy property that matches their implementation exactly
    keypoints = sv.KeyPoints.from_ultralytics(results)

    # Throttle warnings to reduce console spam (only warn every 100 failures)
    global _keypoint_warning_counter

    # Log keypoint statistics for debugging (first few frames only)
    if len(keypoints) > 0:
        # Apply spatial filter to count valid keypoints
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        valid_count = np.sum(mask)

        # Only warn if extremely few valid keypoints (< 4 needed for homography)
        if valid_count < 4:
            _keypoint_warning_counter += 1
            if _keypoint_warning_counter % 100 == 1:
                print(f"[Keypoint Detection] WARNING: Only {valid_count}/32 keypoints with x>1, y>1 "
                      f"({_keypoint_warning_counter} occurrences)")
        else:
            _keypoint_warning_counter = 0  # Reset on success
    else:
        _keypoint_warning_counter += 1
        if _keypoint_warning_counter % 100 == 1:
            print(f"[Keypoint Detection] WARNING: No pitch keypoints detected in frame "
                  f"({_keypoint_warning_counter} occurrences)")

    return detections, keypoints


def normalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Normalize keypoint coordinates to 0-1 range.

    Args:
        keypoints: Array of keypoints with shape (N, 27, 3)
        image_width: Width of the source image
        image_height: Height of the source image

    Returns:
        Normalized keypoints array with coordinates in 0-1 range
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints

    normalized_keypoints = keypoints.copy()
    normalized_keypoints[:, :, 0] /= image_width   # Normalize x coordinates
    normalized_keypoints[:, :, 1] /= image_height  # Normalize y coordinates
    # Visibility values (index 2) remain unchanged

    return normalized_keypoints


def denormalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Denormalize keypoint coordinates from 0-1 range to image coordinates.

    Args:
        keypoints: Array of normalized keypoints with shape (N, 27, 3)
        image_width: Width of the target image
        image_height: Height of the target image

    Returns:
        Denormalized keypoints array with pixel coordinates
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints

    denormalized_keypoints = keypoints.copy()
    denormalized_keypoints[:, :, 0] *= image_width   # Denormalize x coordinates
    denormalized_keypoints[:, :, 1] *= image_height  # Denormalize y coordinates
    # Visibility values remain unchanged

    return denormalized_keypoints


def filter_visible_keypoints(keypoints: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
    """Filter keypoints based on visibility confidence.

    Args:
        keypoints: Array of keypoints with shape (N, 27, 3)
        confidence_threshold: Minimum confidence for a keypoint to be considered visible

    Returns:
        Filtered keypoints with low-confidence points set to (0, 0, 0)
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints

    filtered_keypoints = keypoints.copy()

    # Set invisible keypoints to (0, 0, 0)
    invisible_mask = keypoints[:, :, 2] < confidence_threshold
    filtered_keypoints[invisible_mask] = 0

    return filtered_keypoints


def extract_field_corners(keypoints: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Extract the four corner points of the soccer field from detected keypoints.

    Args:
        keypoints: Array of keypoints with shape (N, 27, 3)

    Returns:
        Dictionary containing corner coordinates:
        {'top_left': (x, y), 'top_right': (x, y),
         'bottom_left': (x, y), 'bottom_right': (x, y)}
    """
    if keypoints is None or keypoints.size == 0:
        return {'top_left': (0, 0), 'top_right': (0, 0),
                'bottom_left': (0, 0), 'bottom_right': (0, 0)}

    # Assuming first detection and using specific keypoint indices for field corners
    # Based on the 27-keypoint model structure from the dataset
    corners = {}

    if keypoints.shape[0] > 0:
        kpts = keypoints[0]  # First detection

        # Extract corner keypoints (indices based on dataset structure)
        # These indices correspond to the field boundary points
        corners['top_left'] = (float(kpts[0, 0]), float(kpts[0, 1])) if kpts[0, 2] > 0 else (0, 0)
        corners['top_right'] = (float(kpts[16, 0]), float(kpts[16, 1])) if kpts[16, 2] > 0 else (0, 0)
        corners['bottom_left'] = (float(kpts[9, 0]), float(kpts[9, 1])) if kpts[9, 2] > 0 else (0, 0)
        corners['bottom_right'] = (float(kpts[25, 0]), float(kpts[25, 1])) if kpts[25, 2] > 0 else (0, 0)

    return corners


def calculate_field_dimensions(corners: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Calculate field dimensions from corner keypoints.

    Args:
        corners: Dictionary of corner coordinates

    Returns:
        Dictionary containing field measurements:
        {'width': field_width, 'height': field_height, 'area': field_area}
    """
    top_left = corners['top_left']
    top_right = corners['top_right']
    bottom_left = corners['bottom_left']
    bottom_right = corners['bottom_right']

    # Calculate field dimensions
    width = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
    height = np.sqrt((bottom_left[0] - top_left[0])**2 + (bottom_left[1] - top_left[1])**2)
    area = width * height

    return {
        'width': float(width),
        'height': float(height),
        'area': float(area)
    }