"""Core Player Detection Functions for Soccer Analysis.

This module provides core functionality for detecting players, ball, and referees
using YOLO models. Pipeline functions have been moved to detection_pipeline.py.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

# ================================================
# Core Detection Functions
# ================================================

def load_detection_model(model_path: str) -> YOLO:
    """Load and return a YOLO detection model.

    Args:
        model_path: Path to the YOLO model file

    Returns:
        YOLO model instance configured to use GPU

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
        print(f"✓ Detection model loaded on GPU: {torch.cuda.get_device_name(GPU_DEVICE)}")
        print(f"✓ InferenceSlicer enabled for improved ball detection (overlapping tiles)")
    else:
        print("⚠️ WARNING: Running on CPU - This will be VERY slow!")
        print(f"✓ InferenceSlicer enabled for improved ball detection (overlapping tiles)")

    return model


def detect_objects_in_frames(model: YOLO, frames, conf_threshold: float = 0.1) -> List:
    """Detect objects in video frames using YOLO model.

    Args:
        model: Loaded YOLO model
        frames: Video frames or single frame
        conf_threshold: Confidence threshold for detections (default: 0.1 for better goalkeeper detection)

    Returns:
        Detection results from YOLO model
    """
    return model(frames, verbose=False, conf=conf_threshold)

def get_detections(detection_model: YOLO, frame: np.ndarray, use_slicer: bool = True, ball_padding: int = 10, conf_threshold: float = 0.1) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
    """Get separated detections for players, ball, and referees.

    Uses InferenceSlicer for improved ball detection. Small objects like balls
    can shrink to barely a few pixels after resizing, making them difficult to detect.
    InferenceSlicer divides the image into overlapping patches, making the ball
    proportionally larger within each patch for better detection accuracy.

    Args:
        detection_model: Loaded YOLO model
        frame: Input frame as numpy array
        use_slicer: Whether to use inference slicer (default: True for better ball detection)
        ball_padding: Padding (in pixels) to add around ball detections for better tracking (default: 10)
        conf_threshold: Confidence threshold for detections (default: 0.1 for better goalkeeper detection)

    Returns:
        Tuple of (player_detections, ball_detections, referee_detections)
    """
    def inference_callback(patch: np.ndarray) -> sv.Detections:
        """Convert YOLO results to supervision format."""
        result = detect_objects_in_frames(detection_model, patch, conf_threshold=conf_threshold)[0]
        return sv.Detections.from_ultralytics(result)

    # Get detections using slicer or direct inference
    if use_slicer:
        h, w, _ = frame.shape

        # Configure slicer with overlapping tiles for smooth tracking
        # Overlap ensures ball detected at tile boundaries won't be missed
        slicer = sv.InferenceSlicer(
            callback=inference_callback,
            slice_wh=(w // 2 + 100, h // 2 + 100),  # Tile size (slightly larger than half frame)
            overlap_ratio_wh=None,  # Explicitly disable deprecated parameter
            overlap_wh=(100, 100),  # 100px overlap between tiles
            iou_threshold=0.1,  # Low threshold for ball (small object)
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION  # Remove duplicate detections
        )
        detections = slicer(frame)
    else:
        detections = inference_callback(frame)

    # Separate detections by class
    player_detections = detections[detections.class_id == 0]
    ball_detections = detections[detections.class_id == 1]
    referee_detections = detections[detections.class_id == 2]

    # Add padding to ball bounding boxes for improved tracking stability
    if len(ball_detections.xyxy) > 0 and ball_padding > 0:
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=ball_padding)

    return player_detections, ball_detections, referee_detections