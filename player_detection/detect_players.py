"""Core Player Detection Functions for Soccer Analysis.

This module provides core functionality for detecting players, ball, and referees
using YOLO models. Pipeline functions have been moved to detection_pipeline.py.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from ultralytics import YOLO
import numpy as np
import supervision as sv
import torch

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
    from constants import REQUIRE_GPU, GPU_DEVICE

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
    else:
        print("⚠️ WARNING: Running on CPU - This will be VERY slow!")

    return model


def detect_objects_in_frames(model: YOLO, frames) -> List:
    """Detect objects in video frames using YOLO model.

    Args:
        model: Loaded YOLO model
        frames: Video frames or single frame

    Returns:
        Detection results from YOLO model
    """
    return model(frames, verbose=False)

def get_detections(detection_model: YOLO, frame: np.ndarray, use_slicer: bool = False) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
    """Get separated detections for players, ball, and referees.

    Args:
        detection_model: Loaded YOLO model
        frame: Input frame as numpy array
        use_slicer: Whether to use inference slicer for large images

    Returns:
        Tuple of (player_detections, ball_detections, referee_detections)
    """
    def inference_callback(frame: np.ndarray) -> sv.Detections:
        """Convert YOLO results to supervision format."""
        result = detect_objects_in_frames(detection_model, frame)[0]
        return sv.Detections.from_ultralytics(result)

    # Get detections using slicer or direct inference
    if use_slicer:
        slicer = sv.InferenceSlicer(callback=inference_callback)
        detections = slicer(frame)
    else:
        detections = inference_callback(frame)

    # Separate detections by class
    player_detections = detections[detections.class_id == 0]
    ball_detections = detections[detections.class_id == 1]
    referee_detections = detections[detections.class_id == 2]

    return player_detections, ball_detections, referee_detections