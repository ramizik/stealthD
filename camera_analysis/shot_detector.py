"""
Camera Shot Detector using PySceneDetect.

Detects camera shot boundaries (cuts, transitions) in soccer videos
to improve tracking stability and homography estimation accuracy.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import os
import pickle
from typing import List, Optional, Tuple

import cv2

try:
    from scenedetect import SceneManager, VideoManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("Warning: PySceneDetect not installed. Shot detection will be disabled.")
    print("Install with: pip install scenedetect[opencv]")


class ShotDetector:
    """
    Detects camera shot boundaries using content-based scene detection.

    Shot boundaries indicate abrupt camera changes (cuts, angle changes, zoom transitions)
    which are important for:
    - Resetting homography estimation
    - Improving tracking stability
    - Segmenting analysis by camera angle
    """

    def __init__(self, threshold: float = 30.0, min_scene_len: int = 15):
        """
        Initialize shot detector.

        Args:
            threshold: Content detection threshold (0-255, typically 10-60)
                      - Lower values: More sensitive, detects subtle changes
                      - Higher values: Less sensitive, only major cuts
                      - Default 30.0 works well for soccer broadcasts
            min_scene_len: Minimum scene length in frames (default 15 = 0.5s at 30fps)
        """
        if not SCENEDETECT_AVAILABLE:
            raise ImportError(
                "PySceneDetect is required for shot detection. "
                "Install with: pip install scenedetect[opencv]"
            )

        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect_shots(self, video_path: str, read_from_stub: bool = False,
                    stub_path: Optional[str] = None, downscale_factor: int = 1) -> List[int]:
        """
        Detect shot boundaries in a video.

        Args:
            video_path: Path to video file
            read_from_stub: If True, try to load cached results
            stub_path: Path to pickle file for caching
            downscale_factor: Downscale factor for faster processing (1=no downscale, 2=half, etc.)

        Returns:
            List of frame indices where shot boundaries occur
            Example: [0, 145, 342, 567] means shots at frames 0-144, 145-341, 342-566, etc.
        """
        # Try to load from cache
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                shot_boundaries = pickle.load(f)
                print(f"  Loaded cached shot boundaries from: {stub_path}")
                return shot_boundaries

        print(f"  Detecting camera shots (threshold={self.threshold})...")

        # Create video manager and scene manager
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()

        # Add content detector with specified threshold
        scene_manager.add_detector(
            ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
        )

        # Get base timecode
        base_timecode = video_manager.get_base_timecode()

        # Apply downscaling if specified (improves speed)
        if downscale_factor > 1:
            video_manager.set_downscale_factor(downscale_factor)

        # Start video manager and detect scenes
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # Get scene list as (start, end) timecode tuples
        scene_list = scene_manager.get_scene_list(base_timecode)

        # Release video manager
        video_manager.release()

        # Extract shot boundaries (start frame of each scene)
        shot_boundaries = [0]  # First frame is always a boundary

        for scene_start, scene_end in scene_list:
            # Get frame number from timecode
            frame_num = scene_start.get_frames()
            if frame_num > 0 and frame_num not in shot_boundaries:
                shot_boundaries.append(frame_num)

        print(f"  Detected {len(shot_boundaries)} camera shots")

        # Save to cache
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(shot_boundaries, f)
            print(f"  Shot boundaries cached to: {stub_path}")

        return shot_boundaries

    def get_shot_segments(self, shot_boundaries: List[int],
                         total_frames: int) -> List[Tuple[int, int]]:
        """
        Convert shot boundaries to (start, end) frame segments.

        Args:
            shot_boundaries: List of frame indices where shots start
            total_frames: Total number of frames in video

        Returns:
            List of (start_frame, end_frame) tuples
            Example: [(0, 144), (145, 341), (342, 566)]
        """
        segments = []

        for i in range(len(shot_boundaries)):
            start_frame = shot_boundaries[i]

            # End frame is either next boundary - 1, or last frame
            if i < len(shot_boundaries) - 1:
                end_frame = shot_boundaries[i + 1] - 1
            else:
                end_frame = total_frames - 1

            segments.append((start_frame, end_frame))

        return segments

    def is_shot_boundary(self, frame_idx: int, shot_boundaries: List[int],
                        tolerance: int = 2) -> bool:
        """
        Check if a frame is near a shot boundary.

        Args:
            frame_idx: Frame index to check
            shot_boundaries: List of shot boundary frame indices
            tolerance: Number of frames around boundary to consider (default 2)

        Returns:
            True if frame is within tolerance of a shot boundary
        """
        for boundary in shot_boundaries:
            if abs(frame_idx - boundary) <= tolerance:
                return True
        return False

    def get_shot_id(self, frame_idx: int, shot_boundaries: List[int]) -> int:
        """
        Get the shot ID for a given frame.

        Args:
            frame_idx: Frame index
            shot_boundaries: List of shot boundary frame indices

        Returns:
            Shot ID (0-indexed), where shot 0 is frames [0, boundary[1]-1], etc.
        """
        shot_id = 0
        for i, boundary in enumerate(shot_boundaries):
            if frame_idx >= boundary:
                shot_id = i
            else:
                break
        return shot_id


def detect_shots_fallback(video_path: str, threshold: float = 30.0) -> List[int]:
    """
    Fallback shot detection using basic frame differencing.

    Used when PySceneDetect is not available. Less accurate but functional.

    Args:
        video_path: Path to video file
        threshold: Difference threshold (0-100, default 30)

    Returns:
        List of frame indices where shot boundaries occur
    """
    print("  Using fallback shot detection (frame differencing)...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    shot_boundaries = [0]
    prev_frame = None

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and resize for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        if prev_frame is not None:
            # Calculate absolute difference
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = diff.mean()

            # If difference exceeds threshold, it's likely a shot boundary
            if mean_diff > threshold:
                shot_boundaries.append(frame_idx)

        prev_frame = gray

    cap.release()

    print(f"  Detected {len(shot_boundaries)} camera shots (fallback)")
    return shot_boundaries
