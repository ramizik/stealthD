"""
Ball Interpolation module for filling missing ball detections.

Uses linear interpolation to estimate ball positions in frames where
detection failed, improving ball tracking continuity.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict, Optional
import numpy as np


class BallInterpolator:
    """
    Interpolate missing ball positions using linear interpolation.

    Ball movement is often linear between detections, making linear
    interpolation effective for filling gaps in tracking.
    """

    def __init__(self):
        """Initialize the ball interpolator."""
        pass

    def interpolate_ball_tracks(self, ball_tracks: Dict) -> Dict:
        """
        Interpolate missing ball positions in track dictionary.

        Args:
            ball_tracks: Dictionary {frame_idx: {ball_id: bbox}}
                        where bbox is [x1, y1, x2, y2]

        Returns:
            Dictionary with interpolated ball positions
        """
        if not ball_tracks:
            return {}

        # Get frame range
        frame_indices = sorted(ball_tracks.keys())
        if len(frame_indices) < 2:
            return ball_tracks  # Need at least 2 points to interpolate

        min_frame = frame_indices[0]
        max_frame = frame_indices[-1]

        # Create result dictionary
        interpolated_tracks = {}

        # Track ball ID (assuming single ball with ID 0 or first ID found)
        ball_id = None
        for frame_data in ball_tracks.values():
            if frame_data:
                ball_id = list(frame_data.keys())[0]
                break

        if ball_id is None:
            return ball_tracks

        # Extract detected positions and their frame indices
        detected_frames = []
        detected_bboxes = []

        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx in ball_tracks and ball_id in ball_tracks[frame_idx]:
                detected_frames.append(frame_idx)
                detected_bboxes.append(ball_tracks[frame_idx][ball_id])

        if len(detected_frames) < 2:
            return ball_tracks  # Not enough points for interpolation

        # Convert to numpy arrays for easier manipulation
        detected_frames_arr = np.array(detected_frames)
        detected_bboxes_arr = np.array(detected_bboxes)  # Shape: (N, 4)

        # Interpolate for each frame in the range
        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx in ball_tracks and ball_id in ball_tracks[frame_idx]:
                # Already have detection, keep it
                interpolated_tracks[frame_idx] = ball_tracks[frame_idx]
            else:
                # Need to interpolate
                # Find surrounding detected frames
                before_idx = detected_frames_arr[detected_frames_arr < frame_idx]
                after_idx = detected_frames_arr[detected_frames_arr > frame_idx]

                if len(before_idx) > 0 and len(after_idx) > 0:
                    # Can interpolate between two detections
                    frame_before = before_idx[-1]
                    frame_after = after_idx[0]

                    idx_before = np.where(detected_frames_arr == frame_before)[0][0]
                    idx_after = np.where(detected_frames_arr == frame_after)[0][0]

                    bbox_before = detected_bboxes_arr[idx_before]
                    bbox_after = detected_bboxes_arr[idx_after]

                    # Linear interpolation
                    t = (frame_idx - frame_before) / (frame_after - frame_before)
                    interpolated_bbox = bbox_before + t * (bbox_after - bbox_before)

                    interpolated_tracks[frame_idx] = {ball_id: interpolated_bbox}
                else:
                    # Cannot interpolate (no surrounding detections)
                    # Leave frame empty
                    pass

        return interpolated_tracks

    def interpolate_ball_positions(
        self,
        ball_tracks: Dict,
        max_gap: int = 10
    ) -> Dict:
        """
        Interpolate ball positions with gap size limit.

        Only interpolates when gap between detections is <= max_gap frames.
        This prevents interpolation across long periods without detection
        (e.g., when ball is out of frame or occluded).

        Args:
            ball_tracks: Dictionary {frame_idx: {ball_id: bbox}}
            max_gap: Maximum gap size (in frames) to interpolate across

        Returns:
            Dictionary with interpolated ball positions
        """
        if not ball_tracks:
            return {}

        # Get frame range
        frame_indices = sorted(ball_tracks.keys())
        if len(frame_indices) < 2:
            return ball_tracks

        min_frame = frame_indices[0]
        max_frame = frame_indices[-1]

        # Find ball ID
        ball_id = None
        for frame_data in ball_tracks.values():
            if frame_data:
                ball_id = list(frame_data.keys())[0]
                break

        if ball_id is None:
            return ball_tracks

        # Extract detected positions
        detected_frames = []
        detected_bboxes = []

        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx in ball_tracks and ball_id in ball_tracks[frame_idx]:
                detected_frames.append(frame_idx)
                detected_bboxes.append(ball_tracks[frame_idx][ball_id])

        if len(detected_frames) < 2:
            return ball_tracks

        detected_frames_arr = np.array(detected_frames)
        detected_bboxes_arr = np.array(detected_bboxes)

        # Interpolate with gap limit
        interpolated_tracks = {}

        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx in ball_tracks and ball_id in ball_tracks[frame_idx]:
                # Keep existing detection
                interpolated_tracks[frame_idx] = ball_tracks[frame_idx]
            else:
                # Check if we can interpolate
                before_idx = detected_frames_arr[detected_frames_arr < frame_idx]
                after_idx = detected_frames_arr[detected_frames_arr > frame_idx]

                if len(before_idx) > 0 and len(after_idx) > 0:
                    frame_before = before_idx[-1]
                    frame_after = after_idx[0]

                    gap = frame_after - frame_before

                    if gap <= max_gap:
                        # Gap is small enough, interpolate
                        idx_before = np.where(detected_frames_arr == frame_before)[0][0]
                        idx_after = np.where(detected_frames_arr == frame_after)[0][0]

                        bbox_before = detected_bboxes_arr[idx_before]
                        bbox_after = detected_bboxes_arr[idx_after]

                        t = (frame_idx - frame_before) / (frame_after - frame_before)
                        interpolated_bbox = bbox_before + t * (bbox_after - bbox_before)

                        interpolated_tracks[frame_idx] = {ball_id: interpolated_bbox}

        return interpolated_tracks

    def get_interpolation_stats(
        self,
        original_tracks: Dict,
        interpolated_tracks: Dict
    ) -> Dict:
        """
        Get statistics about the interpolation process.

        Args:
            original_tracks: Original ball tracks before interpolation
            interpolated_tracks: Ball tracks after interpolation

        Returns:
            Dictionary with interpolation statistics
        """
        original_frames = len(original_tracks)
        interpolated_frames = len(interpolated_tracks)
        added_frames = interpolated_frames - original_frames

        return {
            'original_detections': original_frames,
            'after_interpolation': interpolated_frames,
            'interpolated_frames': added_frames,
            'interpolation_rate': added_frames / interpolated_frames if interpolated_frames > 0 else 0
        }
