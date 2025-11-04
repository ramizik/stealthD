"""Adaptive Homography Mapper for Moving Camera Soccer Analysis.

Builds per-frame homographies with intelligent fallbacks for tactical
camera recordings with panning/zooming motion.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tactical_analysis.sports_compat import SoccerPitchConfiguration


class AdaptiveHomographyMapper:
    """
    Per-frame homography with temporal fallbacks for moving cameras.

    Handles tactical camera recordings with continuous panning by:
    1. Building homography per frame when sufficient features detected
    2. Interpolating from nearby frames when features insufficient
    3. Using camera motion compensation for temporal coherence
    4. Falling back to proportional scaling as last resort
    """

    def __init__(self, field_dimensions: Tuple[float, float] = (105, 68),
                 min_keypoints: int = 4, temporal_window: int = 15):
        """
        Initialize adaptive homography mapper.

        Args:
            field_dimensions: (width, height) in meters (default: FIFA standard)
            min_keypoints: Minimum keypoints required for homography
            temporal_window: Frames to search for temporal fallback
        """
        self.field_width = field_dimensions[0]
        self.field_height = field_dimensions[1]
        self.min_keypoints = min_keypoints
        self.temporal_window = temporal_window

        # Initialize pitch configuration
        self.CONFIG = SoccerPitchConfiguration()
        self.all_pitch_points = self._get_all_pitch_points()

        # Per-frame storage
        self.frame_homographies = {}  # {frame_idx: matrix}
        self.frame_keypoints = {}  # {frame_idx: [(pixel_x, pixel_y, pitch_x, pitch_y)]}
        self.frame_confidences = {}  # {frame_idx: confidence_score}

        # Statistics
        self.total_frames_processed = 0
        self.homography_success_count = 0
        self.temporal_fallback_count = 0
        self.proportional_fallback_count = 0

    def _get_all_pitch_points(self) -> np.ndarray:
        """Get all pitch reference points."""
        all_pitch_points = np.array(self.CONFIG.vertices)
        extra_pitch_points = np.array([
            [2932, 3500],  # Left semicircle rightmost
            [6000, 3500],  # Center point
            [9069, 3500],  # Right semicircle rightmost
        ])
        return np.concatenate((all_pitch_points, extra_pitch_points))

    def add_frame_data(self, frame_idx: int, yolo_keypoints: Optional[np.ndarray],
                       field_line_keypoints: List[Tuple[float, float]],
                       field_line_confidence: float):
        """
        Add detected keypoints for a frame.

        Args:
            frame_idx: Frame index
            yolo_keypoints: YOLO pose keypoints (29, 3) or None
            field_line_keypoints: Field line intersection points
            field_line_confidence: Confidence from field line detection
        """
        self.total_frames_processed += 1

        # Combine YOLO and field line keypoints
        combined_keypoints = []

        # Add YOLO keypoints if available
        if yolo_keypoints is not None and yolo_keypoints.shape[0] > 0:
            keypoints = yolo_keypoints[0]  # Take first detection
            for kp_idx in range(len(keypoints)):
                x, y, conf = keypoints[kp_idx]
                if conf > 0.3:  # Lower threshold for tactical videos
                    # Map to pitch point
                    pitch_idx = self._get_yolo_keypoint_mapping(kp_idx)
                    if pitch_idx is not None and pitch_idx < len(self.all_pitch_points):
                        pitch_point = self.all_pitch_points[pitch_idx]
                        combined_keypoints.append((x, y, pitch_point[0], pitch_point[1], conf))

        # Add field line keypoints (these are more reliable for tactical views)
        # Map field line intersections to nearest pitch reference points
        for pixel_point in field_line_keypoints:
            # For now, use high confidence for field line detections
            # TODO: Implement proper mapping from pixel intersections to pitch points
            combined_keypoints.append((pixel_point[0], pixel_point[1], 0, 0, field_line_confidence))

        # Store keypoints
        self.frame_keypoints[frame_idx] = combined_keypoints

        # Try to build homography for this frame
        if len(combined_keypoints) >= self.min_keypoints:
            success = self._build_frame_homography(frame_idx, combined_keypoints)
            if success:
                self.homography_success_count += 1
                print(f"[Adaptive Homography] Frame {frame_idx}: Built homography from "
                      f"{len(combined_keypoints)} keypoints")
        else:
            print(f"[Adaptive Homography] Frame {frame_idx}: Insufficient keypoints "
                  f"({len(combined_keypoints)}/{self.min_keypoints})")

    def _get_yolo_keypoint_mapping(self, kp_idx: int) -> Optional[int]:
        """Map YOLO keypoint index to pitch point index."""
        # Same mapping as GlobalHomographyMapper
        mapping = np.array([
            0, 1, 9, 4, 12, 2, 6, 3, 7, 5, 32, 13, 16, 14, 15, 33,
            24, 25, 17, 28, 20, 26, 22, 27, 23, 29, 34, 30, 31
        ])
        if kp_idx < len(mapping):
            return int(mapping[kp_idx])
        return None

    def _build_frame_homography(self, frame_idx: int, keypoints: List[Tuple]) -> bool:
        """
        Build homography matrix for a single frame.

        Args:
            frame_idx: Frame index
            keypoints: List of (pixel_x, pixel_y, pitch_x, pitch_y, confidence)

        Returns:
            True if successful, False otherwise
        """
        if len(keypoints) < self.min_keypoints:
            return False

        # Extract pixel and pitch points
        pixel_points = []
        pitch_points = []

        for kp in keypoints:
            pixel_points.append([kp[0], kp[1]])
            pitch_points.append([kp[2], kp[3]])

        pixel_points = np.array(pixel_points, dtype=np.float32)
        pitch_points = np.array(pitch_points, dtype=np.float32)

        try:
            # Build homography with RANSAC
            matrix, mask = cv2.findHomography(
                pixel_points,
                pitch_points,
                cv2.RANSAC,
                5.0
            )

            if matrix is None:
                return False

            # Calculate confidence based on inlier ratio
            inliers = np.sum(mask) if mask is not None else 0
            confidence = inliers / len(keypoints) if len(keypoints) > 0 else 0.0

            # Store if confidence is reasonable
            if confidence > 0.3:
                self.frame_homographies[frame_idx] = matrix
                self.frame_confidences[frame_idx] = float(confidence)
                return True

        except Exception as e:
            print(f"[Adaptive Homography] Frame {frame_idx}: Homography failed - {e}")

        return False

    def transform_point(self, frame_idx: int, pixel_coord: np.ndarray,
                       camera_motion: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, str, float]:
        """
        Transform pixel coordinate to field coordinate with fallbacks.

        Multi-tier strategy:
        1. Use frame homography if available
        2. Interpolate from nearby frames
        3. Use proportional scaling

        Args:
            frame_idx: Frame index
            pixel_coord: (x, y) in pixels
            camera_motion: Optional (dx, dy) camera movement

        Returns:
            Tuple of (field_coord, method, confidence)
        """
        # Compensate for camera motion first
        if camera_motion is not None:
            pixel_coord = pixel_coord.copy()
            pixel_coord[0] -= camera_motion[0]
            pixel_coord[1] -= camera_motion[1]

        # Tier 1: Per-frame homography
        if frame_idx in self.frame_homographies:
            matrix = self.frame_homographies[frame_idx]
            confidence = self.frame_confidences.get(frame_idx, 0.5)
            field_coord = self._apply_homography(pixel_coord, matrix)

            if field_coord is not None:
                return field_coord, 'per-frame', confidence

        # Tier 2: Temporal interpolation from nearby frames
        field_coord, conf = self._temporal_interpolation(frame_idx, pixel_coord)
        if field_coord is not None:
            self.temporal_fallback_count += 1
            return field_coord, 'temporal', conf

        # Tier 3: Proportional scaling fallback
        self.proportional_fallback_count += 1
        field_coord = self._proportional_scaling(pixel_coord)
        return field_coord, 'proportional', 0.3

    def _apply_homography(self, pixel_coord: np.ndarray, matrix: np.ndarray) -> Optional[np.ndarray]:
        """Apply homography transformation to a point."""
        try:
            point = np.array([[pixel_coord]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, matrix)
            field_coord = transformed[0][0]

            # Convert from pitch units to meters
            field_coord[0] = field_coord[0] * (self.field_width / 12000.0)
            field_coord[1] = field_coord[1] * (self.field_height / 7000.0)

            return field_coord
        except:
            return None

    def _temporal_interpolation(self, frame_idx: int, pixel_coord: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Interpolate transformation from nearby frames.

        Args:
            frame_idx: Current frame index
            pixel_coord: Pixel coordinate to transform

        Returns:
            Tuple of (field_coord, confidence) or (None, 0.0)
        """
        # Find nearest frames with valid homography
        nearby_frames = []
        for offset in range(1, self.temporal_window + 1):
            # Check before
            prev_frame = frame_idx - offset
            if prev_frame in self.frame_homographies:
                nearby_frames.append((prev_frame, offset))

            # Check after
            next_frame = frame_idx + offset
            if next_frame in self.frame_homographies:
                nearby_frames.append((next_frame, offset))

            # Stop if we found at least 2
            if len(nearby_frames) >= 2:
                break

        if not nearby_frames:
            return None, 0.0

        # Use closest frame
        nearby_frames.sort(key=lambda x: x[1])
        closest_frame, distance = nearby_frames[0]

        matrix = self.frame_homographies[closest_frame]
        field_coord = self._apply_homography(pixel_coord, matrix)

        if field_coord is not None:
            # Reduce confidence based on temporal distance
            base_conf = self.frame_confidences.get(closest_frame, 0.5)
            confidence = base_conf * np.exp(-distance / 10.0)  # Exponential decay
            return field_coord, float(confidence)

        return None, 0.0

    def _proportional_scaling(self, pixel_coord: np.ndarray,
                            video_dimensions: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        """
        Conservative proportional scaling fallback.

        Args:
            pixel_coord: Pixel coordinate
            video_dimensions: Video (width, height)

        Returns:
            Field coordinate in meters
        """
        video_width, video_height = video_dimensions

        # Assume field occupies central 60% of frame (conservative)
        field_visible_fraction = 0.6
        margin = (1 - field_visible_fraction) / 2

        # Normalize
        norm_x = (pixel_coord[0] / video_width - margin) / field_visible_fraction
        norm_y = (pixel_coord[1] / video_height - margin) / field_visible_fraction

        # Clamp to [0, 1]
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)

        # Scale to field dimensions
        field_coord = np.array([
            norm_x * self.field_width,
            norm_y * self.field_height
        ], dtype=np.float32)

        return field_coord

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'total_frames_processed': self.total_frames_processed,
            'frames_with_homography': len(self.frame_homographies),
            'homography_success_rate': (
                self.homography_success_count / self.total_frames_processed
                if self.total_frames_processed > 0 else 0.0
            ),
            'temporal_fallback_count': self.temporal_fallback_count,
            'proportional_fallback_count': self.proportional_fallback_count,
        }

    def print_statistics(self):
        """Print processing statistics."""
        stats = self.get_statistics()
        print(f"\n[Adaptive Homography] Statistics:")
        print(f"  Total frames processed: {stats['total_frames_processed']}")
        print(f"  Frames with homography: {stats['frames_with_homography']}")
        print(f"  Success rate: {100*stats['homography_success_rate']:.1f}%")
        print(f"  Temporal fallbacks: {stats['temporal_fallback_count']}")
        print(f"  Proportional fallbacks: {stats['proportional_fallback_count']}")
