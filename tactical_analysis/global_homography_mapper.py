"""
Global Homography Mapper for Soccer Field Tracking.

Accumulates keypoints from all video frames to build a single, robust homography
transformation. Provides confidence scores for field regions based on keypoint coverage.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tactical_analysis.sports_compat import ViewTransformer, SoccerPitchConfiguration


class GlobalHomographyMapper:
    """
    Build comprehensive field transformation from all video frames.

    Accumulates keypoint observations throughout video processing,
    then creates one robust homography matrix using RANSAC on all data.
    Provides confidence scoring for field regions based on keypoint coverage.
    """

    def __init__(self, confidence_threshold: float = 0.5,
                 field_dimensions: Tuple[float, float] = (105, 68)):
        """
        Initialize the global homography mapper.

        Args:
            confidence_threshold: Minimum confidence to accept keypoints (default: 0.5)
            field_dimensions: (width, height) in meters (default: 105x68 FIFA standard)
        """
        self.confidence_threshold = confidence_threshold
        self.field_width = field_dimensions[0]
        self.field_height = field_dimensions[1]

        # Initialize pitch configuration
        self.CONFIG = SoccerPitchConfiguration()
        self.all_pitch_points = self._get_all_pitch_points()
        self.our_to_sports_mapping = self._get_keypoint_mapping()

        # Storage for accumulated keypoint observations
        # Structure: {keypoint_idx: [{'frame': int, 'pixel': array, 'pitch': array, 'confidence': float}]}
        self.keypoint_observations = defaultdict(list)

        # Global homography matrix (built after accumulation)
        self.global_matrix = None
        self.matrix_built = False

        # Confidence map: 21x14 grid covering 105x68m field (5m cells)
        self.confidence_grid = None
        self.grid_cell_size = 5.0  # meters
        self.grid_width = int(np.ceil(self.field_width / self.grid_cell_size))
        self.grid_height = int(np.ceil(self.field_height / self.grid_cell_size))

        # Statistics
        self.total_frames_processed = 0
        self.total_keypoints_accumulated = 0
        self.frames_with_keypoints = 0

    def _get_all_pitch_points(self) -> np.ndarray:
        """Get all pitch reference points including extra points."""
        all_pitch_points = np.array(self.CONFIG.vertices)
        extra_pitch_points = np.array([
            [2932, 3500],  # Left Semicircle rightmost point
            [6000, 3500],  # Center Point
            [9069, 3500],  # Right semicircle rightmost point
        ])
        return np.concatenate((all_pitch_points, extra_pitch_points))

    def _get_keypoint_mapping(self) -> np.ndarray:
        """Get mapping from our 29 keypoints to sports library's points."""
        return np.array([
            0,   # 0: sideline_top_left -> corner
            1,   # 1: big_rect_left_top_pt1 -> left penalty
            9,   # 2: big_rect_left_top_pt2 -> left goal
            4,   # 3: big_rect_left_bottom_pt1 -> left goal
            12,  # 4: big_rect_left_bottom_pt2 -> left penalty
            2,   # 5: small_rect_left_top_pt1 -> left goal
            6,   # 6: small_rect_left_top_pt2 -> left goal
            3,   # 7: small_rect_left_bottom_pt1 -> left goal
            7,   # 8: small_rect_left_bottom_pt2 -> left goal
            5,   # 9: sideline_bottom_left -> corner
            32,  # 10: left_semicircle_right -> penalty arc
            13,  # 11: center_line_top -> center
            16,  # 12: center_line_bottom -> center
            14,  # 13: center_circle_top -> center circle
            15,  # 14: center_circle_bottom -> center circle
            33,  # 15: field_center -> center circle
            24,  # 16: sideline_top_right -> corner
            25,  # 17: big_rect_right_top_pt1 -> right penalty
            17,  # 18: big_rect_right_top_pt2 -> right penalty
            28,  # 19: big_rect_right_bottom_pt1 -> right penalty
            20,  # 20: big_rect_right_bottom_pt2 -> right penalty
            26,  # 21: small_rect_right_top_pt1 -> right goal
            22,  # 22: small_rect_right_top_pt2 -> right goal
            27,  # 23: small_rect_right_bottom_pt1 -> right goal
            23,  # 24: small_rect_right_bottom_pt2 -> right goal
            29,  # 25: sideline_bottom_right -> corner
            34,  # 26: right_semicircle_left -> penalty arc
            30,  # 27: center_circle_left -> center_circle_left
            31,  # 28: center_circle_right -> center_circle_right
        ])

    def add_frame_keypoints(self, frame_idx: int, detected_keypoints: Optional[np.ndarray]):
        """
        Accumulate keypoints from a single frame.

        Args:
            frame_idx: Frame index
            detected_keypoints: Array of shape (1, 29, 3) with [x, y, confidence]
                               or None if no keypoints detected
        """
        self.total_frames_processed += 1

        if detected_keypoints is None or detected_keypoints.shape[0] == 0:
            return

        keypoints = detected_keypoints[0]  # Take first detection (1, 29, 3) -> (29, 3)

        # Accumulate each keypoint with sufficient confidence
        keypoints_added = 0
        for kp_idx in range(len(keypoints)):
            x, y, conf = keypoints[kp_idx]

            if conf > self.confidence_threshold:
                # Get corresponding pitch point
                pitch_idx = self.our_to_sports_mapping[kp_idx]
                pitch_point = self.all_pitch_points[pitch_idx]

                # Store observation
                self.keypoint_observations[kp_idx].append({
                    'frame': frame_idx,
                    'pixel': np.array([x, y], dtype=np.float32),
                    'pitch': pitch_point,
                    'confidence': float(conf)
                })
                keypoints_added += 1

        if keypoints_added > 0:
            self.frames_with_keypoints += 1
            self.total_keypoints_accumulated += keypoints_added

    def build_global_homography(self) -> bool:
        """
        Build global homography matrix from all accumulated keypoints.
        Uses RANSAC for robustness against outliers.

        Returns:
            True if successful, False if insufficient data
        """
        print(f"\n[Global Homography] Building from {self.total_keypoints_accumulated} keypoint observations...")

        # Check if we have enough data
        if len(self.keypoint_observations) < 4:
            print(f"[Global Homography] ERROR: Only {len(self.keypoint_observations)} unique keypoints detected. Need at least 4.")
            return False

        # Aggregate all keypoint observations
        all_pixel_points = []
        all_pitch_points = []
        all_weights = []  # Weight by confidence

        for kp_idx, observations in self.keypoint_observations.items():
            for obs in observations:
                all_pixel_points.append(obs['pixel'])
                all_pitch_points.append(obs['pitch'])
                all_weights.append(obs['confidence'])

        all_pixel_points = np.array(all_pixel_points, dtype=np.float32)
        all_pitch_points = np.array(all_pitch_points, dtype=np.float32)

        print(f"[Global Homography] Total observations: {len(all_pixel_points)}")
        print(f"[Global Homography] Unique keypoints: {len(self.keypoint_observations)}")
        print(f"[Global Homography] Frames with keypoints: {self.frames_with_keypoints}/{self.total_frames_processed}")

        # Build homography using RANSAC
        try:
            self.global_matrix, mask = cv2.findHomography(
                all_pixel_points,
                all_pitch_points,
                cv2.RANSAC,
                5.0  # RANSAC threshold
            )

            if self.global_matrix is None:
                print("[Global Homography] ERROR: Failed to compute homography matrix")
                return False

            # Calculate inlier ratio
            inliers = np.sum(mask) if mask is not None else 0
            inlier_ratio = inliers / len(all_pixel_points) if len(all_pixel_points) > 0 else 0
            print(f"[Global Homography] Inliers: {inliers}/{len(all_pixel_points)} ({100*inlier_ratio:.1f}%)")

            self.matrix_built = True

            # Build confidence map
            self._build_confidence_map()

            return True

        except Exception as e:
            print(f"[Global Homography] ERROR: {e}")
            return False

    def _build_confidence_map(self):
        """
        Build confidence map showing which field regions have good keypoint coverage.
        Divides field into 5mx5m grid and calculates confidence per cell.
        """
        # Initialize confidence grid
        self.confidence_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # For each keypoint observation, increment nearby cells
        for kp_idx, observations in self.keypoint_observations.items():
            for obs in observations:
                # Convert pitch coordinates to meters
                pitch_x = obs['pitch'][0] * (self.field_width / 12000.0)
                pitch_y = obs['pitch'][1] * (self.field_height / 7000.0)

                # Find grid cell
                cell_x = int(pitch_x / self.grid_cell_size)
                cell_y = int(pitch_y / self.grid_cell_size)

                # Clamp to grid bounds
                cell_x = np.clip(cell_x, 0, self.grid_width - 1)
                cell_y = np.clip(cell_y, 0, self.grid_height - 1)

                # Increment cell and neighbors (gaussian-like spreading)
                self.confidence_grid[cell_y, cell_x] += 1.0 * obs['confidence']

                # Add to adjacent cells with falloff
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = cell_y + dy, cell_x + dx
                        if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                            self.confidence_grid[ny, nx] += 0.3 * obs['confidence']

        # Normalize to [0, 1] range
        # Threshold: 10 observations = full confidence
        self.confidence_grid = np.minimum(self.confidence_grid / 10.0, 1.0)

        # Calculate statistics
        high_conf_cells = np.sum(self.confidence_grid > 0.7)
        medium_conf_cells = np.sum((self.confidence_grid > 0.4) & (self.confidence_grid <= 0.7))
        low_conf_cells = np.sum(self.confidence_grid <= 0.4)
        total_cells = self.grid_width * self.grid_height

        print(f"[Global Homography] Confidence Map ({self.grid_width}x{self.grid_height} cells):")
        print(f"   High confidence (>0.7): {high_conf_cells}/{total_cells} ({100*high_conf_cells/total_cells:.1f}%)")
        print(f"   Medium confidence (0.4-0.7): {medium_conf_cells}/{total_cells} ({100*medium_conf_cells/total_cells:.1f}%)")
        print(f"   Low confidence (<0.4): {low_conf_cells}/{total_cells} ({100*low_conf_cells/total_cells:.1f}%)")

    def transform_points(self, pixel_coords: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Transform pixel coordinates to pitch coordinates using global homography.

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates

        Returns:
            Tuple of (transformed points in pitch coordinates, average confidence)
        """
        if not self.matrix_built:
            raise RuntimeError("Global homography not built yet. Call build_global_homography() first.")

        if len(pixel_coords) == 0:
            return np.array([]).reshape(0, 2), 0.0

        # Reshape for cv2.perspectiveTransform
        pixel_coords = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)

        # Apply transformation
        pitch_coords = cv2.perspectiveTransform(pixel_coords, self.global_matrix)
        pitch_coords = pitch_coords.reshape(-1, 2)

        # Calculate average confidence for these points (after converting to meters)
        confidences = []
        for point in pitch_coords:
            # Convert to meters
            x_meters = point[0] * (self.field_width / 12000.0)
            y_meters = point[1] * (self.field_height / 7000.0)

            # Get confidence from grid
            conf = self._get_point_confidence(x_meters, y_meters)
            confidences.append(conf)

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return pitch_coords, float(avg_confidence)

    def _get_point_confidence(self, x_meters: float, y_meters: float) -> float:
        """Get confidence score for a specific field position."""
        if self.confidence_grid is None:
            return 0.0

        # Find grid cell
        cell_x = int(x_meters / self.grid_cell_size)
        cell_y = int(y_meters / self.grid_cell_size)

        # Clamp to grid bounds
        cell_x = np.clip(cell_x, 0, self.grid_width - 1)
        cell_y = np.clip(cell_y, 0, self.grid_height - 1)

        return float(self.confidence_grid[cell_y, cell_x])

    def is_position_reliable(self, x_meters: float, y_meters: float,
                            min_confidence: float = 0.7) -> bool:
        """
        Check if a field position has sufficient keypoint coverage.

        Args:
            x_meters: X coordinate in meters (0-105)
            y_meters: Y coordinate in meters (0-68)
            min_confidence: Minimum confidence threshold (default: 0.7)

        Returns:
            True if position is reliable, False otherwise
        """
        confidence = self._get_point_confidence(x_meters, y_meters)
        return confidence >= min_confidence

    def get_field_confidence_map(self) -> np.ndarray:
        """
        Get the confidence map for visualization.

        Returns:
            Array of shape (grid_height, grid_width) with confidence values [0, 1]
        """
        if self.confidence_grid is None:
            return np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        return self.confidence_grid.copy()

    def get_statistics(self) -> Dict:
        """
        Get statistics about the global homography mapper.

        Returns:
            Dictionary with various statistics
        """
        if self.confidence_grid is not None:
            high_conf = np.sum(self.confidence_grid > 0.7)
            total_cells = self.grid_width * self.grid_height
            coverage_percent = 100.0 * high_conf / total_cells if total_cells > 0 else 0.0
        else:
            coverage_percent = 0.0

        return {
            'total_frames_processed': self.total_frames_processed,
            'frames_with_keypoints': self.frames_with_keypoints,
            'total_keypoints_accumulated': self.total_keypoints_accumulated,
            'unique_keypoints_detected': len(self.keypoint_observations),
            'matrix_built': self.matrix_built,
            'high_confidence_coverage_percent': coverage_percent,
            'grid_dimensions': (self.grid_width, self.grid_height)
        }
