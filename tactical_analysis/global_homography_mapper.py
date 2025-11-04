"""
Global Homography Mapper for Soccer Field Tracking.

Accumulates keypoints from all video frames to build a single, robust homography
transformation. Provides confidence scores for field regions based on keypoint coverage.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tactical_analysis.sports_compat import (SoccerPitchConfiguration,
                                             ViewTransformer)


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

        # === ROBUST MULTI-TIER FALLBACK SYSTEM ===
        # Per-frame homography cache: {frame_idx: matrix}
        self.per_frame_matrices = {}

        # Temporal smoothing window (30 frames = 1 second at 30fps)
        self.temporal_window_size = 30
        self.frame_matrix_history = []  # List of (frame_idx, matrix) tuples

        # Last-known-good matrix cache
        self.last_valid_matrix = None
        self.last_valid_frame = None

        # Per-player position memory: {player_id: {'frame': int, 'position': array, 'pixel': array}}
        self.player_position_memory = {}

        # Debug statistics for multi-tier fallback
        self.debug_stats = {
            'per_frame_success': 0,
            'temporal_fallback': 0,
            'global_fallback': 0,
            'last_valid_fallback': 0,
            'player_memory_fallback': 0,
            'proportional_fallback': 0,
            'total_transformations': 0
        }

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
        """Get mapping from our 32 keypoints to sports library's points."""
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
            # Additional keypoints from 32-point model (29, 30, 31) - mapping TBD
            33,  # 29: keypoint_29 -> center point (placeholder)
            33,  # 30: keypoint_30 -> center point (placeholder)
            33,  # 31: keypoint_31 -> center point (placeholder)
        ])

    def add_frame_keypoints(self, frame_idx: int, detected_keypoints: Optional[np.ndarray]):
        """
        Accumulate keypoints from a single frame.

        Args:
            frame_idx: Frame index
            detected_keypoints: Array of shape (1, 32, 3) with [x, y, confidence]
                               or None if no keypoints detected
        """
        self.total_frames_processed += 1

        if detected_keypoints is None or detected_keypoints.shape[0] == 0:
            return

        keypoints = detected_keypoints[0]  # Take first detection (1, 32, 3) -> (32, 3)

        # Accumulate each keypoint with sufficient confidence AND spatial validity
        # CRITICAL FIX (from research code): Exclude invalid keypoints (x<=1, y<=1)
        # that corrupt homography calculation
        keypoints_added = 0
        for kp_idx in range(len(keypoints)):
            x, y, conf = keypoints[kp_idx]

            # Combined confidence + spatial filtering (matches HomographyTransformer)
            if conf > self.confidence_threshold and x > 1 and y > 1:
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

    def transform_points_robust(self, pixel_coords: np.ndarray, frame_idx: int = None,
                               player_id: int = None, video_dimensions: Tuple[int, int] = None,
                               per_frame_keypoints: Dict = None) -> Tuple[np.ndarray, str]:
        """
        ROBUST multi-tier transformation that NEVER returns null coordinates.

        Fallback hierarchy (6 tiers):
        1. Per-frame homography (if sufficient keypoints detected this frame)
        2. Temporal smoothing (average of last 30 frames' matrices)
        3. Global homography (if matrix_built and high confidence)
        4. Last-known-good matrix (cached from most recent successful transform)
        5. Per-player position memory (player's last known valid position)
        6. Proportional field estimation (ALWAYS works - conservative fallback)

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates
            frame_idx: Current frame index (for temporal/debug logging)
            player_id: Player ID (for per-player position memory)
            video_dimensions: (width, height) for proportional fallback
            per_frame_keypoints: Dict with keypoint detections for this frame
                                Format: {kp_idx: {'pixel': array, 'confidence': float}}

        Returns:
            Tuple of (pitch_coords in pitch units, fallback_method_used)
            ALWAYS returns valid coordinates - never returns None
        """
        if len(pixel_coords) == 0:
            return np.array([]).reshape(0, 2), 'empty'

        self.debug_stats['total_transformations'] += 1

        # === TIER 1: Per-frame homography ===
        if per_frame_keypoints and len(per_frame_keypoints) >= 4:
            try:
                matrix = self._build_per_frame_matrix(per_frame_keypoints)
                if matrix is not None and self._is_matrix_valid(matrix, pixel_coords):
                    # Cache this matrix
                    if frame_idx is not None:
                        self.per_frame_matrices[frame_idx] = matrix
                        self.frame_matrix_history.append((frame_idx, matrix))
                        if len(self.frame_matrix_history) > self.temporal_window_size:
                            self.frame_matrix_history.pop(0)
                        self.last_valid_matrix = matrix
                        self.last_valid_frame = frame_idx

                    result = self._apply_matrix(matrix, pixel_coords)
                    self.debug_stats['per_frame_success'] += 1

                    # Update player position memory
                    if player_id is not None and result is not None and len(result) > 0:
                        self._update_player_memory(player_id, frame_idx, result[0], pixel_coords[0])

                    return result, 'per_frame'
            except Exception as e:
                print(f"[DEBUG] Tier 1 failed (frame {frame_idx}): {e}")

        # === TIER 2: Temporal smoothing ===
        if len(self.frame_matrix_history) >= 3:  # Need at least 3 recent frames
            try:
                avg_matrix = self._get_temporal_average_matrix()
                if avg_matrix is not None and self._is_matrix_valid(avg_matrix, pixel_coords):
                    result = self._apply_matrix(avg_matrix, pixel_coords)
                    self.debug_stats['temporal_fallback'] += 1

                    if player_id is not None and result is not None and len(result) > 0:
                        self._update_player_memory(player_id, frame_idx, result[0], pixel_coords[0])

                    return result, 'temporal'
            except Exception as e:
                print(f"[DEBUG] Tier 2 failed (frame {frame_idx}): {e}")

        # === TIER 3: Global homography ===
        if self.matrix_built and self.global_matrix is not None:
            try:
                if self._is_matrix_valid(self.global_matrix, pixel_coords):
                    result = self._apply_matrix(self.global_matrix, pixel_coords)
                    self.debug_stats['global_fallback'] += 1

                    if player_id is not None and result is not None and len(result) > 0:
                        self._update_player_memory(player_id, frame_idx, result[0], pixel_coords[0])

                    return result, 'global'
            except Exception as e:
                print(f"[DEBUG] Tier 3 failed (frame {frame_idx}): {e}")

        # === TIER 4: Last-known-good matrix ===
        if self.last_valid_matrix is not None:
            try:
                if self._is_matrix_valid(self.last_valid_matrix, pixel_coords):
                    result = self._apply_matrix(self.last_valid_matrix, pixel_coords)
                    self.debug_stats['last_valid_fallback'] += 1

                    if player_id is not None and result is not None and len(result) > 0:
                        self._update_player_memory(player_id, frame_idx, result[0], pixel_coords[0])

                    return result, f'last_valid_frame_{self.last_valid_frame}'
            except Exception as e:
                print(f"[DEBUG] Tier 4 failed (frame {frame_idx}): {e}")

        # === TIER 5: Per-player position memory ===
        if player_id is not None and player_id in self.player_position_memory:
            memory = self.player_position_memory[player_id]
            # Only use if recent (within 30 frames)
            if frame_idx is not None and abs(frame_idx - memory['frame']) <= 30:
                self.debug_stats['player_memory_fallback'] += 1
                return np.array([memory['position']]), f'player_memory_frame_{memory["frame"]}'

        # === TIER 6: Proportional field estimation (ALWAYS WORKS) ===
        if video_dimensions is not None:
            result = self._proportional_estimate(pixel_coords, video_dimensions)
            self.debug_stats['proportional_fallback'] += 1
            print(f"[DEBUG] Tier 6 proportional fallback used (frame {frame_idx}, player {player_id})")
            return result, 'proportional'

        # Ultimate fallback: return center of field
        print(f"[WARNING] All tiers failed! Returning field center (frame {frame_idx}, player {player_id})")
        center_pitch = np.array([[6000.0, 3500.0]])  # Center of 12000x7000 pitch
        return center_pitch, 'emergency_center'

    def _build_per_frame_matrix(self, keypoints: Dict) -> Optional[np.ndarray]:
        """Build homography matrix from this frame's keypoints."""
        pixel_points = []
        pitch_points = []

        for kp_idx, kp_data in keypoints.items():
            if kp_data['confidence'] < self.confidence_threshold:
                continue

            # Map our keypoint index to pitch point
            if kp_idx < len(self.our_to_sports_mapping):
                sports_idx = self.our_to_sports_mapping[kp_idx]
                if sports_idx < len(self.all_pitch_points):
                    pixel_points.append(kp_data['pixel'])
                    pitch_points.append(self.all_pitch_points[sports_idx])

        if len(pixel_points) < 4:
            return None

        pixel_pts = np.array(pixel_points, dtype=np.float32)
        pitch_pts = np.array(pitch_points, dtype=np.float32)

        matrix, _ = cv2.findHomography(pixel_pts, pitch_pts, cv2.RANSAC, 5.0)
        return matrix

    def _get_temporal_average_matrix(self) -> Optional[np.ndarray]:
        """Average the last N homography matrices for temporal smoothing."""
        if len(self.frame_matrix_history) == 0:
            return None

        # Simple element-wise averaging
        matrices = [m for _, m in self.frame_matrix_history if m is not None]
        if len(matrices) == 0:
            return None

        avg_matrix = np.mean(matrices, axis=0)
        return avg_matrix

    def _is_matrix_valid(self, matrix: np.ndarray, pixel_coords: np.ndarray,
                        max_coord: float = 15000.0) -> bool:
        """
        Check if a matrix produces reasonable coordinates.

        Args:
            matrix: 3x3 homography matrix
            pixel_coords: Sample pixel coordinates to test
            max_coord: Maximum acceptable pitch coordinate (pitch is 12000x7000)

        Returns:
            True if matrix produces valid coordinates
        """
        if matrix is None:
            return False

        try:
            # Test with first coordinate
            test_point = pixel_coords[0:1].reshape(-1, 1, 2).astype(np.float32)
            result = cv2.perspectiveTransform(test_point, matrix).reshape(-1, 2)

            # Check if result is reasonable (within expanded bounds)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return False

            if np.any(np.abs(result) > max_coord):
                return False

            return True
        except:
            return False

    def _apply_matrix(self, matrix: np.ndarray, pixel_coords: np.ndarray) -> np.ndarray:
        """Apply homography matrix to pixel coordinates."""
        coords = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(coords, matrix)
        return result.reshape(-1, 2)

    def _proportional_estimate(self, pixel_coords: np.ndarray,
                              video_dimensions: Tuple[int, int]) -> np.ndarray:
        """
        Conservative proportional estimation: ALWAYS returns valid coordinates.
        Assumes field occupies central 60% of frame to avoid overestimation.
        """
        video_width, video_height = video_dimensions

        # Assume field occupies central 60% of frame
        field_fraction = 0.6
        margin = (1 - field_fraction) / 2

        # Normalize pixel coordinates to [0, 1]
        norm_x = (pixel_coords[:, 0] / video_width - margin) / field_fraction
        norm_y = (pixel_coords[:, 1] / video_height - margin) / field_fraction

        # Clamp to [0, 1]
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)

        # Scale to pitch coordinates (12000x7000)
        pitch_coords = np.zeros_like(pixel_coords, dtype=np.float32)
        pitch_coords[:, 0] = norm_x * 12000.0
        pitch_coords[:, 1] = norm_y * 7000.0

        return pitch_coords

    def _update_player_memory(self, player_id: int, frame_idx: int,
                              position: np.ndarray, pixel: np.ndarray):
        """Update per-player position memory for fallback."""
        self.player_position_memory[player_id] = {
            'frame': frame_idx,
            'position': position.copy(),
            'pixel': pixel.copy()
        }

    def print_debug_stats(self):
        """Print debug statistics about which fallback tiers were used."""
        total = self.debug_stats['total_transformations']
        if total == 0:
            print("[DEBUG] No transformations performed yet")
            return

        print(f"\n[DEBUG] Transformation Statistics ({total} total):")
        print(f"  Tier 1 (Per-frame):     {self.debug_stats['per_frame_success']:6d} ({100*self.debug_stats['per_frame_success']/total:5.1f}%)")
        print(f"  Tier 2 (Temporal):      {self.debug_stats['temporal_fallback']:6d} ({100*self.debug_stats['temporal_fallback']/total:5.1f}%)")
        print(f"  Tier 3 (Global):        {self.debug_stats['global_fallback']:6d} ({100*self.debug_stats['global_fallback']/total:5.1f}%)")
        print(f"  Tier 4 (Last-valid):    {self.debug_stats['last_valid_fallback']:6d} ({100*self.debug_stats['last_valid_fallback']/total:5.1f}%)")
        print(f"  Tier 5 (Player memory): {self.debug_stats['player_memory_fallback']:6d} ({100*self.debug_stats['player_memory_fallback']/total:5.1f}%)")
        print(f"  Tier 6 (Proportional):  {self.debug_stats['proportional_fallback']:6d} ({100*self.debug_stats['proportional_fallback']/total:5.1f}%)\n")

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
