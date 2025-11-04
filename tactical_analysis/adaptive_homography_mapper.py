"""Adaptive Homography Mapper for Moving Camera Soccer Analysis.

Builds per-frame homographies with intelligent fallbacks for tactical
camera recordings with panning/zooming motion.

STABILITY ENHANCEMENTS:
- Temporal homography smoothing (EMA) to prevent matrix instability
- Position validation with bounds checking
- Per-player position smoothing to reduce jitter
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from collections import defaultdict, deque
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
                 min_keypoints: int = 4, temporal_window: int = 15,
                 ema_alpha: float = 0.3, max_matrix_change: float = 50.0,
                 position_smoothing_buffer: int = 3):
        """
        Initialize adaptive homography mapper with stability features.

        Args:
            field_dimensions: (width, height) in meters (default: FIFA standard)
            min_keypoints: Minimum keypoints required for homography (must be >= 4)
            temporal_window: Frames to search for temporal fallback
            ema_alpha: EMA smoothing weight for new matrices (0.3 = 30% new, 70% previous)
            max_matrix_change: Maximum allowed matrix element change (reject if exceeded)
            position_smoothing_buffer: Number of positions to buffer per player for smoothing
        """
        self.field_width = field_dimensions[0]
        self.field_height = field_dimensions[1]
        self.min_keypoints = max(4, min_keypoints)  # Ensure at least 4 for homography
        self.temporal_window = temporal_window

        # STABILITY PARAMETERS
        self.ema_alpha = ema_alpha  # Lower = more smoothing (0.2-0.4 recommended)
        self.max_matrix_change = max_matrix_change  # Reject extreme changes
        self.position_smoothing_buffer = position_smoothing_buffer

        # Initialize pitch configuration with 32 dense keypoints
        self.CONFIG = SoccerPitchConfiguration()

        # Get all 32 pitch reference points
        self.all_pitch_points = np.array(self.CONFIG.vertices, dtype=np.float32)

        # Per-frame storage
        self.frame_homographies = {}  # {frame_idx: matrix}
        self.frame_keypoints = {}  # {frame_idx: [(pixel_x, pixel_y, pitch_x, pitch_y)]}
        self.frame_confidences = {}  # {frame_idx: confidence_score}

        # TEMPORAL SMOOTHING STORAGE
        self.H_previous = None  # Last stable homography matrix
        self.H_smooth = None    # Current smoothed homography
        self.H_buffer = deque(maxlen=5)  # Buffer of recent matrices for median filtering

        # PER-PLAYER POSITION SMOOTHING
        self.player_position_buffers = {}  # {player_id: deque of positions}

        # Statistics
        self.total_frames_processed = 0
        self.homography_success_count = 0
        self.temporal_fallback_count = 0
        self.proportional_fallback_count = 0
        self.matrix_rejected_count = 0  # New: Count unstable matrices rejected

    def add_frame_data(self, frame_idx: int, yolo_keypoints: Optional[np.ndarray]):
        """
        Add detected keypoints for a frame.

        Args:
            frame_idx: Frame index
            yolo_keypoints: YOLO pose keypoints (32, 3) or None
        """
        self.total_frames_processed += 1

        # Collect YOLO keypoints
        combined_keypoints = []

        # Add YOLO keypoints if available
        if yolo_keypoints is not None and yolo_keypoints.shape[0] > 0:
            keypoints = yolo_keypoints[0]  # Take first detection

            # DEBUG: Log raw keypoint data for first 3 frames
            if frame_idx < 3:
                total_kp = len(keypoints)
                spatial_valid = np.sum((keypoints[:, 0] > 1) & (keypoints[:, 1] > 1))
                print(f"[Adaptive Homography] Frame {frame_idx} DEBUG: "
                      f"Total keypoints: {total_kp}, Spatial valid (x>1,y>1): {spatial_valid}")

            for kp_idx in range(len(keypoints)):
                x, y, conf = keypoints[kp_idx]
                # MATCH RESEARCH CODE: ONLY spatial filtering, NO confidence threshold
                # Research code: mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
                # They do NOT filter by confidence - only by spatial validity
                if x > 1 and y > 1:
                    # Map to pitch point
                    pitch_idx = self._get_yolo_keypoint_mapping(kp_idx)
                    if pitch_idx is not None and pitch_idx < len(self.all_pitch_points):
                        pitch_point = self.all_pitch_points[pitch_idx]
                        combined_keypoints.append((x, y, pitch_point[0], pitch_point[1], conf))

        # Store keypoints
        self.frame_keypoints[frame_idx] = combined_keypoints

        # Try to build homography for this frame
        if len(combined_keypoints) >= self.min_keypoints:
            success = self._build_frame_homography(frame_idx, combined_keypoints)
            if success:
                self.homography_success_count += 1
                # Log first 5 successes to verify it's working
                if self.homography_success_count <= 5:
                    print(f"[Adaptive Homography] Frame {frame_idx}: ✓ Built homography from {len(combined_keypoints)} keypoints")
        else:
            # Log first 5 failures AND sample every 100 frames to see pattern
            if self.total_frames_processed <= 5 or frame_idx % 100 == 0:
                print(f"[Adaptive Homography] Frame {frame_idx}: Insufficient keypoints "
                      f"({len(combined_keypoints)}/{self.min_keypoints})")

    def _get_yolo_keypoint_mapping(self, kp_idx: int) -> Optional[int]:
        """
        Map YOLO keypoint index (32 keypoints) to pitch point index (35 points).

        The mapping ensures YOLO detected keypoints correspond to the correct
        pitch reference points from our 35-point configuration.
        """
        # Mapping from 32 YOLO keypoints to 35 pitch points (0-indexed)
        mapping = np.array([
            0,   # 0 -> Point 1 (top-left corner)
            1,   # 1 -> Point 2 (left penalty area top)
            2,   # 2 -> Point 3 (left goal area top)
            3,   # 3 -> Point 4 (left goal area bottom)
            4,   # 4 -> Point 5 (left penalty area bottom)
            5,   # 5 -> Point 6 (bottom-left corner)
            6,   # 6 -> Point 7 (left goal box top-right)
            7,   # 7 -> Point 8 (left goal box bottom-right)
            8,   # 8 -> Point 9 (left penalty spot)
            9,   # 9 -> Point 10 (left penalty box top-right)
            10,  # 10 -> Point 11
            11,  # 11 -> Point 12
            12,  # 12 -> Point 13 (left penalty box bottom-right)
            13,  # 13 -> Point 14 (center line top)
            14,  # 14 -> Point 15 (center circle top)
            15,  # 15 -> Point 16 (center circle bottom)
            16,  # 16 -> Point 17 (center line bottom)
            17,  # 17 -> Point 18 (right penalty box top-left)
            18,  # 18 -> Point 19
            19,  # 19 -> Point 20
            20,  # 20 -> Point 21 (right penalty box bottom-left)
            21,  # 21 -> Point 22 (right penalty spot)
            22,  # 22 -> Point 23 (right goal box top-left)
            23,  # 23 -> Point 24 (right goal box bottom-left)
            24,  # 24 -> Point 25 (top-right corner)
            25,  # 25 -> Point 26 (right penalty area top)
            26,  # 26 -> Point 27 (right goal area top)
            27,  # 27 -> Point 28 (right goal area bottom)
            28,  # 28 -> Point 29 (right penalty area bottom)
            29,  # 29 -> Additional keypoint (placeholder)
            30,  # 30 -> Additional keypoint (placeholder)
            31,  # 31 -> Additional keypoint (placeholder)
        ])
        if kp_idx < len(mapping):
            return int(mapping[kp_idx])
        return None

    def _build_frame_homography(self, frame_idx: int, keypoints: List[Tuple]) -> bool:
        """
        Build homography matrix for a single frame WITH TEMPORAL SMOOTHING.

        Implements stability improvements:
        1. Matrix change validation (reject extreme changes)
        2. Exponential moving average (EMA) smoothing
        3. Fallback to previous stable matrix if new one is unstable

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
            # MATCH RESEARCH CODE: Use DEFAULT cv2.findHomography (NO RANSAC)
            # Research: self.m, _ = cv2.findHomography(source, target)
            # They do NOT use RANSAC - just trust the spatially filtered keypoints
            H_new, _ = cv2.findHomography(pixel_points, pitch_points)

            if H_new is None:
                if frame_idx < 5:
                    print(f"[Adaptive Homography] Frame {frame_idx}: Homography calculation failed")
                return False

            # STABILITY CHECK 1: Validate matrix change magnitude
            if self.H_previous is not None:
                matrix_diff = np.abs(H_new - self.H_previous).max()

                # Reject extreme changes that cause position jumps
                if matrix_diff > self.max_matrix_change:
                    self.matrix_rejected_count += 1

                    # Log first 10 rejections and occasional samples
                    if self.matrix_rejected_count <= 10 or frame_idx % 100 == 0:
                        print(f"[Adaptive Homography] Frame {frame_idx}: ⚠️ REJECTED unstable matrix "
                              f"(max_diff: {matrix_diff:.1f} > threshold: {self.max_matrix_change}) "
                              f"- using previous stable matrix")

                    # Reuse previous stable matrix instead of this unstable one
                    matrix = self.H_previous.copy()
                    confidence = 0.8  # Slightly lower confidence for reused matrix
                else:
                    # STABILITY CHECK 2: Apply EMA temporal smoothing
                    if self.H_smooth is None:
                        # First frame - use as-is
                        self.H_smooth = H_new
                        matrix = H_new
                    else:
                        # Exponential Moving Average: H_smooth = alpha*H_new + (1-alpha)*H_smooth
                        self.H_smooth = self.ema_alpha * H_new + (1 - self.ema_alpha) * self.H_smooth
                        matrix = self.H_smooth

                    # Add to buffer for optional median filtering
                    self.H_buffer.append(H_new)

                    confidence = 1.0  # Full confidence for smoothed matrix

                    # Log first 5 successful smoothings
                    if self.homography_success_count < 5:
                        print(f"[Adaptive Homography] Frame {frame_idx}: ✓ Applied EMA smoothing "
                              f"(diff: {matrix_diff:.1f}, alpha: {self.ema_alpha})")
            else:
                # First frame ever - no smoothing possible
                matrix = H_new
                self.H_smooth = H_new
                confidence = 1.0

            # Update previous matrix for next frame comparison
            self.H_previous = matrix.copy()

            # SUCCESS - Store the smoothed homography matrix
            self.frame_homographies[frame_idx] = matrix
            self.frame_confidences[frame_idx] = confidence

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
        """Apply homography transformation to a point with validation."""
        try:
            point = np.array([[pixel_coord]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, matrix)
            field_coord = transformed[0][0]

            # Convert from pitch units to meters
            field_coord[0] = field_coord[0] * (self.field_width / 12000.0)
            field_coord[1] = field_coord[1] * (self.field_height / 7000.0)

            # CRITICAL VALIDATION: Reject results that are wildly out of bounds
            # Allow some margin (50m) for edge detection, but reject extreme outliers
            if (field_coord[0] < -50 or field_coord[0] > 155 or
                field_coord[1] < -50 or field_coord[1] > 118):
                # This transformation is corrupted - return None to trigger fallback
                return None

            return field_coord
        except:
            return None

    def validate_position(self, position: np.ndarray, previous_position: Optional[np.ndarray] = None,
                         dt: float = 0.04, max_speed_ms: float = 12.5) -> Tuple[bool, str]:
        """
        Validate transformed position for physical plausibility.

        Args:
            position: Current position [x, y] in meters
            previous_position: Previous position [x, y] in meters (optional)
            dt: Time delta between frames (default: 0.04s for 25fps)
            max_speed_ms: Maximum allowed speed in m/s (default: 12.5 m/s = 45 km/h)

        Returns:
            Tuple of (is_valid, reason)
        """
        # Bounds check with 10m safety margin
        if not (-10 <= position[0] <= self.field_width + 10 and
                -10 <= position[1] <= self.field_height + 10):
            return False, f"out_of_bounds ({position[0]:.1f}, {position[1]:.1f})"

        # Distance check if we have previous position
        if previous_position is not None:
            distance = np.linalg.norm(position - previous_position)
            max_distance = max_speed_ms * dt  # e.g., 12.5 m/s * 0.04s = 0.5m per frame

            if distance > max_distance:
                implied_speed_kmh = (distance / dt) * 3.6
                return False, f"impossible_jump ({distance:.2f}m = {implied_speed_kmh:.0f} km/h)"

        return True, "valid"

    def smooth_player_position(self, player_id: int, position_raw: np.ndarray) -> np.ndarray:
        """
        Apply per-player position smoothing using median filter.

        Similar to Roboflow's BallTracker but for all players.

        Args:
            player_id: Unique player/tracker ID
            position_raw: Raw transformed position [x, y]

        Returns:
            Smoothed position [x, y]
        """
        if player_id not in self.player_position_buffers:
            self.player_position_buffers[player_id] = deque(maxlen=self.position_smoothing_buffer)

        # Add to buffer
        self.player_position_buffers[player_id].append(position_raw)

        # Return median of buffered positions
        if len(self.player_position_buffers[player_id]) >= 2:
            positions = np.array(self.player_position_buffers[player_id])
            return np.median(positions, axis=0)
        else:
            # Not enough history - return raw
            return position_raw

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
        """Get processing statistics including stability metrics."""
        return {
            'total_frames_processed': self.total_frames_processed,
            'frames_with_homography': len(self.frame_homographies),
            'homography_success_rate': (
                self.homography_success_count / self.total_frames_processed
                if self.total_frames_processed > 0 else 0.0
            ),
            'temporal_fallback_count': self.temporal_fallback_count,
            'proportional_fallback_count': self.proportional_fallback_count,
            'matrix_rejected_count': self.matrix_rejected_count,  # NEW
            'matrix_rejection_rate': (
                self.matrix_rejected_count / self.total_frames_processed
                if self.total_frames_processed > 0 else 0.0
            ),  # NEW
            'smoothing_active': self.H_smooth is not None,  # NEW
        }

    def print_statistics(self):
        """Print processing statistics including stability metrics."""
        stats = self.get_statistics()
        print(f"\n[Adaptive Homography] Statistics:")
        print(f"  Total frames processed: {stats['total_frames_processed']}")
        print(f"  Frames with homography: {stats['frames_with_homography']}")
        print(f"  Success rate: {100*stats['homography_success_rate']:.1f}%")
        print(f"  Temporal fallbacks: {stats['temporal_fallback_count']}")
        print(f"  Proportional fallbacks: {stats['proportional_fallback_count']}")
        print(f"  Unstable matrices rejected: {stats['matrix_rejected_count']} "
              f"({100*stats['matrix_rejection_rate']:.1f}%)")  # NEW
        print(f"  EMA smoothing active: {stats['smoothing_active']}")  # NEW
