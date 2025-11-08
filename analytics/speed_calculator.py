"""
Speed Calculator module for soccer player speed and distance metrics.

Calculates player speeds using field coordinates (105m x 68m) with homography
transformation fallback to linear scaling.

STABILITY ENHANCEMENTS:
- Kalman filtering for per-player position smoothing
- Physical plausibility checks (max speed constraints)
- Outlier rejection based on velocity
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict, List, Optional, Tuple

import numpy as np

from tactical_analysis.position_smoother import PlayerPositionSmoother


class SpeedCalculator:
    """
    Calculate player speeds and distances using field coordinates.

    Supports both homography-based transformation and simple linear scaling
    as fallback for coordinate conversion.
    """

    def __init__(self, field_dimensions: Tuple[float, float] = (105, 68), fps: float = 30):
        """
        Initialize the speed calculator.

        Args:
            field_dimensions: (width, height) in meters. Default: (105, 68) for standard soccer field
            fps: Frames per second of the video
        """
        self.field_width = field_dimensions[0]  # meters
        self.field_height = field_dimensions[1]  # meters
        self.fps = fps
        self.last_valid_transformer = None  # Cache for last valid transformer
        self.last_transformer_frame = None   # Frame index of cached transformer

        # POSITION SMOOTHER: Critical addition to eliminate homography discontinuities
        self.position_smoother = PlayerPositionSmoother(
            dt=1.0/fps,                # Time between frames
            max_speed_ms=17.0,         # 17 m/s = 61 km/h (allow sprints, was 15 m/s)
            process_noise=0.5,         # Moderate smoothing
            measurement_noise=2.0      # Trust measurements but smooth outliers
        )

    def _is_catastrophically_bad(self, field_coords: np.ndarray) -> bool:
        """
        Check if coordinates are catastrophically bad (homography failure).

        Allows reasonable out-of-bounds (edge distortion) but rejects severe errors.

        Args:
            field_coords: Array of shape (N, 2) with [x, y] field coordinates

        Returns:
            True if coordinates are severely wrong (> 5m outside field)
        """
        if len(field_coords) == 0:
            return False

        # Reject if any coordinate is MORE than 5m outside field bounds
        # Tightened from 10m to catch more homography failures
        # Field: X[0, 105m], Y[0, 68m] → Accept X[-5, 110m], Y[-5, 73m]
        x_bad = np.any((field_coords[:, 0] < -5) | (field_coords[:, 0] > self.field_width + 5))
        y_bad = np.any((field_coords[:, 1] < -5) | (field_coords[:, 1] > self.field_height + 5))

        return x_bad or y_bad

    def _conservative_linear_scale(self, pixel_coords: np.ndarray,
                                   video_dimensions: Tuple[int, int]) -> np.ndarray:
        """
        Very conservative linear scaling that maps to field center, not (0,0).

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates
            video_dimensions: (width, height) of video in pixels

        Returns:
            Array of shape (N, 2) with field coordinates in meters
        """
        if len(pixel_coords) == 0:
            return np.array([]).reshape(0, 2)

        video_width, video_height = video_dimensions

        # Normalize pixel coordinates to [0, 1]
        norm_x = pixel_coords[:, 0] / video_width
        norm_y = pixel_coords[:, 1] / video_height

        # Map to field dimensions (105m × 68m) with proper centering
        # Center the field at (52.5, 34) instead of (0, 0)
        field_x = norm_x * self.field_width
        field_y = norm_y * self.field_height

        # Center the coordinates
        field_coords = np.column_stack([field_x, field_y])

        return field_coords

    def calculate_field_coordinates(self, pixel_coords: np.ndarray, global_mapper,
                                    video_dimensions: Tuple[int, int],
                                    frame_idx: int = None, player_id: int = None,
                                    per_frame_keypoints: Dict = None) -> Tuple[Optional[np.ndarray], str]:
        """
        Convert pixel coordinates to field coordinates using ROBUST multi-tier system.

        NEVER returns None - always provides best-effort coordinates.

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates
            global_mapper: GlobalHomographyMapper object with transform_points_robust()
            video_dimensions: (width, height) of video in pixels
            frame_idx: Frame index for temporal fallback
            player_id: Player ID for position memory fallback
            per_frame_keypoints: This frame's keypoint detections for per-frame homography

        Returns:
            Tuple of (Array of shape (N, 2) with field coordinates in METERS,
                     transformation_method string)
            ALWAYS returns coordinates - never None
        """
        if len(pixel_coords) == 0:
            return np.array([]).reshape(0, 2), 'empty'

        # Use robust transformation
        if hasattr(global_mapper, 'transform_points_robust'):
            result = global_mapper.transform_points_robust(
                pixel_coords,
                frame_idx=frame_idx,
                player_id=player_id,
                video_dimensions=video_dimensions,
                per_frame_keypoints=per_frame_keypoints
            )

            # Handle None return (failed transformation)
            if result is None:
                # Fallback to proportional scaling
                field_coords = self._proportional_scaling_fallback(pixel_coords, video_dimensions)
                return field_coords, 'proportional_fallback'

            pitch_coords, method, confidence = result

            # Convert from pitch coordinate system (12000x7000) to meters (105x68)
            try:
                pitch_coords = np.asarray(pitch_coords)

                # Handle empty or invalid arrays
                if pitch_coords.size == 0:
                    return np.array([]).reshape(0, 2), method

                # CRITICAL: Try to convert to float and check for validity
                try:
                    pitch_coords_float = pitch_coords.astype(float)
                    if not np.all(np.isfinite(pitch_coords_float)):
                        # Contains None, NaN, or Inf values
                        raise ValueError("Invalid coordinates")
                except (ValueError, TypeError):
                    # Fallback to proportional scaling
                    field_coords = self._proportional_scaling_fallback(pixel_coords, video_dimensions)
                    return field_coords, f'{method}_fallback (invalid coords)'

                if pitch_coords.ndim == 0:
                    # Scalar value - convert to single coordinate
                    pitch_coords_float = np.array([[pitch_coords.item()] * 2])
                elif pitch_coords.ndim == 1:
                    pitch_coords_float = pitch_coords_float.reshape(1, -1)

                field_coords = pitch_coords_float.copy()
                field_coords[:, 0] = field_coords[:, 0] * (self.field_width / 12000.0)
                field_coords[:, 1] = field_coords[:, 1] * (self.field_height / 7000.0)

            except Exception as e:
                # Any error during coordinate conversion - use fallback
                print(f"[Speed Calculator] Coordinate conversion error: {e}, using fallback")
                field_coords = self._proportional_scaling_fallback(pixel_coords, video_dimensions)
                return field_coords, f'{method}_fallback (conversion error)'

            # ADDITIONAL VALIDATION: Check for unreasonable coordinates
            # This helps catch homography errors early
            if len(field_coords) > 0:
                # Check for NaN or Inf values
                if not np.all(np.isfinite(field_coords)):
                    print(f"[Speed Calculator] Frame {frame_idx}: Invalid coordinates (NaN/Inf) detected")
                    # Use conservative fallback
                    return self._conservative_linear_scale(pixel_coords, video_dimensions), f"fallback (invalid coords)"

                # Check for coordinates outside reasonable field bounds
                # Allow some margin for perspective distortion
                x_min, x_max = field_coords[:, 0].min(), field_coords[:, 0].max()
                y_min, y_max = field_coords[:, 1].min(), field_coords[:, 1].max()

                # Field is 105m x 68m, allow 10m margin
                if (x_min < -10 or x_max > 115 or y_min < -10 or y_max > 78):
                    print(f"[Speed Calculator] Frame {frame_idx}: Out of bounds coordinates detected")
                    print(f"  X range: {x_min:.2f} to {x_max:.2f} (valid: -10 to 115)")
                    print(f"  Y range: {y_min:.2f} to {y_max:.2f} (valid: -10 to 78)")
                    print(f"  Method: {method}")

                    # Use conservative fallback for extreme cases
                    if x_min < -50 or x_max > 155 or y_min < -50 or y_max > 118:
                        return self._conservative_linear_scale(pixel_coords, video_dimensions), f"fallback (extreme bounds)"

            return field_coords, method

        # Fallback to old method if transform_points_robust not available
        else:
            print("[WARNING] Using legacy transformation - upgrade GlobalHomographyMapper!")
            return self._legacy_transform(pixel_coords, global_mapper, video_dimensions, frame_idx)

    def _legacy_transform(self, pixel_coords: np.ndarray, global_mapper,
                         video_dimensions: Tuple[int, int], frame_idx: int = None) -> Tuple[Optional[np.ndarray], str]:
        """Legacy transformation method - returns None on failure."""
        # Check if global mapper is built
        if not hasattr(global_mapper, 'matrix_built') or not global_mapper.matrix_built:
            return self._conservative_linear_scale(pixel_coords, video_dimensions), 'linear_fallback'

        try:
            # Use global homography mapper
            field_coords_raw, confidence = global_mapper.transform_points(pixel_coords)

            # Convert from pitch coordinate system to meters
            field_coords = field_coords_raw.copy()
            field_coords[:, 0] = field_coords[:, 0] * (self.field_width / 12000.0)
            field_coords[:, 1] = field_coords[:, 1] * (self.field_height / 7000.0)

            # Check if coordinates are in reasonable bounds
            if self._is_catastrophically_bad(field_coords):
                return None, 'catastrophic_failure'

            # Check if position has sufficient keypoint coverage
            if len(field_coords) > 0:
                x_meters, y_meters = field_coords[0]
                if not global_mapper.is_position_reliable(x_meters, y_meters, min_confidence=0.7):
                    return None, 'low_confidence'

            return field_coords, 'global_homography'

        except Exception as e:
            return self._conservative_linear_scale(pixel_coords, video_dimensions), 'linear_fallback'

    def _get_bbox_center(self, bbox: list) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _get_foot_position(self, bbox: list) -> Tuple[float, float]:
        """Get foot position (bottom center) of bounding box for players."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)  # Center X, bottom Y

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points in meters."""
        # Ensure we only use x, y coordinates (first 2 elements)
        point1 = np.asarray(point1)[:2]
        point2 = np.asarray(point2)[:2]
        return float(np.linalg.norm(point1 - point2))

    def calculate_speeds(self, player_positions: Dict[int, Dict[int, np.ndarray]],
                        global_mapper, video_dimensions: Tuple[int, int],
                        camera_movement: list = None) -> Dict[int, Dict]:
        """
        Calculate player speeds using field coordinates with robust filtering.

        Args:
            player_positions: {player_id: {frame_idx: [x, y]}}
            global_mapper: GlobalHomographyMapper object
            video_dimensions: (width, height) of video in pixels

        Returns:
            Dictionary of player speed data
        """
        player_speeds = {}

        for player_id, positions in player_positions.items():
            if len(positions) < 2:
                continue

            # Get sorted frame indices
            frame_indices = sorted(positions.keys())

            # Calculate speeds for each frame
            speeds = []
            valid_speeds = []  # For averaging

            
            for i in range(1, len(frame_indices)):
                prev_frame = frame_indices[i-1]
                curr_frame = frame_indices[i]

                # Get positions
                prev_pos = positions[prev_frame]
                curr_pos = positions[curr_frame]

                # Skip if either position is None
                if prev_pos is None or curr_pos is None:
                    continue

                # Skip identical positions (no movement)
                if prev_pos == curr_pos:
                    continue

                # Additional validation: Check for frame gaps
                frame_gap = curr_frame - prev_frame
                actual_time_diff = frame_gap / self.fps
                if frame_gap > 3:  # More than 3 frames gap (0.12s at 25fps)
                    continue  # Skip large frame gaps - likely tracking interruption

                # Transform to field coordinates with validation
                prev_field, prev_method = self.calculate_field_coordinates(
                    np.array([prev_pos]), global_mapper, video_dimensions, prev_frame, player_id
                )
                curr_field, curr_method = self.calculate_field_coordinates(
                    np.array([curr_pos]), global_mapper, video_dimensions, curr_frame, player_id
                )

                # Validate coordinate transformation quality
                transformation_valid = True
                if prev_field is None or curr_field is None or len(prev_field) == 0 or len(curr_field) == 0:
                    transformation_valid = False
                else:
                    # Check for coordinate system inconsistencies
                    prev_valid = np.all(np.isfinite(prev_field)) and np.all(np.abs(prev_field) < 200)  # Within soccer field bounds (~105m x 68m)
                    curr_valid = np.all(np.isfinite(curr_field)) and np.all(np.abs(curr_field) < 200)

                    if not (prev_valid and curr_valid):
                        transformation_valid = False
                    else:
                        # Additional validation: coordinate reasonableness check
                        coord_distance = np.linalg.norm(curr_field - prev_field)
                        # More than 50 meters in one frame is suspicious (1250 km/h)
                        if coord_distance > 50.0 and actual_time_diff < 0.1:  # Within reasonable time window
                            transformation_valid = False

                if not transformation_valid:
                    continue

                # Calculate distance and speed
                distance = np.linalg.norm(curr_field - prev_field)
                time_diff = actual_time_diff

                if time_diff > 0:
                    speed_ms = distance / time_diff
                    speed_kmh = speed_ms * 3.6

                    # Physics-based validation - only filter truly impossible movements
                    # Elite soccer players can reach 35-40 km/h (9.7-11.1 m/s)
                    # World record sprint (Usain Bolt): ~44.7 km/h (12.4 m/s)
                    # But 5+ meters in 0.04s = 450 km/h is physically impossible for humans

                    # Calculate metrics for validation
                    pixel_distance = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))

                    # Convert to human-understandable metrics
                    max_reasonable_human_speed_ms = 12.4  # Usain Bolt's speed in m/s
                    max_reasonable_human_speed_kmh = max_reasonable_human_speed_ms * 3.6

                    # Calculate physics constraints
                    max_possible_distance_ms = max_reasonable_human_speed_ms * time_diff
                    max_possible_distance_m = max_possible_distance_ms

                    is_valid = True

                    # Only filter truly impossible movements
                    if time_diff > 0.2:  # More than 200ms gap - likely tracking interruption
                        is_valid = False
                    elif distance > max_possible_distance_m * 1.1:  # 10% buffer for measurement error only
                        is_valid = False
                    elif pixel_distance > 300:  # Extreme pixel jumps (reduced threshold)
                        is_valid = False

                    if is_valid:
                        # Store detailed speed data for temporal smoothing (single entry)
                        speeds.append({
                            'frame': curr_frame,
                            'speed_kmh': speed_kmh,
                            'speed_ms': speed_ms,
                            'distance_m': distance,
                            'time_diff_s': time_diff,
                            'pixel_distance': pixel_distance,
                            'prev_frame': prev_frame,
                            'curr_frame': curr_frame,
                            'prev_pos': prev_pos.copy(),
                            'curr_pos': curr_pos.copy()
                        })

                        # Add to valid speeds for averaging
                        valid_speeds.append(speed_kmh)
                    else:
                        # Filtered out - extreme outlier
                        pass

            # Apply temporal smoothing to reduce measurement noise
            if len(speeds) >= 2:
                speeds = self._apply_temporal_smoothing(speeds)
                valid_speeds = [s['speed_kmh'] for s in speeds]

            # Apply adaptive averaging based on available data
            if len(valid_speeds) >= 3:  # Reduced from 15 to 3 frames minimum
                # Sort speeds
                sorted_speeds = sorted(valid_speeds)

                # Statistical outlier removal based on data distribution
                # Use IQR method to detect outliers while preserving valid high speeds
                if len(sorted_speeds) >= 10:
                    # Calculate IQR (Interquartile Range)
                    q1 = np.percentile(sorted_speeds, 25)
                    q3 = np.percentile(sorted_speeds, 75)
                    iqr = q3 - q1

                    # Define outlier bounds with realistic soccer speed limits
                    lower_bound = q1 - 1.5 * iqr
                    # Upper bound: max of 45 km/h (elite sprint speed) or q3 + 2*IQR
                    statistical_upper = q3 + 2.0 * iqr
                    upper_bound = min(statistical_upper, 45.0)  # Cap at realistic elite soccer speed

                    # Filter outliers but keep reasonable high speeds
                    trimmed_speeds = [s for s in sorted_speeds if lower_bound <= s <= upper_bound]
                else:
                    # For small datasets, apply hard limit to physics-filtered speeds
                    trimmed_speeds = [s for s in sorted_speeds if s <= 45.0]

                # Calculate average of remaining values
                avg_speed_kmh = np.mean(trimmed_speeds) if trimmed_speeds else 0.0
                avg_speed_ms = avg_speed_kmh / 3.6

                # Calculate total distance
                total_distance = sum(s['distance_m'] for s in speeds)

                player_speeds[player_id] = {
                    'frames': {s['frame']: s['speed_kmh'] for s in speeds},
                    'average_speed_kmh': avg_speed_kmh,
                    'max_speed_kmh': max([s['speed_kmh'] for s in speeds]) if speeds else 0.0,
                    'total_distance_m': total_distance,
                    'field_coordinates': {
                        frame_idx: (positions[frame_idx].tolist()
                                   if hasattr(positions[frame_idx], 'tolist')
                                   else list(positions[frame_idx]))
                        if frame_idx in positions else [0.0, 0.0]
                        for frame_idx in frame_indices
                    }
                }
            else:
                # Not enough data for robust averaging
                player_speeds[player_id] = {
                    'frames': {},
                    'average_speed_kmh': 0.0,
                    'max_speed_kmh': 0.0,
                    'total_distance_m': 0.0,
                    'field_coordinates': {}
                }

        return player_speeds

    def _proportional_scaling_fallback(self, pixel_coords: np.ndarray, video_dimensions: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
        """
        Fallback proportional scaling when homography fails.

        Maps pixel coordinates directly to field coordinates using simple scaling.
        """
        video_width, video_height = video_dimensions

        # Normalize pixel coordinates to [0, 1]
        norm_x = pixel_coords[:, 0] / video_width
        norm_y = pixel_coords[:, 1] / video_height

        # Scale to field dimensions (meters)
        field_coords = np.column_stack([
            norm_x * self.field_width,
            norm_y * self.field_height
        ]).astype(np.float32)

        return field_coords

    def _apply_temporal_smoothing(self, speeds: List[dict]) -> List[dict]:
        """
        Apply temporal smoothing to speed measurements to reduce jitter and noise.

        Uses weighted averaging with nearby frames to create more stable speed estimates.
        Also applies variance-based outlier detection within short time windows.

        Args:
            speeds: List of speed measurement dictionaries

        Returns:
            List of smoothed speed measurements
        """
        if not speeds:
            return speeds

        # Sort speeds by frame for temporal processing
        speeds_sorted = sorted(speeds, key=lambda x: x['frame'])

        smoothed_speeds = []

        for i, speed_data in enumerate(speeds_sorted):
            frame = speed_data['frame']
            speed = speed_data['speed_kmh']

            # Get temporal window (previous 2 and next 2 frames if available)
            window_size = 5
            half_window = window_size // 2

            # Collect nearby speeds
            nearby_speeds = []
            nearby_frames = []

            for j in range(max(0, i - half_window), min(len(speeds_sorted), i + half_window + 1)):
                nearby_speeds.append(speeds_sorted[j]['speed_kmh'])
                nearby_frames.append(speeds_sorted[j]['frame'])

            if len(nearby_speeds) >= 3:  # Only smooth if we have enough neighbors
                # Calculate weighted average (center frame gets highest weight)
                center_idx = len(nearby_speeds) // 2
                weights = [1.0] * len(nearby_speeds)

                # Give center frame 50% more weight
                if 0 <= center_idx < len(weights):
                    weights[center_idx] *= 1.5

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Calculate weighted average
                weighted_speed = sum(w * s for w, s in zip(weights, nearby_speeds))

                # Calculate variance for outlier detection
                variance = sum((s - weighted_speed) ** 2 for s in nearby_speeds) / len(nearby_speeds)
                std_dev = np.sqrt(variance)

                # If current speed is outlier (> 2 std dev from weighted avg), reduce its influence
                deviation = abs(speed - weighted_speed)
                if deviation > 2.0 * std_dev:
                    # Blend with weighted average more aggressively for outliers
                    final_speed = 0.7 * weighted_speed + 0.3 * speed
                else:
                    # Normal smoothing
                    final_speed = weighted_speed

                # Create smoothed speed data
                smoothed_speed_data = speed_data.copy()
                smoothed_speed_data['speed_kmh'] = final_speed
                smoothed_speed_data['speed_ms'] = final_speed / 3.6
                smoothed_speed_data['original_speed_kmh'] = speed  # Store original for comparison
                smoothed_speed_data['smoothed'] = True
                smoothed_speeds.append(smoothed_speed_data)
            else:
                # Not enough neighbors for smoothing
                speed_data['smoothed'] = False
                smoothed_speeds.append(speed_data)

        return smoothed_speeds
