"""
Speed Calculator module for soccer player speed and distance metrics.

Calculates player speeds using field coordinates (105m x 68m) with homography
transformation fallback to linear scaling.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict, Optional, Tuple

import numpy as np


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
        Very conservative linear scaling that assumes only the central portion
        of the frame contains the field. This prevents massive overestimation
        when homography is unavailable.

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates
            video_dimensions: (width, height) of video in pixels

        Returns:
            Array of shape (N, 2) with field coordinates in meters
        """
        if len(pixel_coords) == 0:
            return np.array([]).reshape(0, 2)

        video_width, video_height = video_dimensions

        # Very conservative: assume field occupies only central 50% of frame
        # This underestimates rather than overestimates distances
        field_visible_fraction = 0.5
        margin = (1 - field_visible_fraction) / 2

        # Normalize pixel coordinates
        norm_x = (pixel_coords[:, 0] / video_width - margin) / field_visible_fraction
        norm_y = (pixel_coords[:, 1] / video_height - margin) / field_visible_fraction

        # Clamp to [0, 1] to keep within field boundaries
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)

        # Scale to field dimensions
        field_coords = np.zeros_like(pixel_coords, dtype=np.float32)
        field_coords[:, 0] = norm_x * self.field_width
        field_coords[:, 1] = norm_y * self.field_height

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
            pitch_coords, method = global_mapper.transform_points_robust(
                pixel_coords,
                frame_idx=frame_idx,
                player_id=player_id,
                video_dimensions=video_dimensions,
                per_frame_keypoints=per_frame_keypoints
            )

            # Convert from pitch coordinate system (12000x7000) to meters (105x68)
            field_coords = pitch_coords.copy()
            field_coords[:, 0] = field_coords[:, 0] * (self.field_width / 12000.0)
            field_coords[:, 1] = field_coords[:, 1] * (self.field_height / 7000.0)

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
        return float(np.linalg.norm(point1 - point2))

    def calculate_speeds(self, player_tracks: Dict, adaptive_mapper,
                        video_dimensions: Tuple[int, int], camera_movement: list = None) -> Dict:
        """
        Calculate speeds for all players across all frames using adaptive per-frame transformation.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: [x1, y1, x2, y2]}}
            adaptive_mapper: AdaptiveHomographyMapper object with per-frame homographies
            video_dimensions: (width, height) of video in pixels
            camera_movement: List of [x_movement, y_movement] per frame

        Returns:
            Dictionary with structure:
            {
                player_id: {
                    'frames': {frame_idx: speed_kmh},
                    'average_speed_kmh': float,
                    'max_speed_kmh': float,
                    'total_distance_m': float,
                    'field_coordinates': {frame_idx: [x, y]}
                }
            }
        """
        player_speeds = {}

        print(f"\n[SPEED DEBUG] Starting speed calculation with ADAPTIVE per-frame transformation:")
        print(f"   Video dimensions: {video_dimensions}")
        print(f"   Field dimensions: {self.field_width}m x {self.field_height}m")
        print(f"   FPS: {self.fps}")

        # Get adaptive mapper statistics
        if hasattr(adaptive_mapper, 'get_statistics'):
            mapper_stats = adaptive_mapper.get_statistics()
            print(f"   Adaptive mapper statistics:")
            print(f"     - Frames with homography: {mapper_stats.get('frames_with_homography', 0)}")
            print(f"     - Temporal interpolation rate: {mapper_stats.get('temporal_interpolation_rate', 0):.1f}%")
            print(f"     - Proportional fallback rate: {mapper_stats.get('proportional_fallback_rate', 0):.1f}%")

        # Get all unique player IDs
        all_player_ids = set()
        for frame_data in player_tracks.values():
            if frame_data and -1 not in frame_data:
                all_player_ids.update(frame_data.keys())

        # Calculate speeds for each player
        for player_id in all_player_ids:
            # Get all frames where this player appears
            player_frames = []
            for frame_idx, frame_data in sorted(player_tracks.items()):
                if frame_data and player_id in frame_data and -1 not in frame_data:
                    player_frames.append((frame_idx, frame_data[player_id]))

            if len(player_frames) < 2:
                continue  # Need at least 2 frames to calculate speed

            # Calculate field coordinates for all frames
            frame_speeds = {}
            field_coordinates = {}
            distances = []

            prev_field_coord = None
            prev_frame_idx = None

            # Track max speed sample for this player (for debug)
            max_speed_sample = None

            for i, (frame_idx, bbox) in enumerate(player_frames):
                # Get foot position (bottom center) for players - more accurate for speed calculation
                foot_pos = self._get_foot_position(bbox)
                foot_pos_array = np.array(foot_pos, dtype=np.float32)

                # Get camera motion for this frame if available
                camera_motion = None
                if camera_movement and frame_idx < len(camera_movement):
                    camera_motion = tuple(camera_movement[frame_idx])

                # Convert to field coordinates using adaptive per-frame transformation
                field_coord, transform_method, confidence = adaptive_mapper.transform_point(
                    frame_idx, foot_pos_array, camera_motion
                )

                # Should never be None with adaptive transform, but check anyway
                if field_coord is None:
                    print(f"[WARNING] Unexpected None coordinate for player {player_id} frame {frame_idx}")
                    continue

                # DEBUG: Log out-of-bounds coordinates
                if (field_coord[0] < 0 or field_coord[0] > self.field_width or
                    field_coord[1] < 0 or field_coord[1] > self.field_height):
                    print(f"[COORD DEBUG] Player {player_id} Frame {frame_idx}: "
                          f"OUT OF BOUNDS ({field_coord[0]:.2f}, {field_coord[1]:.2f})m "
                          f"from pixel ({foot_pos[0]:.1f}, {foot_pos[1]:.1f}) "
                          f"via {transform_method} (conf: {confidence:.2f})")

                field_coordinates[int(frame_idx)] = [float(field_coord[0]), float(field_coord[1])]

                # Calculate speed if we have a previous position
                if prev_field_coord is not None:
                    # Calculate distance in meters
                    distance = self._calculate_distance(field_coord, prev_field_coord)

                    # Calculate time difference in seconds
                    frame_diff = frame_idx - prev_frame_idx
                    time_diff = frame_diff / self.fps

                    # Calculate speed in meters per second, then convert to km/h
                    if time_diff > 0:
                        instant_speed = distance / time_diff
                        speed_kmh = instant_speed * 3.6  # Convert m/s to km/h

                        # Sanity check: If speed is unreasonably high, it's likely a tracking/transformation error
                        # Human max sprint is ~12 m/s (43 km/h). Allow up to 15 m/s (54 km/h) with margin
                        if instant_speed > 15:  # 15 m/s = 54 km/h
                            # DEBUG: Track filtered speed
                            if max_speed_sample is None or speed_kmh > max_speed_sample['speed_kmh']:
                                max_speed_sample = {
                                    'speed_kmh': speed_kmh,
                                    'distance_m': distance,
                                    'frame_diff': frame_diff,
                                    'time_diff': time_diff,
                                    'from_frame': prev_frame_idx,
                                    'to_frame': frame_idx,
                                    'from_pos': prev_field_coord.copy(),
                                    'to_pos': field_coord.copy(),
                                    'transform_method': transform_method,
                                    'filtered': True
                                }
                            # Skip this measurement, likely an error
                            continue

                        distances.append(distance)

                        # Store actual speed (sanity check already filtered impossible speeds > 54 km/h)
                        frame_speeds[int(frame_idx)] = round(speed_kmh, 2)

                        # Track max speed sample for debug
                        if max_speed_sample is None or speed_kmh > max_speed_sample.get('speed_kmh', 0):
                            max_speed_sample = {
                                'speed_kmh': speed_kmh,
                                'distance_m': distance,
                                'frame_diff': frame_diff,
                                'time_diff': time_diff,
                                'from_frame': prev_frame_idx,
                                'to_frame': frame_idx,
                                'from_pos': prev_field_coord.copy(),
                                'to_pos': field_coord.copy(),
                                'transform_method': transform_method,
                                'filtered': False
                            }

                prev_field_coord = field_coord
                prev_frame_idx = frame_idx

            # Calculate aggregate statistics
            if frame_speeds:
                speeds_list = list(frame_speeds.values())
                player_speeds[int(player_id)] = {
                    'frames': frame_speeds,
                    'average_speed_kmh': round(np.mean(speeds_list), 2),
                    'max_speed_kmh': round(np.max(speeds_list), 2),
                    'total_distance_m': round(sum(distances), 2),
                    'field_coordinates': field_coordinates,
                    '_debug_max_speed_sample': max_speed_sample  # Store for debug
                }

        # Print debug statistics from AdaptiveHomographyMapper
        if hasattr(adaptive_mapper, 'print_debug_stats'):
            adaptive_mapper.print_debug_stats()

        # Find top 5 fastest players
        if player_speeds:
            sorted_players = sorted(player_speeds.items(),
                                   key=lambda x: x[1]['max_speed_kmh'],
                                   reverse=True)[:5]

            print(f"\n[SPEED DEBUG] Top 5 Fastest Players:")
            for rank, (pid, data) in enumerate(sorted_players, 1):
                max_sample = data.get('_debug_max_speed_sample')
                if max_sample:
                    print(f"   {rank}. Player {pid}: {data['max_speed_kmh']:.2f} km/h (avg: {data['average_speed_kmh']:.2f} km/h)")
                    if 'filtered' in max_sample:
                        print(f"      Max speed: {max_sample['speed_kmh']:.2f} km/h "
                              f"({'FILTERED' if max_sample['filtered'] else 'accepted'})")
                    print(f"      Distance: {max_sample['distance_m']:.2f}m over {max_sample['frame_diff']} frames "
                          f"({max_sample['time_diff']:.3f}s)")
                    print(f"      From frame {max_sample['from_frame']} to {max_sample['to_frame']}")
                    print(f"      Position: ({max_sample['from_pos'][0]:.2f}, {max_sample['from_pos'][1]:.2f}) -> "
                          f"({max_sample['to_pos'][0]:.2f}, {max_sample['to_pos'][1]:.2f}) meters")
                    print(f"      Transform method: {max_sample['transform_method']}")


        return player_speeds
