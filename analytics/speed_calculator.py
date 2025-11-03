"""
Speed Calculator module for soccer player speed and distance metrics.

Calculates player speeds using field coordinates (105m x 68m) with homography
transformation fallback to linear scaling.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, Tuple, Optional


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
                                    frame_idx: int = None, debug_stats: dict = None) -> Tuple[Optional[np.ndarray], str]:
        """
        Convert pixel coordinates to field coordinates using global homography mapper.

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates
            global_mapper: GlobalHomographyMapper object
            video_dimensions: (width, height) of video in pixels
            frame_idx: Frame index for logging (optional)
            debug_stats: Dictionary to track transformation method usage (optional)

        Returns:
            Tuple of (Array of shape (N, 2) with field coordinates in meters or None if low confidence,
                     transformation_method string)
        """
        if len(pixel_coords) == 0:
            return np.array([]).reshape(0, 2), 'empty'

        # Check if global mapper is built
        if not hasattr(global_mapper, 'matrix_built') or not global_mapper.matrix_built:
            # Fallback to conservative linear scaling if no global homography
            if debug_stats is not None:
                debug_stats['linear_fallback'] += 1
            return self._conservative_linear_scale(pixel_coords, video_dimensions), 'linear_fallback'

        try:
            # Use global homography mapper
            field_coords_raw, confidence = global_mapper.transform_points(pixel_coords)

            # Convert from pitch coordinate system to meters
            # SoccerPitchConfiguration uses 12000x7000 units for 105x68m pitch
            # Scale: X axis: 12000 units = 105m, Y axis: 7000 units = 68m
            field_coords = field_coords_raw.copy()
            field_coords[:, 0] = field_coords[:, 0] * (self.field_width / 12000.0)
            field_coords[:, 1] = field_coords[:, 1] * (self.field_height / 7000.0)

            # Check if coordinates are in reasonable bounds
            if self._is_catastrophically_bad(field_coords):
                if debug_stats is not None:
                    debug_stats['catastrophic_failures'] += 1
                # Return None to skip this position
                return None, 'catastrophic_failure'

            # Check if position has sufficient keypoint coverage
            if len(field_coords) > 0:
                x_meters, y_meters = field_coords[0]
                if not global_mapper.is_position_reliable(x_meters, y_meters, min_confidence=0.7):
                    if debug_stats is not None:
                        debug_stats['low_confidence'] = debug_stats.get('low_confidence', 0) + 1
                    # Return None to skip this position
                    return None, 'low_confidence'

            if debug_stats is not None:
                debug_stats['global_homography_success'] = debug_stats.get('global_homography_success', 0) + 1

            return field_coords, 'global_homography'

        except Exception as e:
            if debug_stats is not None:
                debug_stats['homography_failed'] += 1
            # Fallback to conservative linear scaling
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

    def calculate_speeds(self, player_tracks: Dict, global_mapper,
                        video_dimensions: Tuple[int, int]) -> Dict:
        """
        Calculate speeds for all players across all frames using global homography.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: [x1, y1, x2, y2]}}
            global_mapper: GlobalHomographyMapper object
            video_dimensions: (width, height) of video in pixels

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

        # DEBUG: Track transformation method usage
        debug_stats = {
            'global_homography_success': 0,
            'homography_failed': 0,
            'linear_fallback': 0,
            'catastrophic_failures': 0,
            'low_confidence': 0,
            'positions_skipped': 0
        }

        print(f"\n[SPEED DEBUG] Starting speed calculation with global homography:")
        print(f"   Video dimensions: {video_dimensions}")
        print(f"   Field dimensions: {self.field_width}m x {self.field_height}m")
        print(f"   FPS: {self.fps}")

        # Get global mapper statistics
        if hasattr(global_mapper, 'get_statistics'):
            mapper_stats = global_mapper.get_statistics()
            print(f"   Global mapper statistics:")
            print(f"     - Keypoints accumulated: {mapper_stats.get('total_keypoints_accumulated', 0)}")
            print(f"     - Unique keypoints: {mapper_stats.get('unique_keypoints_detected', 0)}")
            print(f"     - High-confidence coverage: {mapper_stats.get('high_confidence_coverage_percent', 0):.1f}%")

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
                foot_pos = np.array([self._get_foot_position(bbox)])

                # Convert to field coordinates using global homography
                field_coord, transform_method = self.calculate_field_coordinates(
                    foot_pos, global_mapper, video_dimensions, frame_idx, debug_stats
                )

                # Skip if coordinate conversion failed or is low confidence
                if field_coord is None or len(field_coord) == 0:
                    debug_stats['positions_skipped'] += 1
                    continue

                field_coord = field_coord[0]  # Get single point
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

        # DEBUG: Print summary statistics
        print(f"\n[SPEED DEBUG] Transformation Statistics:")
        print(f"   Global homography success: {debug_stats.get('global_homography_success', 0)}")
        print(f"   Homography failed: {debug_stats.get('homography_failed', 0)}")
        print(f"   Catastrophic failures: {debug_stats.get('catastrophic_failures', 0)}")
        print(f"   Low confidence positions: {debug_stats.get('low_confidence', 0)}")
        print(f"   Linear fallback used: {debug_stats.get('linear_fallback', 0)}")
        print(f"   Positions skipped: {debug_stats.get('positions_skipped', 0)}")

        # Calculate success rate
        total_attempts = sum(debug_stats.values())
        success_count = debug_stats.get('global_homography_success', 0)
        if total_attempts > 0:
            success_rate = 100.0 * success_count / total_attempts
            print(f"   Success rate: {success_rate:.1f}%")

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
                    print(f"      Max speed: {max_sample['speed_kmh']:.2f} km/h "
                          f"({'FILTERED' if max_sample['filtered'] else 'accepted'})")
                    print(f"      Distance: {max_sample['distance_m']:.2f}m over {max_sample['frame_diff']} frames "
                          f"({max_sample['time_diff']:.3f}s)")
                    print(f"      From frame {max_sample['from_frame']} to {max_sample['to_frame']}")
                    print(f"      Position: ({max_sample['from_pos'][0]:.2f}, {max_sample['from_pos'][1]:.2f}) -> "
                          f"({max_sample['to_pos'][0]:.2f}, {max_sample['to_pos'][1]:.2f}) meters")
                    print(f"      Transform method: {max_sample['transform_method']}")

        return player_speeds
