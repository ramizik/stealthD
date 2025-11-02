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

    def calculate_field_coordinates(self, pixel_coords: np.ndarray, view_transformer,
                                    video_dimensions: Tuple[int, int],
                                    frame_idx: int = None) -> np.ndarray:
        """
        Convert pixel coordinates to field coordinates.
        Try homography first, use cached transformer if current is None,
        fallback to conservative linear scaling as last resort.

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates
            view_transformer: ViewTransformer object for homography (can be None)
            video_dimensions: (width, height) of video in pixels
            frame_idx: Frame index for caching (optional)

        Returns:
            Array of shape (N, 2) with field coordinates in meters
        """
        if len(pixel_coords) == 0:
            return np.array([]).reshape(0, 2)

        # Try current frame's transformer first
        if view_transformer is not None:
            try:
                field_coords = view_transformer.transform_points(pixel_coords)
                # Convert from pitch coordinate system to meters
                # SoccerPitchConfiguration uses 12000x7000 units for 105x68m pitch
                # Scale: X axis: 12000 units = 105m, Y axis: 7000 units = 68m
                field_coords[:, 0] = field_coords[:, 0] * (self.field_width / 12000.0)
                field_coords[:, 1] = field_coords[:, 1] * (self.field_height / 7000.0)

                # Cache this valid transformer
                self.last_valid_transformer = view_transformer
                self.last_transformer_frame = frame_idx

                return field_coords
            except Exception as e:
                pass

        # Try cached transformer if available (from a nearby frame)
        if self.last_valid_transformer is not None:
            try:
                # Use cached transformer (camera movement is already compensated in tracks)
                field_coords = self.last_valid_transformer.transform_points(pixel_coords)
                field_coords[:, 0] = field_coords[:, 0] * (self.field_width / 12000.0)
                field_coords[:, 1] = field_coords[:, 1] * (self.field_height / 7000.0)
                return field_coords
            except Exception as e:
                pass

        # Last resort: Conservative linear scaling
        # Assume only central 50% of frame shows the field (very conservative)
        return self._conservative_linear_scale(pixel_coords, video_dimensions)

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

    def calculate_speeds(self, player_tracks: Dict, view_transformers: Dict,
                        video_dimensions: Tuple[int, int]) -> Dict:
        """
        Calculate speeds for all players across all frames.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: [x1, y1, x2, y2]}}
            view_transformers: Dictionary {frame_idx: ViewTransformer or None}
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

        # Reset transformer cache for new video
        self.last_valid_transformer = None
        self.last_transformer_frame = None

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

            for i, (frame_idx, bbox) in enumerate(player_frames):
                # Get foot position (bottom center) for players - more accurate for speed calculation
                foot_pos = np.array([self._get_foot_position(bbox)])

                # Get view transformer for this frame
                view_transformer = view_transformers.get(frame_idx, None)

                # Convert to field coordinates (with caching and fallback)
                field_coord = self.calculate_field_coordinates(
                    foot_pos, view_transformer, video_dimensions, frame_idx
                )

                if len(field_coord) == 0:
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

                        # Sanity check: If speed is unreasonably high, it's likely a tracking/transformation error
                        # Human max sprint is ~12 m/s (43 km/h). Allow up to 15 m/s (54 km/h) with margin
                        if instant_speed > 15:  # 15 m/s = 54 km/h
                            # Skip this measurement, likely an error
                            continue

                        distances.append(distance)
                        speed_kmh = instant_speed * 3.6  # Convert m/s to km/h

                        # Store actual speed (sanity check already filtered impossible speeds > 54 km/h)
                        frame_speeds[int(frame_idx)] = round(speed_kmh, 2)

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
                    'field_coordinates': field_coordinates
                }

        return player_speeds
