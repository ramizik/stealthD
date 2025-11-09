"""
Improved Player Ball Assigner with consistent coordinate normalization.

Key improvements:
1. Unified coordinate normalization
2. Increased detection radius for better assignment
3. Confidence scoring for assignments
4. Better handling of edge cases
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, Optional, Tuple, List


class ImprovedPlayerBallAssigner:
    """
    Improved ball assignment with consistent coordinate handling.
    """

    def __init__(self, max_ball_distance: float = 10.0):
        """
        Initialize the player ball assigner.

        Args:
            max_ball_distance: Maximum distance in meters for ball assignment (default: 10.0m)
                             Increased from 8.0m to catch more assignments
        """
        self.max_ball_distance = max_ball_distance

    def _normalize_to_meters(self, coord: np.ndarray) -> np.ndarray:
        """
        Unified coordinate normalization to meters.

        Handles:
        - Meters (0-120, 0-80): Return as-is
        - Pitch coordinates (12000x7000): Convert to meters
        - Scaled coordinates: Detect and convert
        """
        coord = np.asarray(coord, dtype=float)[:2]

        # Already in meters (standard pitch: 105m x 68m)
        if 0 <= coord[0] <= 130 and 0 <= coord[1] <= 90:
            return coord

        # Pitch coordinates (12000x7000) -> meters (105x68)
        if coord[0] > 1000 or coord[1] > 1000:
            return np.array([
                coord[0] * (105.0 / 12000.0),
                coord[1] * (68.0 / 7000.0)
            ])

        # Scaled coordinates (detect scale factor)
        if coord[0] > 200 or coord[1] > 200:
            # Estimate scale based on typical field dimensions
            scale_x = 105.0 / coord[0] if coord[0] > 0 else 0.1
            scale_y = 68.0 / coord[1] if coord[1] > 0 else 0.1
            scale = (scale_x + scale_y) / 2
            return coord * scale

        # Moderate scaling (50-200 range)
        if coord[0] > 50 or coord[1] > 50:
            return coord * 0.5  # Try 50% scaling

        return coord

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance with coordinate normalization."""
        if point1 is None or point2 is None:
            return float('inf')

        p1 = self._normalize_to_meters(np.asarray(point1)[:2])
        p2 = self._normalize_to_meters(np.asarray(point2)[:2])

        distance = np.linalg.norm(p1 - p2)
        return float(distance)

    def assign_ball_to_player(self, player_positions: Dict[int, np.ndarray],
                              ball_position: Optional[np.ndarray]) -> Optional[Tuple[int, float]]:
        """
        Assign ball to closest player within threshold distance.

        Args:
            player_positions: Dictionary {player_id: np.array([x, y])}
            ball_position: Ball position as np.array([x, y]), or None

        Returns:
            Tuple of (player_id, distance) or None if no player within threshold
        """
        if ball_position is None or len(player_positions) == 0:
            return None

        min_distance = float('inf')
        assigned_player = None

        for player_id, player_pos in player_positions.items():
            if player_pos is None:
                continue

            distance = self._calculate_distance(player_pos, ball_position)

            if distance < self.max_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        if assigned_player is not None:
            return (assigned_player, min_distance)

        return None

    def assign_ball_for_all_frames(self, player_tracks: Dict, ball_tracks: Dict,
                                   field_coords_players: Dict,
                                   field_coords_ball: Dict) -> Dict[int, Tuple[int, float]]:
        """
        Assign ball to players for all frames with confidence scores.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: bbox}}
            ball_tracks: Dictionary {frame_idx: bbox}
            field_coords_players: Dictionary {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
            field_coords_ball: Dictionary {frame_idx: [x, y]}

        Returns:
            Dictionary {frame_idx: (player_id, distance)}
        """
        ball_assignments = {}

        frame_indices = sorted(set(player_tracks.keys()) & set(ball_tracks.keys()))

        for frame_idx in frame_indices:
            # Get player positions at this frame
            player_positions = {}
            for player_id, player_data in field_coords_players.items():
                if 'field_coordinates' in player_data:
                    field_coords = player_data['field_coordinates'].get(frame_idx)
                    if field_coords is not None:
                        player_positions[player_id] = np.array(field_coords)

            # Get ball position at this frame
            ball_position = None
            if frame_idx in field_coords_ball:
                ball_coords = field_coords_ball[frame_idx]
                if ball_coords is not None and len(ball_coords) == 2:
                    ball_position = np.array(ball_coords)

            # Assign ball to player
            result = self.assign_ball_to_player(player_positions, ball_position)

            if result is not None:
                player_id, distance = result
                ball_assignments[int(frame_idx)] = (int(player_id), float(distance))

        return ball_assignments

    def get_simple_assignments(self, ball_assignments_with_distance: Dict[int, Tuple[int, float]]) -> Dict[int, int]:
        """
        Convert assignments with distances to simple player ID mapping.

        Args:
            ball_assignments_with_distance: Dictionary {frame_idx: (player_id, distance)}

        Returns:
            Dictionary {frame_idx: player_id}
        """
        return {frame_idx: player_id for frame_idx, (player_id, _) in ball_assignments_with_distance.items()}

    def count_player_touches(self, ball_assignments: Dict[int, int]) -> Dict[int, int]:
        """
        Count touches from frame-by-frame assignments.

        A touch is counted when ball is assigned to a new player.
        """
        touches = {}
        prev_player = None

        sorted_frames = sorted(ball_assignments.keys())

        for frame_idx in sorted_frames:
            current_player = ball_assignments[frame_idx]

            if current_player != prev_player and current_player is not None:
                if current_player not in touches:
                    touches[current_player] = 0
                touches[current_player] += 1

            prev_player = current_player

        return touches

    def get_player_touch_frames(self, ball_assignments: Dict[int, int]) -> Dict[int, List[int]]:
        """
        Get list of frames where each player touched the ball.
        """
        touch_frames = {}
        prev_player = None

        sorted_frames = sorted(ball_assignments.keys())

        for frame_idx in sorted_frames:
            current_player = ball_assignments[frame_idx]

            if current_player != prev_player and current_player is not None:
                if current_player not in touch_frames:
                    touch_frames[current_player] = []
                touch_frames[current_player].append(int(frame_idx))

            prev_player = current_player

        return touch_frames

    def analyze_assignment_quality(self, ball_assignments: Dict[int, Tuple[int, float]]) -> Dict:
        """
        Analyze quality of ball assignments.

        Returns:
            Dictionary with assignment statistics
        """
        if not ball_assignments:
            return {
                'total_assignments': 0,
                'avg_distance': 0.0,
                'max_distance': 0.0,
                'min_distance': 0.0
            }

        distances = [dist for _, dist in ball_assignments.values()]

        return {
            'total_assignments': len(ball_assignments),
            'avg_distance': round(np.mean(distances), 2),
            'max_distance': round(np.max(distances), 2),
            'min_distance': round(np.min(distances), 2),
            'std_distance': round(np.std(distances), 2)
        }