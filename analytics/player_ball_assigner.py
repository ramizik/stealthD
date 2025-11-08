"""
Player Ball Assigner module for frame-by-frame ball-to-player assignment.

Uses field coordinates for accurate distance calculation and assigns the ball
to the closest player within threshold distance on every frame.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, Optional, Tuple


class PlayerBallAssigner:
    """
    Assign ball to closest player on a frame-by-frame basis.

    Uses field coordinates for accurate distance measurement across different
    camera angles and zoom levels. More granular than possession events for
    accurate touch counting.
    """

    def __init__(self, max_ball_distance: float = 2.5):
        """
        Initialize the player ball assigner.

        Args:
            max_ball_distance: Maximum distance in meters for ball assignment (default: 2.5m)
        """
        self.max_ball_distance = max_ball_distance

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points in meters."""
        if point1 is None or point2 is None:
            return float('inf')

        # Ensure we only use x, y coordinates (first 2 elements)
        point1 = np.asarray(point1)[:2]
        point2 = np.asarray(point2)[:2]

        return float(np.linalg.norm(point1 - point2))

    def assign_ball_to_player(self, player_positions: Dict[int, np.ndarray],
                              ball_position: Optional[np.ndarray]) -> Optional[int]:
        """
        Assign ball to closest player within threshold distance.

        Args:
            player_positions: Dictionary {player_id: np.array([x, y])} in field coordinates
            ball_position: Ball position as np.array([x, y]) in field coordinates, or None

        Returns:
            player_id of player assigned to ball, or None if no player within threshold
        """
        if ball_position is None or len(player_positions) == 0:
            return None

        min_distance = float('inf')
        assigned_player = None

        for player_id, player_pos in player_positions.items():
            if player_pos is not None:
                distance = self._calculate_distance(player_pos, ball_position)

                if distance < self.max_ball_distance:
                    if distance < min_distance:
                        min_distance = distance
                        assigned_player = player_id

        return assigned_player

    def assign_ball_for_all_frames(self, player_tracks: Dict, ball_tracks: Dict,
                                   field_coords_players: Dict, field_coords_ball: Dict) -> Dict:
        """
        Assign ball to players for all frames in the video.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: bbox}}
            ball_tracks: Dictionary {frame_idx: bbox}
            field_coords_players: Dictionary from SpeedCalculator with field coordinates
                                 {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
            field_coords_ball: Dictionary {frame_idx: [x, y]} ball field coordinates

        Returns:
            Dictionary {frame_idx: player_id} mapping frames to assigned players
        """
        ball_assignments = {}

        # Get sorted frame indices
        frame_indices = sorted(player_tracks.keys())

        for frame_idx in frame_indices:
            # Get player positions at this frame
            player_positions = {}
            for player_id, player_data in field_coords_players.items():
                if 'field_coordinates' in player_data:
                    field_coords = player_data['field_coordinates'].get(frame_idx, None)
                    if field_coords is not None:
                        player_positions[player_id] = np.array(field_coords)

            # Get ball position at this frame
            ball_position = None
            if frame_idx in field_coords_ball:
                ball_coords = field_coords_ball[frame_idx]
                if ball_coords is not None and len(ball_coords) == 2:
                    ball_position = np.array(ball_coords)

            # Assign ball to player
            assigned_player = self.assign_ball_to_player(player_positions, ball_position)

            if assigned_player is not None:
                ball_assignments[int(frame_idx)] = int(assigned_player)

        return ball_assignments

    def count_player_touches(self, ball_assignments: Dict) -> Dict[int, int]:
        """
        Count number of touches for each player from frame-by-frame assignments.

        A touch is counted when:
        - Ball is assigned to a player for the first time
        - Ball is reassigned to the same player after being with another player or unassigned

        Args:
            ball_assignments: Dictionary {frame_idx: player_id}

        Returns:
            Dictionary {player_id: touch_count}
        """
        touches = {}
        prev_player = None

        # Sort frames to process in order
        sorted_frames = sorted(ball_assignments.keys())

        for frame_idx in sorted_frames:
            current_player = ball_assignments[frame_idx]

            # Count a touch when ball is assigned to a different player
            if current_player != prev_player and current_player is not None:
                if current_player not in touches:
                    touches[current_player] = 0
                touches[current_player] += 1

            prev_player = current_player

        return touches

    def get_player_touch_frames(self, ball_assignments: Dict) -> Dict[int, list]:
        """
        Get list of frames where each player touched the ball.

        Args:
            ball_assignments: Dictionary {frame_idx: player_id}

        Returns:
            Dictionary {player_id: [frame_idx1, frame_idx2, ...]}
        """
        touch_frames = {}
        prev_player = None

        # Sort frames to process in order
        sorted_frames = sorted(ball_assignments.keys())

        for frame_idx in sorted_frames:
            current_player = ball_assignments[frame_idx]

            # Record touch frame when ball is assigned to a different player
            if current_player != prev_player and current_player is not None:
                if current_player not in touch_frames:
                    touch_frames[current_player] = []
                touch_frames[current_player].append(int(frame_idx))

            prev_player = current_player

        return touch_frames
