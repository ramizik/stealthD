"""
Simple Player Ball Assigner using bbox-based distance calculation.

This is a simpler alternative to the field-coordinate-based assigner that uses
bounding box positions (left/right foot) for distance calculation.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict

import numpy as np

from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """
    Simple ball-to-player assigner using bbox-based distance calculation.

    Assigns the ball to the closest player based on distance between ball position
    and player's left/right foot positions (from bounding box).
    """

    def __init__(self, max_player_ball_distance=70):
        """
        Initialize the PlayerBallAssigner.

        Args:
            max_player_ball_distance: Maximum distance in pixels for ball assignment (default: 70)
        """
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assign ball to the closest player within threshold distance.

        Args:
            players: Dictionary {player_id: {'bbox': [x1, y1, x2, y2], ...}}
            ball_bbox: Ball bounding box as [x1, y1, x2, y2]

        Returns:
            player_id of assigned player, or -1 if no player within threshold
        """
        if ball_bbox is None or len(ball_bbox) < 4:
            return -1

        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            if 'bbox' not in player:
                continue

            player_bbox = player['bbox']
            if player_bbox is None or len(player_bbox) < 4:
                continue

            # Calculate distance to left and right foot positions
            # Left foot: (x1, y2) - bottom left corner
            # Right foot: (x2, y2) - bottom right corner
            distance_left = measure_distance(
                np.array([player_bbox[0], player_bbox[3]]),
                np.array(ball_position)
            )
            distance_right = measure_distance(
                np.array([player_bbox[2], player_bbox[3]]),
                np.array(ball_position)
            )

            # Use minimum distance (closest foot)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def assign_ball_for_all_frames(self, player_tracks: Dict, ball_tracks: Dict,
                                   field_coords_players: Dict = None,
                                   field_coords_ball: Dict = None) -> Dict[int, tuple]:
        """
        Assign ball to players for all frames using bbox-based distance.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: bbox}} or {frame_idx: {player_id: {'bbox': bbox}}}
            ball_tracks: Dictionary {frame_idx: bbox} or {frame_idx: [x1, y1, x2, y2]}
            field_coords_players: Not used in simple assigner (kept for compatibility)
            field_coords_ball: Not used in simple assigner (kept for compatibility)

        Returns:
            Dictionary {frame_idx: (player_id, distance)} where distance is in pixels
        """
        ball_assignments = {}

        frame_indices = sorted(set(player_tracks.keys()) & set(ball_tracks.keys()))

        for frame_idx in frame_indices:
            # Get players dict for this frame
            if frame_idx not in player_tracks:
                continue

            frame_players = player_tracks[frame_idx]
            players_dict = {}

            for player_id, player_data in frame_players.items():
                # Handle both formats: direct bbox or dict with bbox
                if isinstance(player_data, dict) and 'bbox' in player_data:
                    bbox = player_data['bbox']
                elif isinstance(player_data, list) and len(player_data) >= 4:
                    bbox = player_data
                else:
                    continue

                if bbox is None or any(b is None for b in bbox):
                    continue

                players_dict[player_id] = {'bbox': bbox}

            # Get ball bbox for this frame
            ball_track_data = ball_tracks.get(frame_idx)
            if ball_track_data is None:
                continue

            # Handle both formats: list (bbox) or dict (enhanced format)
            if isinstance(ball_track_data, dict):
                # Enhanced format: {0: {'bbox': [x1, y1, x2, y2], ...}}
                if 0 in ball_track_data and isinstance(ball_track_data[0], dict):
                    ball_bbox = ball_track_data[0].get('bbox', None)
                else:
                    continue
            elif isinstance(ball_track_data, list) and len(ball_track_data) >= 4:
                ball_bbox = ball_track_data
            else:
                continue

            if ball_bbox is None or any(b is None for b in ball_bbox):
                continue

            # Assign ball to player
            assigned_player = self.assign_ball_to_player(players_dict, ball_bbox)

            if assigned_player != -1:
                # Calculate distance for compatibility
                ball_center = get_center_of_bbox(ball_bbox)
                player_bbox = players_dict[assigned_player]['bbox']
                player_left_foot = np.array([player_bbox[0], player_bbox[3]])
                player_right_foot = np.array([player_bbox[2], player_bbox[3]])
                distance_left = measure_distance(player_left_foot, np.array(ball_center))
                distance_right = measure_distance(player_right_foot, np.array(ball_center))
                distance = min(distance_left, distance_right)

                ball_assignments[int(frame_idx)] = (int(assigned_player), float(distance))

        return ball_assignments

    def get_simple_assignments(self, ball_assignments_with_distance: Dict[int, tuple]) -> Dict[int, int]:
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

        Args:
            ball_assignments: Dictionary {frame_idx: player_id}

        Returns:
            Dictionary {player_id: touch_count}
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

    def get_player_touch_frames(self, ball_assignments: Dict[int, int]) -> Dict[int, list]:
        """
        Get list of frames where each player touched the ball.

        Args:
            ball_assignments: Dictionary {frame_idx: player_id}

        Returns:
            Dictionary {player_id: [frame_idx1, frame_idx2, ...]}
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

    def analyze_assignment_quality(self, ball_assignments: Dict[int, tuple]) -> Dict:
        """
        Analyze the quality of ball assignments.

        Args:
            ball_assignments: Dictionary {frame_idx: (player_id, distance)}

        Returns:
            Dictionary with quality metrics
        """
        if not ball_assignments:
            return {
                'total_assignments': 0,
                'avg_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0
            }

        distances = [distance for _, (_, distance) in ball_assignments.items()]

        return {
            'total_assignments': len(ball_assignments),
            'avg_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances))
        }
