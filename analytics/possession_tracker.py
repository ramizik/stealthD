"""
Ball Possession Tracker module for detecting and tracking ball possession events.

Tracks when players have possession of the ball based on proximity (2.0m threshold).
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, List, Tuple, Optional


class PossessionTracker:
    """
    Track ball possession events and calculate possession statistics.

    Detects possession based on distance threshold and tracks possession
    transitions between players.
    """

    def __init__(self, possession_threshold: float = 1.5, min_possession_frames: int = 3):
        """
        Initialize the possession tracker with tighter parameters for improved detection.

        Args:
            possession_threshold: Distance threshold in meters for possession detection (default: 1.5m)
            min_possession_frames: Minimum consecutive frames to count as valid possession (default: 3)
        """
        self.possession_threshold = possession_threshold
        self.min_possession_frames = min_possession_frames

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points in meters."""
        if point1 is None or point2 is None:
            return float('inf')

        # Ensure we only use x, y coordinates (first 2 elements)
        point1 = np.asarray(point1)[:2]
        point2 = np.asarray(point2)[:2]

        return float(np.linalg.norm(point1 - point2))

    def detect_possession(self, player_positions: Dict[int, np.ndarray],
                         ball_position: Optional[np.ndarray]) -> Optional[int]:
        """
        Detect which player has possession at a given frame.

        Args:
            player_positions: Dictionary {player_id: np.array([x, y])} in field coordinates
            ball_position: Ball position as np.array([x, y]) in field coordinates, or None

        Returns:
            player_id of possessing player, or None if no possession
        """
        if ball_position is None or len(player_positions) == 0:
            return None

        # Calculate distances from ball to all players
        min_distance = float('inf')
        possessing_player = None

        for player_id, player_pos in player_positions.items():
            if player_pos is not None:
                distance = self._calculate_distance(player_pos, ball_position)
                if distance < min_distance and distance <= self.possession_threshold:
                    min_distance = distance
                    possessing_player = player_id

        return possessing_player

    def track_possessions(self, player_tracks: Dict, ball_tracks: Dict,
                         field_coords_players: Dict, field_coords_ball: Dict,
                         fps: float = 30) -> Tuple[List[Dict], Dict]:
        """
        Track all possession events throughout the match.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: bbox}}
            ball_tracks: Dictionary {frame_idx: bbox}
            field_coords_players: Dictionary from SpeedCalculator with field coordinates
                                 {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
            field_coords_ball: Dictionary {frame_idx: [x, y]} ball field coordinates
            fps: Frames per second of video

        Returns:
            Tuple of:
            - List of possession events with metadata
            - Dictionary {player_id: total_possession_time_seconds}
        """
        possession_events = []
        player_possession_time = {}

        # Get sorted frame indices
        frame_indices = sorted(player_tracks.keys())

        current_possessing_player = None
        possession_start_frame = None
        possession_frames = []

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

            # Detect possession at this frame
            possessing_player = self.detect_possession(player_positions, ball_position)

            # Track possession transitions
            if possessing_player != current_possessing_player:
                # Possession changed or ended
                if current_possessing_player is not None and len(possession_frames) >= self.min_possession_frames:
                    # Record the previous possession event
                    duration_seconds = len(possession_frames) / fps

                    # Get team assignment if available
                    team = None
                    if player_tracks.get(possession_start_frame):
                        frame_data = player_tracks[possession_start_frame]
                        if current_possessing_player in frame_data:
                            team = 0  # Default team, will be updated with actual team data

                    event = {
                        'player_id': int(current_possessing_player),
                        'team': team,
                        'start_frame': int(possession_start_frame),
                        'end_frame': int(possession_frames[-1]),
                        'duration_seconds': round(duration_seconds, 2),
                        'num_frames': len(possession_frames)
                    }
                    possession_events.append(event)

                    # Update total possession time for player
                    if current_possessing_player not in player_possession_time:
                        player_possession_time[int(current_possessing_player)] = 0.0
                    player_possession_time[int(current_possessing_player)] += duration_seconds

                # Start new possession
                current_possessing_player = possessing_player
                possession_start_frame = frame_idx
                possession_frames = [frame_idx] if possessing_player is not None else []
            else:
                # Continuing possession
                if possessing_player is not None:
                    possession_frames.append(frame_idx)

        # Handle final possession if video ends during possession
        if current_possessing_player is not None and len(possession_frames) >= self.min_possession_frames:
            duration_seconds = len(possession_frames) / fps
            event = {
                'player_id': int(current_possessing_player),
                'team': None,
                'start_frame': int(possession_start_frame),
                'end_frame': int(possession_frames[-1]),
                'duration_seconds': round(duration_seconds, 2),
                'num_frames': len(possession_frames)
            }
            possession_events.append(event)

            if current_possessing_player not in player_possession_time:
                player_possession_time[int(current_possessing_player)] = 0.0
            player_possession_time[int(current_possessing_player)] += duration_seconds

        # Round all possession times
        for player_id in player_possession_time:
            player_possession_time[player_id] = round(player_possession_time[player_id], 2)

        return possession_events, player_possession_time

    def update_possession_with_teams(self, possession_events: List[Dict],
                                     team_assignments: Dict) -> List[Dict]:
        """
        Update possession events with team assignments.

        Args:
            possession_events: List of possession events
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}

        Returns:
            Updated possession events with team information
        """
        for event in possession_events:
            player_id = event['player_id']
            start_frame = event['start_frame']

            # Look for team assignment around start frame
            if start_frame in team_assignments:
                if player_id in team_assignments[start_frame]:
                    event['team'] = int(team_assignments[start_frame][player_id])

        return possession_events
