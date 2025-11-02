"""
Pass Detector module for identifying and validating pass events.

Detects passes based on possession transfers with distance and duration validation.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, List, Optional


class PassDetector:
    """
    Detect and validate pass events in soccer matches.

    Identifies passes based on possession transfers between same-team players,
    with validation for minimum distance and possession duration.
    """

    def __init__(self, min_pass_distance: float = 3.0, min_possession_duration: float = 0.5):
        """
        Initialize the pass detector.

        Args:
            min_pass_distance: Minimum distance in meters between players to count as pass (default: 3.0m)
            min_possession_duration: Minimum duration in seconds for valid possession (default: 0.5s)
        """
        self.min_pass_distance = min_pass_distance
        self.min_possession_duration = min_possession_duration

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points in meters."""
        if point1 is None or point2 is None:
            return 0.0
        return float(np.linalg.norm(point1 - point2))

    def _get_player_position(self, player_id: int, frame_idx: int,
                            field_coords_players: Dict) -> Optional[np.ndarray]:
        """Get player position at a specific frame."""
        if player_id in field_coords_players:
            player_data = field_coords_players[player_id]
            if 'field_coordinates' in player_data:
                coords = player_data['field_coordinates'].get(frame_idx, None)
                if coords is not None:
                    return np.array(coords)
        return None

    def detect_passes(self, possession_events: List[Dict], field_coords_players: Dict,
                     team_assignments: Dict, fps: float = 30) -> List[Dict]:
        """
        Detect pass events from possession transitions.

        Args:
            possession_events: List of possession events from PossessionTracker
            field_coords_players: Dictionary {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}
            fps: Frames per second of video

        Returns:
            List of pass events with metadata
        """
        passes = []

        # Sort possession events by start frame
        sorted_events = sorted(possession_events, key=lambda x: x['start_frame'])

        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]

            # Check if both possessions meet minimum duration requirement
            if (current_event['duration_seconds'] < self.min_possession_duration or
                next_event['duration_seconds'] < self.min_possession_duration):
                continue

            passer_id = current_event['player_id']
            receiver_id = next_event['player_id']

            # Skip if same player (not a pass)
            if passer_id == receiver_id:
                continue

            # Get team assignments for both players
            passer_frame = current_event['end_frame']
            receiver_frame = next_event['start_frame']

            passer_team = None
            receiver_team = None

            # Try to get team from possession event first
            if current_event.get('team') is not None:
                passer_team = current_event['team']
            elif passer_frame in team_assignments and passer_id in team_assignments[passer_frame]:
                passer_team = team_assignments[passer_frame][passer_id]

            if next_event.get('team') is not None:
                receiver_team = next_event['team']
            elif receiver_frame in team_assignments and receiver_id in team_assignments[receiver_frame]:
                receiver_team = team_assignments[receiver_frame][receiver_id]

            # Only count as pass if both players are on same team
            if passer_team is None or receiver_team is None or passer_team != receiver_team:
                continue

            # Get positions of passer and receiver
            passer_pos = self._get_player_position(passer_id, passer_frame, field_coords_players)
            receiver_pos = self._get_player_position(receiver_id, receiver_frame, field_coords_players)

            if passer_pos is None or receiver_pos is None:
                continue

            # Calculate distance between players
            pass_distance = self._calculate_distance(passer_pos, receiver_pos)

            # Validate minimum pass distance
            if pass_distance < self.min_pass_distance:
                continue

            # Calculate pass timestamp
            pass_timestamp = passer_frame / fps

            # Create pass event
            pass_event = {
                'frame': int(passer_frame),
                'timestamp': round(pass_timestamp, 2),
                'from_player': int(passer_id),
                'to_player': int(receiver_id),
                'team': int(passer_team),
                'distance_m': round(pass_distance, 2),
                'start_position': [round(float(passer_pos[0]), 2), round(float(passer_pos[1]), 2)],
                'end_position': [round(float(receiver_pos[0]), 2), round(float(receiver_pos[1]), 2)]
            }

            passes.append(pass_event)

        return passes

    def count_player_passes(self, passes: List[Dict]) -> Dict[int, Dict[str, int]]:
        """
        Count passes completed and received by each player.

        Args:
            passes: List of pass events

        Returns:
            Dictionary {player_id: {'passes_completed': int, 'passes_received': int}}
        """
        player_pass_counts = {}

        for pass_event in passes:
            from_player = pass_event['from_player']
            to_player = pass_event['to_player']

            # Initialize if needed
            if from_player not in player_pass_counts:
                player_pass_counts[from_player] = {'passes_completed': 0, 'passes_received': 0}
            if to_player not in player_pass_counts:
                player_pass_counts[to_player] = {'passes_completed': 0, 'passes_received': 0}

            # Increment counts
            player_pass_counts[from_player]['passes_completed'] += 1
            player_pass_counts[to_player]['passes_received'] += 1

        return player_pass_counts
