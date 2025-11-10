"""
Team Assignment Stabilizer

Ensures consistent team assignments throughout the video by using
majority voting across all frames for each player ID.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict
from collections import Counter


class TeamStabilizer:
    """
    Stabilizes team assignments using majority voting.

    Prevents brief misclassifications by looking at all frames where
    a player appears and assigning them to their most common team.
    """

    def __init__(self, min_frames_threshold: int = 10):
        """
        Initialize team stabilizer.

        Args:
            min_frames_threshold: Minimum frames a player must appear in to be considered
                                 for stabilization (prevents noise from brief detections)
        """
        self.min_frames_threshold = min_frames_threshold
        self.stable_teams = {}  # {player_id: stable_team_id}

    def analyze_team_assignments(self, player_classids: Dict) -> Dict[int, int]:
        """
        Analyze all team assignments and determine stable team for each player.

        Args:
            player_classids: Dictionary {frame_idx: {player_id: team_id}}

        Returns:
            Dictionary {player_id: stable_team_id}
        """
        # Collect all team assignments per player
        player_teams = {}  # {player_id: [team_id, team_id, ...]}

        for frame_idx, frame_assignments in player_classids.items():
            if frame_assignments is None:
                continue

            for player_id, team_id in frame_assignments.items():
                if player_id not in player_teams:
                    player_teams[player_id] = []
                player_teams[player_id].append(team_id)

        # Determine stable team for each player using majority voting
        stable_teams = {}

        for player_id, team_assignments in player_teams.items():
            # Only stabilize if player appears in enough frames
            if len(team_assignments) < self.min_frames_threshold:
                # For players with few appearances, use their most common team anyway
                pass

            # Count team occurrences
            team_counter = Counter(team_assignments)

            # Assign most common team (majority vote)
            stable_team = team_counter.most_common(1)[0][0]
            stable_teams[player_id] = stable_team

            # Log if there was significant disagreement (indicates correction)
            if len(team_counter) > 1:
                total_assignments = len(team_assignments)
                majority_count = team_counter[stable_team]
                agreement_percentage = (majority_count / total_assignments) * 100

                if agreement_percentage < 90:
                    print(f"  Player #{player_id}: Stabilized to team {stable_team} "
                          f"({agreement_percentage:.1f}% agreement, corrected {total_assignments - majority_count} frames)")

        self.stable_teams = stable_teams
        return stable_teams

    def apply_stable_teams(self, player_classids: Dict) -> Dict:
        """
        Apply stable team assignments to all frames.

        Args:
            player_classids: Dictionary {frame_idx: {player_id: team_id}}

        Returns:
            Dictionary {frame_idx: {player_id: stable_team_id}} with stabilized assignments
        """
        if not self.stable_teams:
            raise RuntimeError("Must call analyze_team_assignments() before applying stable teams")

        stabilized_classids = {}

        for frame_idx, frame_assignments in player_classids.items():
            if frame_assignments is None:
                stabilized_classids[frame_idx] = None
                continue

            stabilized_frame = {}
            for player_id, _ in frame_assignments.items():
                # Replace with stable team (or keep original if not in stable_teams)
                stable_team = self.stable_teams.get(player_id, frame_assignments[player_id])
                stabilized_frame[player_id] = stable_team

            stabilized_classids[frame_idx] = stabilized_frame

        return stabilized_classids

    def stabilize_teams(self, player_classids: Dict) -> Dict:
        """
        Complete team stabilization pipeline.

        Analyzes all frames, determines stable teams via majority voting,
        and applies them consistently across the entire video.

        Args:
            player_classids: Dictionary {frame_idx: {player_id: team_id}}

        Returns:
            Dictionary {frame_idx: {player_id: stable_team_id}} with stabilized assignments
        """
        # Analyze to determine stable teams
        self.analyze_team_assignments(player_classids)

        # Apply stable teams to all frames
        stabilized_classids = self.apply_stable_teams(player_classids)

        return stabilized_classids

    def get_team_summary(self) -> Dict:
        """
        Get summary of stable team assignments.

        Returns:
            Dictionary with team assignment statistics
        """
        if not self.stable_teams:
            return {}

        # Count players per team
        team_counts = Counter(self.stable_teams.values())

        return {
            'total_players': len(self.stable_teams),
            'team_0_count': team_counts.get(0, 0),
            'team_1_count': team_counts.get(1, 0),
            'player_teams': dict(self.stable_teams)
        }
