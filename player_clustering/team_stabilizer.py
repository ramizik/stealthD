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

    def stabilize_teams(self, player_classids: Dict, player_tracks: Dict = None) -> Dict:
        """
        Complete team stabilization pipeline.

        Analyzes all frames, determines stable teams via majority voting,
        and applies them consistently across the entire video.

        Args:
            player_classids: Dictionary {frame_idx: {player_id: team_id}}
            player_tracks: Optional dictionary {frame_idx: {player_id: bbox}} for spatial correction

        Returns:
            Dictionary {frame_idx: {player_id: stable_team_id}} with stabilized assignments
        """
        # Analyze to determine stable teams
        self.analyze_team_assignments(player_classids)

        # Apply stable teams to all frames
        stabilized_classids = self.apply_stable_teams(player_classids)

        # Post-processing: Use spatial consistency to correct obvious misclassifications
        if player_tracks is not None:
            stabilized_classids = self._correct_spatial_outliers(
                stabilized_classids, player_tracks
            )

        return stabilized_classids

    def _correct_spatial_outliers(self, player_classids: Dict, player_tracks: Dict) -> Dict:
        """
        Correct obvious team assignment errors using spatial consistency.

        Only corrects players with low agreement (<60%) and strong spatial evidence
        that they're on the wrong team.

        Args:
            player_classids: Dictionary {frame_idx: {player_id: team_id}}
            player_tracks: Dictionary {frame_idx: {player_id: bbox}}

        Returns:
            Corrected team assignments
        """
        import numpy as np
        from collections import Counter

        # First, identify players with low agreement during stabilization
        # We need to re-analyze to get agreement percentages
        player_teams = {}
        for frame_idx, frame_assignments in player_classids.items():
            if frame_assignments is None:
                continue
            for player_id, team_id in frame_assignments.items():
                if player_id not in player_teams:
                    player_teams[player_id] = []
                player_teams[player_id].append(team_id)

        # Find players with low agreement (<60%)
        low_agreement_players = []
        for player_id, team_assignments in player_teams.items():
            if len(team_assignments) < 10:  # Skip players with few appearances
                continue
            team_counter = Counter(team_assignments)
            stable_team, count = team_counter.most_common(1)[0]
            agreement = count / len(team_assignments)
            if agreement < 0.60:  # Less than 60% agreement
                low_agreement_players.append(player_id)

        if len(low_agreement_players) == 0:
            return player_classids

        # For each low-agreement player, check spatial consistency across frames
        player_spatial_evidence = {}  # {player_id: {team_id: count}}
        for player_id in low_agreement_players:
            player_spatial_evidence[player_id] = Counter()

            for frame_idx, frame_assignments in player_classids.items():
                if (frame_assignments is None or frame_idx not in player_tracks or
                    player_id not in frame_assignments or player_id not in player_tracks[frame_idx]):
                    continue

                frame_tracks = player_tracks[frame_idx]
                if -1 in frame_tracks:
                    continue

                player_bbox = frame_tracks[player_id]
                player_center_x = (player_bbox[0] + player_bbox[2]) / 2.0
                player_center_y = (player_bbox[1] + player_bbox[3]) / 2.0

                # Find nearby players (within 250 pixels)
                nearby_teams = []
                for other_id, other_bbox in frame_tracks.items():
                    if other_id == player_id or other_id not in frame_assignments:
                        continue

                    other_center_x = (other_bbox[0] + other_bbox[2]) / 2.0
                    other_center_y = (other_bbox[1] + other_bbox[3]) / 2.0

                    distance = np.sqrt((player_center_x - other_center_x)**2 +
                                     (player_center_y - other_center_y)**2)

                    if distance < 250:  # Within 250 pixels
                        nearby_teams.append(frame_assignments[other_id])

                # If most nearby players are on a specific team, count as evidence
                if len(nearby_teams) >= 3:
                    team_counts = Counter(nearby_teams)
                    dominant_team, count = team_counts.most_common(1)[0]
                    if count / len(nearby_teams) >= 0.75:  # 75%+ nearby players on same team
                        player_spatial_evidence[player_id][dominant_team] += 1

        # Determine which players should be corrected
        players_to_correct = {}
        for player_id, evidence in player_spatial_evidence.items():
            if len(evidence) == 0:
                continue
            dominant_team, count = evidence.most_common(1)[0]
            current_team = self.stable_teams.get(player_id)

            # Only correct if:
            # 1. Spatial evidence strongly suggests a different team
            # 2. Evidence appears in at least 20% of frames where player appears
            total_frames = len(player_teams.get(player_id, []))
            if (current_team is not None and dominant_team != current_team and
                count >= max(5, total_frames * 0.20)):  # At least 5 frames or 20% of appearances
                players_to_correct[player_id] = dominant_team

        if len(players_to_correct) == 0:
            return player_classids

        # Apply corrections
        corrected_classids = {}
        for frame_idx, frame_assignments in player_classids.items():
            if frame_assignments is None:
                corrected_classids[frame_idx] = None
                continue

            corrected_frame = frame_assignments.copy()
            for player_id, correct_team in players_to_correct.items():
                if player_id in corrected_frame:
                    corrected_frame[player_id] = correct_team

            corrected_classids[frame_idx] = corrected_frame

        # Update stable teams
        for player_id, correct_team in players_to_correct.items():
            self.stable_teams[player_id] = correct_team
            print(f"  Player #{player_id}: Corrected to team {correct_team} based on spatial consistency")

        return corrected_classids

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
