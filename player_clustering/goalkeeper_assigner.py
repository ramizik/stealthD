"""
Goalkeeper Assignment module for assigning goalkeepers to teams.

Assigns goalkeepers to teams based on proximity to team centroids,
ensuring accurate team identification even when goalkeepers wear
different colors than field players.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Optional

import numpy as np
import supervision as sv


class GoalkeeperAssigner:
    """
    Assigns goalkeepers to teams based on spatial proximity to team centroids.

    This handles cases where goalkeepers wear different colors than their teammates,
    making color-based clustering unreliable for goalkeeper team assignment.
    """

    def __init__(self):
        """Initialize the goalkeeper assigner."""
        pass

    def assign_goalkeeper_teams(
        self,
        players: sv.Detections,
        goalkeepers: sv.Detections,
        player_team_ids: np.ndarray,
        frame_width: int = 1920
    ) -> np.ndarray:
        """
        Assign goalkeepers to teams based on their position on the field.

        IMPROVED STRATEGY (position-based instead of centroid-based):
        - Goalkeepers on LEFT half of frame (x < frame_width/2) → Team 0
        - Goalkeepers on RIGHT half of frame (x > frame_width/2) → Team 1

        This is more reliable than centroid-based assignment because:
        1. Goalkeepers stay at their goal positions
        2. Works even with few field players visible
        3. Immune to field player clustering/positioning
        4. Works with brief goalkeeper appearances

        Args:
            players: Field player detections
            goalkeepers: Goalkeeper detections
            player_team_ids: Team assignments for field players (0 or 1)
            frame_width: Video frame width in pixels (default 1920)

        Returns:
            Team IDs for goalkeepers (0 or 1)
        """
        if len(goalkeepers) == 0:
            return np.array([])

        # Get goalkeeper positions (bottom-center)
        goalkeeper_positions = goalkeepers.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )

        # Assign teams based on x-position: left half = Team 0, right half = Team 1
        goalkeeper_team_ids = []
        frame_center_x = frame_width / 2.0

        for gk_pos in goalkeeper_positions:
            x_pos = gk_pos[0]
            # Left half of frame → Team 0, Right half → Team 1
            team_id = 0 if x_pos < frame_center_x else 1
            goalkeeper_team_ids.append(team_id)

        return np.array(goalkeeper_team_ids)

    def assign_goalkeeper_teams_from_tracks(
        self,
        player_tracks: dict,
        goalkeeper_tracks: dict,
        team_assignments: dict,
        frame_idx: int
    ) -> Optional[dict]:
        """
        Assign goalkeepers to teams for a specific frame using track data.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: bbox}}
            goalkeeper_tracks: Dictionary {frame_idx: {gk_id: bbox}}
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}
            frame_idx: Frame index to process

        Returns:
            Dictionary {gk_id: team_id} for goalkeepers in the frame, or None if no data
        """
        if frame_idx not in player_tracks or frame_idx not in goalkeeper_tracks:
            return None

        if frame_idx not in team_assignments:
            return None

        # Convert tracks to detections
        player_bboxes = list(player_tracks[frame_idx].values())
        gk_bboxes = list(goalkeeper_tracks[frame_idx].values())
        gk_ids = list(goalkeeper_tracks[frame_idx].keys())

        if len(player_bboxes) == 0 or len(gk_bboxes) == 0:
            return None

        # Create supervision Detections objects
        players = sv.Detections(xyxy=np.array(player_bboxes))
        goalkeepers = sv.Detections(xyxy=np.array(gk_bboxes))

        # Get team IDs for field players
        player_ids = list(player_tracks[frame_idx].keys())
        player_team_ids = np.array([
            team_assignments[frame_idx].get(pid, 0)
            for pid in player_ids
        ])

        # Assign goalkeeper teams
        gk_team_ids = self.assign_goalkeeper_teams(players, goalkeepers, player_team_ids)

        # Create result dictionary
        gk_assignments = {gk_id: int(team_id) for gk_id, team_id in zip(gk_ids, gk_team_ids)}

        return gk_assignments