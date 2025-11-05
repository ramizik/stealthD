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
        player_team_ids: np.ndarray
    ) -> np.ndarray:
        """
        Assign goalkeepers to teams based on proximity to team centroids.

        Strategy:
        1. Calculate centroid of each team's field players
        2. Assign each goalkeeper to the closest team centroid

        Args:
            players: Field player detections
            goalkeepers: Goalkeeper detections
            player_team_ids: Team assignments for field players (0 or 1)

        Returns:
            Team IDs for goalkeepers (0 or 1)
        """
        if len(goalkeepers) == 0:
            return np.array([])

        if len(players) == 0 or len(player_team_ids) == 0:
            # No field players to compare against, default to team 0
            return np.zeros(len(goalkeepers), dtype=int)

        # Get bottom-center positions (feet)
        goalkeeper_positions = goalkeepers.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )
        player_positions = players.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )

        # Calculate team centroids
        team_0_mask = player_team_ids == 0
        team_1_mask = player_team_ids == 1

        # Handle edge cases where one team has no players
        if team_0_mask.sum() == 0 and team_1_mask.sum() == 0:
            return np.zeros(len(goalkeepers), dtype=int)
        elif team_0_mask.sum() == 0:
            return np.ones(len(goalkeepers), dtype=int)
        elif team_1_mask.sum() == 0:
            return np.zeros(len(goalkeepers), dtype=int)

        team_0_centroid = player_positions[team_0_mask].mean(axis=0)
        team_1_centroid = player_positions[team_1_mask].mean(axis=0)

        # Assign each goalkeeper to closest team
        goalkeeper_team_ids = []
        for gk_pos in goalkeeper_positions:
            dist_0 = np.linalg.norm(gk_pos - team_0_centroid)
            dist_1 = np.linalg.norm(gk_pos - team_1_centroid)
            goalkeeper_team_ids.append(0 if dist_0 < dist_1 else 1)

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