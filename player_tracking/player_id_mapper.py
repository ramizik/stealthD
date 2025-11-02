"""
Player ID Mapper module for consistent player ID assignment.

Maps ByteTrack IDs to fixed player IDs:
- Players: 1-22 (1-11 for Team 0, 12-22 for Team 1)
- Referee: 0
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict


class PlayerIDMapper:
    """
    Maps dynamic ByteTrack IDs to consistent fixed player IDs.

    Assigns IDs based on:
    - Team assignment (determines range: 1-11 or 12-22)
    - Player track history and position consistency
    """

    def __init__(self):
        """Initialize the player ID mapper."""
        self.bytetrack_to_fixed: Dict[int, int] = {}  # {bytetrack_id: fixed_id}
        self.fixed_to_bytetrack: Dict[int, int] = {}  # {fixed_id: bytetrack_id} for reverse lookup
        self.team_assignments: Dict[int, int] = {}  # {bytetrack_id: team_id}
        self.track_statistics: Dict[int, Dict] = {}  # {bytetrack_id: {team_counts, frame_count, etc}}

        # Fixed ID ranges
        self.TEAM_0_MIN = 1
        self.TEAM_0_MAX = 11
        self.TEAM_1_MIN = 12
        self.TEAM_1_MAX = 22
        self.REFEREE_ID = 0

        # Next available IDs for each team
        self.next_team0_id = 1
        self.next_team1_id = 12

    def analyze_tracks(self, player_tracks: Dict, team_assignments: Dict):
        """
        Analyze all tracks to build ID mapping.

        Args:
            player_tracks: Dictionary {frame_idx: {bytetrack_id: bbox}}
            team_assignments: Dictionary {frame_idx: {bytetrack_id: team_id}}
        """
        print("Analyzing tracks to build consistent ID mapping...")

        # Collect statistics for each ByteTrack ID
        for frame_idx, frame_tracks in player_tracks.items():
            if -1 in frame_tracks:
                continue  # Skip invalid frames

            for bytetrack_id in frame_tracks.keys():
                if bytetrack_id not in self.track_statistics:
                    self.track_statistics[bytetrack_id] = {
                        'team_counts': defaultdict(int),
                        'total_frames': 0,
                        'first_frame': frame_idx,
                        'last_frame': frame_idx
                    }

                # Update statistics
                self.track_statistics[bytetrack_id]['total_frames'] += 1
                self.track_statistics[bytetrack_id]['last_frame'] = frame_idx

                # Count team assignments
                if frame_idx in team_assignments and bytetrack_id in team_assignments[frame_idx]:
                    team_id = team_assignments[frame_idx][bytetrack_id]
                    self.track_statistics[bytetrack_id]['team_counts'][team_id] += 1

        # Determine team for each ByteTrack ID (most common team)
        for bytetrack_id, stats in self.track_statistics.items():
            if stats['team_counts']:
                # Get most common team
                most_common_team = max(stats['team_counts'].items(), key=lambda x: x[1])[0]
                self.team_assignments[bytetrack_id] = most_common_team
            else:
                # No team assignment found, default to team 0
                self.team_assignments[bytetrack_id] = 0

        # Build mapping: sort by first appearance, assign IDs within team ranges
        # Separate by team
        team0_tracks = []
        team1_tracks = []

        for bytetrack_id, team_id in self.team_assignments.items():
            stats = self.track_statistics[bytetrack_id]
            track_info = {
                'bytetrack_id': bytetrack_id,
                'team': team_id,
                'first_frame': stats['first_frame'],
                'total_frames': stats['total_frames']
            }

            if team_id == 0:
                team0_tracks.append(track_info)
            else:
                team1_tracks.append(track_info)

        # Sort by total frames (descending) - most present players get fixed IDs
        # This ensures the 11 most visible players per team get IDs 1-22
        team0_tracks.sort(key=lambda x: (-x['total_frames'], x['first_frame']))
        team1_tracks.sort(key=lambda x: (-x['total_frames'], x['first_frame']))

        # Assign fixed IDs for Team 0 (1-11) - top 11 most visible players
        team0_mapped = 0
        for i, track_info in enumerate(team0_tracks[:11]):  # Top 11 by frame presence
            fixed_id = self.TEAM_0_MIN + i
            self.bytetrack_to_fixed[track_info['bytetrack_id']] = fixed_id
            self.fixed_to_bytetrack[fixed_id] = track_info['bytetrack_id']
            self.next_team0_id = max(self.next_team0_id, fixed_id + 1)
            team0_mapped += 1

        # Assign fixed IDs for Team 1 (12-22) - top 11 most visible players
        team1_mapped = 0
        for i, track_info in enumerate(team1_tracks[:11]):  # Top 11 by frame presence
            fixed_id = self.TEAM_1_MIN + i
            self.bytetrack_to_fixed[track_info['bytetrack_id']] = fixed_id
            self.fixed_to_bytetrack[fixed_id] = track_info['bytetrack_id']
            self.next_team1_id = max(self.next_team1_id, fixed_id + 1)
            team1_mapped += 1

        total_mapped = len(self.bytetrack_to_fixed)
        print(f"ID Mapping complete: {total_mapped} players mapped to fixed IDs 1-22")
        print(f"  Team 0: {team0_mapped} players (IDs 1-{team0_mapped})")
        print(f"  Team 1: {team1_mapped} players (IDs 12-{11 + team1_mapped})")

        # Report unmapped tracks (likely tracking artifacts or brief appearances)
        total_tracks = len(team0_tracks) + len(team1_tracks)
        if total_tracks > 22:
            unmapped_count = total_tracks - 22
            print(f"  Note: {unmapped_count} additional tracks detected (likely tracking fragments or brief appearances)")

    def map_player_tracks(self, player_tracks: Dict) -> Dict:
        """
        Remap player tracks from ByteTrack IDs to fixed IDs.
        Only the 22 mapped players (11 per team) are kept in output.

        Args:
            player_tracks: Dictionary {frame_idx: {bytetrack_id: bbox}}

        Returns:
            Dictionary {frame_idx: {fixed_id: bbox}}
        """
        remapped_tracks = {}

        for frame_idx, frame_tracks in player_tracks.items():
            if -1 in frame_tracks:
                remapped_tracks[frame_idx] = {-1: [None]*4}
                continue

            remapped_frame = {}
            for bytetrack_id, bbox in frame_tracks.items():
                if bytetrack_id in self.bytetrack_to_fixed:
                    # Use mapped fixed ID (1-22)
                    fixed_id = self.bytetrack_to_fixed[bytetrack_id]
                    remapped_frame[fixed_id] = bbox
                # Unmapped tracks are ignored (likely tracking artifacts after stitching)

            remapped_tracks[frame_idx] = remapped_frame if remapped_frame else {-1: [None]*4}

        return remapped_tracks

    def map_team_assignments(self, team_assignments: Dict) -> Dict:
        """
        Remap team assignments from ByteTrack IDs to fixed IDs.
        Only the 22 mapped players (11 per team) are kept in output.

        Args:
            team_assignments: Dictionary {frame_idx: {bytetrack_id: team_id}}

        Returns:
            Dictionary {frame_idx: {fixed_id: team_id}}
        """
        remapped_assignments = {}

        for frame_idx, frame_teams in team_assignments.items():
            remapped_frame = {}
            for bytetrack_id, team_id in frame_teams.items():
                if bytetrack_id in self.bytetrack_to_fixed:
                    # Use mapped fixed ID (1-22)
                    fixed_id = self.bytetrack_to_fixed[bytetrack_id]
                    remapped_frame[fixed_id] = team_id
                # Unmapped team assignments are ignored
            remapped_assignments[frame_idx] = remapped_frame

        return remapped_assignments

    def map_referee_tracks(self, referee_tracks: Dict) -> Dict:
        """
        Remap referee tracks to always use ID 0.

        Args:
            referee_tracks: Dictionary {frame_idx: {referee_id: bbox}}

        Returns:
            Dictionary {frame_idx: {0: bbox}} (always ID 0)
        """
        remapped_tracks = {}

        for frame_idx, frame_tracks in referee_tracks.items():
            if -1 in frame_tracks:
                remapped_tracks[frame_idx] = {-1: [None]*4}
                continue

            # Combine all referees into single ID 0 (take first/primary referee)
            if len(frame_tracks) > 0:
                first_referee_bbox = list(frame_tracks.values())[0]
                remapped_tracks[frame_idx] = {self.REFEREE_ID: first_referee_bbox}
            else:
                remapped_tracks[frame_idx] = {-1: [None]*4}

        return remapped_tracks

    def get_fixed_id(self, bytetrack_id: int) -> Optional[int]:
        """Get fixed ID for a ByteTrack ID, or None if not mapped."""
        return self.bytetrack_to_fixed.get(bytetrack_id)

    def get_bytetrack_id(self, fixed_id: int) -> Optional[int]:
        """Get ByteTrack ID for a fixed ID, or None if not mapped."""
        return self.fixed_to_bytetrack.get(fixed_id)

    def get_mapping_summary(self) -> Dict:
        """Get summary of ID mappings."""
        return {
            'bytetrack_to_fixed': self.bytetrack_to_fixed,
            'fixed_to_bytetrack': self.fixed_to_bytetrack,
            'team_assignments': self.team_assignments
        }
