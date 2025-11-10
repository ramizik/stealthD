"""
Team Assignment Stabilizer

Ensures consistent team assignments throughout the video by using
majority voting across all frames for each player ID.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_clustering.fast_features import FastFeatureExtractor


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
        self.player_sequences: Dict[int, List[Tuple[int, int]]] = {}
        self.player_team_counts: Dict[int, Counter] = {}
        self.player_agreement: Dict[int, float] = {}
        self.player_color_distances: Dict[int, Tuple[float, float]] = {}
        self.color_based_teams: Dict[int, int] = {}

    def analyze_team_assignments(self, player_classids: Dict) -> Dict[int, int]:
        """
        Analyze all team assignments and determine stable team for each player.

        Args:
            player_classids: Dictionary {frame_idx: {player_id: team_id}}

        Returns:
            Dictionary {player_id: stable_team_id}
        """
        # Collect all team assignments per player
        player_teams = defaultdict(list)  # {player_id: [(frame_idx, team_id), ...]}

        for frame_idx, frame_assignments in player_classids.items():
            if frame_assignments is None:
                continue

            for player_id, team_id in frame_assignments.items():
                player_teams[player_id].append((frame_idx, team_id))

        # Determine stable team for each player using majority voting
        stable_teams = {}

        for player_id in sorted(player_teams.keys()):
            team_assignments_raw = sorted(player_teams[player_id], key=lambda x: x[0])
            team_assignments = [team_id for _, team_id in team_assignments_raw]

            # Only stabilize if player appears in enough frames
            if len(team_assignments) < self.min_frames_threshold:
                # For players with few appearances, use their most common team anyway
                pass

            # Count team occurrences
            team_counter = Counter(team_assignments)

            total_assignments = len(team_assignments)
            majority_count = team_counter[team_counter.most_common(1)[0][0]]
            # Assign most common team (majority vote)
            stable_team = team_counter.most_common(1)[0][0]
            stable_teams[player_id] = stable_team

            # Log if there was significant disagreement (indicates correction)
            agreement_percentage = (majority_count / total_assignments) * 100 if total_assignments else 0.0
            if len(team_counter) > 1 and agreement_percentage < 90:
                print(f"  Player #{player_id}: Stabilized to team {stable_team} "
                      f"({agreement_percentage:.1f}% agreement, corrected {total_assignments - majority_count} frames)")

            # Store statistics for potential refinement
            self.player_sequences[player_id] = team_assignments_raw
            self.player_team_counts[player_id] = team_counter
            self.player_agreement[player_id] = agreement_percentage

        # Lightweight post-stabilization sanity check:
        # If there is exactly one outlier inside a fixed-ID range (1-11 or 12-22),
        # flip that one player's team to match the range majority.
        # This corrects lone persistent mistakes (commonly fixed ID #12).
        if stable_teams:
            # Fixed ID ranges
            team0_range_ids = [pid for pid in stable_teams.keys() if 1 <= pid <= 11]
            team1_range_ids = [pid for pid in stable_teams.keys() if 12 <= pid <= 22]

            # Labels within ranges
            team0_labels = [stable_teams[pid] for pid in team0_range_ids]
            team1_labels = [stable_teams[pid] for pid in team1_range_ids]

            def majority_or_default(labels, default_team):
                if not labels:
                    return default_team
                cnt = Counter(labels)
                return cnt.most_common(1)[0][0]

            def correct_single_outlier(ids_in_range, expected_team):
                outliers = [pid for pid in ids_in_range if stable_teams.get(pid, expected_team) != expected_team]
                inliers = [pid for pid in ids_in_range if stable_teams.get(pid, expected_team) == expected_team]
                # Flip only when there is a clear majority and exactly one outlier
                if len(outliers) == 1 and len(inliers) >= 5:
                    return outliers[0]
                return None

            expected_team_team0_range = majority_or_default(team0_labels, 0)
            expected_team_team1_range = majority_or_default(team1_labels, 1)

            flip_id_team0 = correct_single_outlier(team0_range_ids, expected_team_team0_range)
            flip_id_team1 = correct_single_outlier(team1_range_ids, expected_team_team1_range)

            if flip_id_team0 is not None:
                stable_teams[flip_id_team0] = expected_team_team0_range
                print(f"  TeamStabilizer: Corrected single outlier in IDs 1-11 → flipped #{flip_id_team0} to team {expected_team_team0_range}")
            if flip_id_team1 is not None:
                stable_teams[flip_id_team1] = expected_team_team1_range
                print(f"  TeamStabilizer: Corrected single outlier in IDs 12-22 → flipped #{flip_id_team1} to team {expected_team_team1_range}")

        self.stable_teams = stable_teams
        return stable_teams

    def refine_with_color(
        self,
        player_tracks: Dict,
        stabilized_classids: Dict,
        video_path: str,
        ambiguous_threshold: float = 70.0,
        seed_threshold: float = 85.0,
        max_samples_per_player: int = 20,
        color_margin: float = 0.05,
        balance_margin: float = 0.02,
    ) -> Dict:
        """
        Refine ambiguous player team assignments using jersey color histograms.

        Args:
            player_tracks: Dictionary {frame_idx: {player_id: bbox or dict}}
            stabilized_classids: Stabilized team assignments {frame_idx: {player_id: team_id}}
            video_path: Path to source video for extracting color samples
            ambiguous_threshold: Agreement % below which players are considered ambiguous
            seed_threshold: Agreement % required to serve as color prototype seed
            max_samples_per_player: Max frames to sample per player for color features
            color_margin: Required distance improvement (in L2) before flipping to opposite team
            balance_margin: Extra tolerance when balancing team counts (allows slight distance increase)

        Returns:
            Refined team assignments dictionary
        """
        if not self.player_sequences:
            raise RuntimeError("analyze_team_assignments() must be called before refine_with_color()")

        # Reset color diagnostics to current stable state by default
        self.player_color_distances = {}
        self.color_based_teams = dict(self.stable_teams)

        ambiguous_players = [
            player_id
            for player_id, agreement in self.player_agreement.items()
            if agreement < ambiguous_threshold and player_id in self.stable_teams
        ]

        if not ambiguous_players:
            return stabilized_classids

        # Identify high-confidence seeds per team
        seeds = {0: [], 1: []}
        for player_id, agreement in self.player_agreement.items():
            team = self.stable_teams.get(player_id)
            if team is None:
                continue
            if agreement >= seed_threshold:
                seeds[team].append(player_id)

        if not seeds[0] or not seeds[1]:
            print("  TeamStabilizer: Skipping color refinement (insufficient high-confidence seeds).")
            return stabilized_classids

        # Build lookup for bounding boxes per player
        player_bboxes: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
        for frame_idx, frame_tracks in player_tracks.items():
            if frame_tracks is None:
                continue
            for player_id, track_data in frame_tracks.items():
                if player_id == -1:
                    continue
                if isinstance(track_data, dict):
                    bbox = track_data.get('bbox')
                else:
                    bbox = track_data
                if bbox is None or any(b is None for b in bbox):
                    continue
                player_bboxes[player_id].append((frame_idx, bbox))

        players_of_interest = set(self.stable_teams.keys())
        frame_requests: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)

        def _select_samples(entries: List[Tuple[int, List[float]]]) -> List[Tuple[int, List[float]]]:
            if not entries:
                return []
            entries_sorted = sorted(entries, key=lambda x: x[0])
            if len(entries_sorted) <= max_samples_per_player:
                return entries_sorted
            indices = np.linspace(0, len(entries_sorted) - 1, max_samples_per_player, dtype=int)
            unique_indices = sorted(set(int(idx) for idx in indices))
            return [entries_sorted[idx] for idx in unique_indices]

        for player_id in players_of_interest:
            samples = _select_samples(player_bboxes.get(player_id, []))
            for frame_idx, bbox in samples:
                frame_requests[frame_idx].append((player_id, bbox))

        if not frame_requests:
            print("  TeamStabilizer: Skipping color refinement (no frame samples available).")
            return stabilized_classids

        feature_extractor = FastFeatureExtractor()
        player_features: Dict[int, List[np.ndarray]] = defaultdict(list)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"  TeamStabilizer: Could not open video for color refinement: {video_path}")
            return stabilized_classids

        frame_idx = 0
        requested_frames = set(frame_requests.keys())

        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_idx in requested_frames:
                height, width = frame.shape[:2]
                for player_id, bbox in frame_requests[frame_idx]:
                    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width, x2))
                    y2 = max(0, min(height, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    feature = feature_extractor.extract_color_histogram(crop)
                    player_features[player_id].append(feature)
            frame_idx += 1

        capture.release()

        # Compute team prototypes from high-confidence seeds
        team_prototypes: Dict[int, np.ndarray] = {}
        for team_id in (0, 1):
            seed_vectors = []
            for player_id in seeds[team_id]:
                vectors = player_features.get(player_id)
                if vectors:
                    seed_vectors.append(np.mean(vectors, axis=0))
            if seed_vectors:
                team_prototypes[team_id] = np.mean(seed_vectors, axis=0)

        # Fallback: if a team lacks seeds but we still gathered features, build prototype from nearest players
        for team_id in (0, 1):
            if team_id not in team_prototypes and seeds[team_id]:
                continue
            if team_id not in team_prototypes:
                candidate_vectors = []
                for player_id, vectors in player_features.items():
                    if vectors and self.stable_teams.get(player_id) == team_id:
                        candidate_vectors.append(np.mean(vectors, axis=0))
                if candidate_vectors:
                    team_prototypes[team_id] = np.mean(candidate_vectors, axis=0)

        if len(team_prototypes) < 2:
            print("  TeamStabilizer: Skipping color refinement (failed to build team prototypes).")
            return stabilized_classids

        # Cache distances for all players with features
        player_color_distances: Dict[int, Tuple[float, float]] = {}
        for player_id, vectors in player_features.items():
            if not vectors:
                continue
            player_vector = np.mean(vectors, axis=0)
            dist0 = float(np.linalg.norm(player_vector - team_prototypes[0]))
            dist1 = float(np.linalg.norm(player_vector - team_prototypes[1]))
            if np.isnan(dist0) or np.isnan(dist1):
                continue
            player_color_distances[player_id] = (dist0, dist1)

        # Store for external inspection/debugging
        self.player_color_distances = player_color_distances
        self.color_based_teams = {
            player_id: (0 if dists[0] <= dists[1] else 1)
            for player_id, dists in player_color_distances.items()
        }
        # Ensure players without color distances retain stable assignment
        for player_id, team in self.stable_teams.items():
            self.color_based_teams.setdefault(player_id, team)

        updated_players = []
        for player_id in ambiguous_players:
            if player_id not in player_color_distances:
                continue
            dist_team0, dist_team1 = player_color_distances[player_id]
            current_team = self.stable_teams.get(player_id)
            if current_team is None:
                continue
            other_team = 1 - current_team
            current_dist = dist_team0 if current_team == 0 else dist_team1
            other_dist = dist_team1 if current_team == 0 else dist_team0

            if other_dist + color_margin < current_dist:
                self.stable_teams[player_id] = other_team
                updated_players.append(player_id)

        if not updated_players:
            return stabilized_classids

        # Balance team counts to maintain near-equal distribution
        team_counts = Counter(self.stable_teams.values())
        total_players = len(self.stable_teams)
        expected_per_team = total_players // 2  # Assumes near-even teams (22 -> 11)

        def _rebalance(team_id: int):
            nonlocal team_counts
            if team_counts[team_id] <= expected_per_team:
                return
            candidates = []
            for player_id, team in self.stable_teams.items():
                if team != team_id:
                    continue
                if player_id not in player_color_distances:
                    continue
                if player_id in seeds[team_id]:
                    continue  # avoid flipping high-confidence seeds
                dist0, dist1 = player_color_distances[player_id]
                current_dist = dist0 if team_id == 0 else dist1
                other_dist = dist1 if team_id == 0 else dist0
                margin = current_dist - other_dist
                candidates.append((margin, player_id, current_dist, other_dist))

            candidates.sort()  # smallest margin first (closest to other team)
            for margin, player_id, current_dist, other_dist in candidates:
                if team_counts[team_id] <= expected_per_team:
                    break
                # Allow move if other team distance not significantly worse
                if other_dist <= current_dist + balance_margin:
                    new_team = 1 - team_id
                    if self.stable_teams[player_id] != new_team:
                        self.stable_teams[player_id] = new_team
                        if player_id not in updated_players:
                            updated_players.append(player_id)
                    team_counts[team_id] -= 1
                    team_counts[new_team] += 1

        _rebalance(0)
        _rebalance(1)

        # Apply refined teams to all frames
        for frame_idx, frame_assignments in stabilized_classids.items():
            if frame_assignments is None:
                continue
            for player_id in updated_players:
                if player_id in frame_assignments:
                    frame_assignments[player_id] = self.stable_teams[player_id]

        print(f"  TeamStabilizer: Refined {len(updated_players)} ambiguous player(s) via jersey color: {sorted(updated_players)}")
        return stabilized_classids

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
