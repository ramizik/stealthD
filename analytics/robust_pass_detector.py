"""
Robust Pass Detector with fallback validation when ball coordinates are unreliable.

This version handles cases where:
1. Ball field coordinates are missing or unstable
2. Homography transformation has low success rate
3. Ball movement appears as 0.0m due to coordinate issues
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class RobustPassDetector:
    """
    Pass detector with robust fallback validation.
    """

    def __init__(self,
                 min_pass_distance: float = 0.3,
                 max_pass_distance: float = 70.0,
                 max_gap_frames: int = 150,
                 velocity_threshold: float = 0.5,
                 min_ball_movement: float = 2.0,
                 enable_fallback_mode: bool = True):
        """
        Initialize robust pass detector.

        Args:
            enable_fallback_mode: If True, uses player positions when ball coords are unreliable
        """
        self.min_pass_distance = min_pass_distance
        self.max_pass_distance = max_pass_distance
        self.max_gap_frames = max_gap_frames
        self.velocity_threshold = velocity_threshold
        self.min_ball_movement = min_ball_movement
        self.enable_fallback_mode = enable_fallback_mode

    def _normalize_to_meters(self, coord: np.ndarray) -> np.ndarray:
        """Robust coordinate normalization."""
        coord = np.asarray(coord, dtype=float)[:2]

        # Already in meters
        if 0 <= coord[0] <= 130 and 0 <= coord[1] <= 90:
            return coord

        # Pitch coordinates
        if coord[0] > 1000 or coord[1] > 1000:
            return np.array([
                coord[0] * (105.0 / 12000.0),
                coord[1] * (68.0 / 7000.0)
            ])

        # Scaled coordinates
        if coord[0] > 200 or coord[1] > 200:
            scale_x = 105.0 / coord[0] if coord[0] > 0 else 0.1
            scale_y = 68.0 / coord[1] if coord[1] > 0 else 0.1
            scale = (scale_x + scale_y) / 2
            return coord * scale

        return coord

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate distance with normalization."""
        if point1 is None or point2 is None:
            return float('inf')

        p1 = self._normalize_to_meters(np.asarray(point1)[:2])
        p2 = self._normalize_to_meters(np.asarray(point2)[:2])

        return float(np.linalg.norm(p1 - p2))

    def _check_ball_coord_quality(self, ball_field_coords: Dict,
                                  start_frame: int, end_frame: int) -> Tuple[bool, float]:
        """
        Check if ball field coordinates are reliable in the given frame range.

        Returns:
            (is_reliable, coverage_percentage)
        """
        if not ball_field_coords:
            return False, 0.0

        total_frames = end_frame - start_frame + 1
        frames_with_coords = 0

        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in ball_field_coords:
                coords = ball_field_coords[frame_idx]
                if coords is not None and len(coords) == 2:
                    # Check if coordinates are valid (not all zeros)
                    if coords[0] != 0.0 or coords[1] != 0.0:
                        frames_with_coords += 1

        coverage = frames_with_coords / total_frames if total_frames > 0 else 0.0
        is_reliable = coverage >= 0.4  # Need at least 40% coverage

        return is_reliable, coverage

    def _get_ball_trajectory(self, ball_field_coords: Dict,
                            start_frame: int, end_frame: int) -> Optional[Dict]:
        """Analyze ball trajectory with quality check."""
        # Check quality first
        is_reliable, coverage = self._check_ball_coord_quality(
            ball_field_coords, start_frame, end_frame
        )

        if not is_reliable:
            return None

        # Get all ball positions
        positions = []
        frames = []

        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in ball_field_coords:
                pos = ball_field_coords[frame_idx]
                if pos is not None and len(pos) == 2:
                    if pos[0] != 0.0 or pos[1] != 0.0:  # Skip zero coords
                        positions.append(self._normalize_to_meters(np.array(pos)))
                        frames.append(frame_idx)

        if len(positions) < 2:
            return None

        positions = np.array(positions)

        # Calculate metrics
        total_distance = 0
        for i in range(len(positions) - 1):
            total_distance += np.linalg.norm(positions[i+1] - positions[i])

        direct_distance = np.linalg.norm(positions[-1] - positions[0])
        efficiency = direct_distance / total_distance if total_distance > 0 else 0

        time_elapsed = (frames[-1] - frames[0]) / 30.0
        avg_velocity = total_distance / time_elapsed if time_elapsed > 0 else 0

        return {
            'positions': positions,
            'frames': frames,
            'total_distance': total_distance,
            'direct_distance': direct_distance,
            'efficiency': efficiency,
            'avg_velocity': avg_velocity,
            'num_positions': len(positions),
            'coverage': coverage
        }

    def _validate_pass_fallback(self, passer_pos: np.ndarray, receiver_pos: np.ndarray,
                                gap_frames: int, fps: float = 30.0) -> Tuple[bool, float, str]:
        """
        Fallback validation using only player positions when ball coords are unreliable.

        This is less accurate but allows detection when ball tracking fails.
        """
        passer_pos = self._normalize_to_meters(passer_pos)
        receiver_pos = self._normalize_to_meters(receiver_pos)

        player_distance = np.linalg.norm(receiver_pos - passer_pos)

        # Check distance bounds
        if player_distance < self.min_pass_distance:
            return False, 0.0, f"too_close_{player_distance:.1f}m"
        if player_distance > self.max_pass_distance:
            return False, 0.0, f"too_far_{player_distance:.1f}m"

        # Use gap frames as proxy for pass likelihood
        time_gap = gap_frames / fps

        # Short gaps are more likely to be valid passes
        if gap_frames <= 10:  # ~0.33 seconds
            confidence = 0.75
            reason = f"fallback_short_gap_{time_gap:.2f}s"
        elif gap_frames <= 30:  # ~1 second
            confidence = 0.65
            reason = f"fallback_medium_gap_{time_gap:.2f}s"
        elif gap_frames <= 60:  # ~2 seconds
            confidence = 0.50
            reason = f"fallback_long_gap_{time_gap:.2f}s"
        else:
            # Very long gaps - require shorter distance
            if player_distance < 20.0:
                confidence = 0.40
                reason = f"fallback_very_long_gap_{time_gap:.2f}s"
            else:
                return False, 0.0, f"gap_too_long_{time_gap:.2f}s"

        # Adjust confidence based on distance
        # Typical passes are 5-30m
        if 5.0 <= player_distance <= 30.0:
            confidence *= 1.1  # Boost for typical pass distance
            reason += "+typical_distance"
        elif player_distance < 5.0:
            confidence *= 0.9  # Reduce for very short
            reason += "+short_distance"
        elif player_distance > 40.0:
            confidence *= 0.8  # Reduce for very long
            reason += "+long_distance"

        confidence = min(confidence, 1.0)

        return True, confidence, reason

    def _validate_pass_with_trajectory(self, passer_pos: np.ndarray, receiver_pos: np.ndarray,
                                      trajectory: Optional[Dict], gap_frames: int,
                                      fps: float = 30.0) -> Tuple[bool, float, str]:
        """
        Validate pass with trajectory or fallback to player positions.
        """
        # If trajectory data is unreliable or missing, use fallback
        if trajectory is None or trajectory['num_positions'] < 2:
            if self.enable_fallback_mode:
                return self._validate_pass_fallback(passer_pos, receiver_pos, gap_frames, fps)
            else:
                # Without fallback, require minimum trajectory data
                if gap_frames <= 10:
                    return True, 0.5, "no_trajectory_short_gap"
                return False, 0.0, "no_trajectory_data"

        # Full trajectory validation
        passer_pos = self._normalize_to_meters(passer_pos)
        receiver_pos = self._normalize_to_meters(receiver_pos)

        player_distance = np.linalg.norm(receiver_pos - passer_pos)

        if player_distance < self.min_pass_distance:
            return False, 0.0, f"too_close_{player_distance:.1f}m"
        if player_distance > self.max_pass_distance:
            return False, 0.0, f"too_far_{player_distance:.1f}m"

        ball_distance = trajectory['direct_distance']

        # Relaxed ball movement requirement if coverage is low
        min_movement = self.min_ball_movement
        if trajectory['coverage'] < 0.6:
            min_movement = 1.0  # Lower threshold for sparse data

        if ball_distance < min_movement:
            # Try fallback if enabled
            if self.enable_fallback_mode:
                return self._validate_pass_fallback(passer_pos, receiver_pos, gap_frames, fps)
            return False, 0.0, f"ball_didnt_move_{ball_distance:.1f}m"

        confidence = 1.0
        reasons = []

        # Direction check
        if len(trajectory['positions']) >= 2:
            passer_to_receiver = receiver_pos - passer_pos
            ball_start = trajectory['positions'][0]
            ball_end = trajectory['positions'][-1]
            ball_movement = ball_end - ball_start

            if np.linalg.norm(passer_to_receiver) > 0 and np.linalg.norm(ball_movement) > 0:
                direction_similarity = np.dot(passer_to_receiver, ball_movement) / (
                    np.linalg.norm(passer_to_receiver) * np.linalg.norm(ball_movement)
                )

                if direction_similarity < 0.3:
                    confidence *= 0.5
                    reasons.append(f"direction_mismatch_{direction_similarity:.2f}")
                else:
                    reasons.append(f"good_direction_{direction_similarity:.2f}")

        # Velocity check - relaxed for sparse data
        velocity = trajectory['avg_velocity']
        if velocity < self.velocity_threshold:
            if trajectory['coverage'] > 0.6:
                confidence *= 0.7  # Penalize more with good coverage
            else:
                confidence *= 0.85  # Penalize less with sparse coverage
            reasons.append(f"low_velocity_{velocity:.1f}ms")
        else:
            reasons.append(f"good_velocity_{velocity:.1f}ms")

        # Trajectory efficiency
        efficiency = trajectory['efficiency']
        if efficiency < 0.5:
            confidence *= 0.8
            reasons.append(f"curved_path_{efficiency:.2f}")
        else:
            reasons.append(f"straight_path_{efficiency:.2f}")

        # Distance ratio
        distance_ratio = ball_distance / player_distance if player_distance > 0 else 0
        if distance_ratio < 0.5 or distance_ratio > 2.0:
            confidence *= 0.7
            reasons.append(f"distance_mismatch_{distance_ratio:.2f}")
        else:
            reasons.append(f"good_distance_{distance_ratio:.2f}")

        # Add coverage info
        reasons.append(f"coverage_{trajectory['coverage']:.1%}")

        reason_str = "+".join(reasons)
        return True, confidence, reason_str

    def _get_player_team_robust(self, player_id: int, center_frame: int,
                               team_assignments: Dict, window: int = 50) -> Optional[int]:
        """Get player team with temporal analysis."""
        team_votes = []

        for offset in range(-window, window + 1):
            frame_idx = center_frame + offset
            if frame_idx in team_assignments and player_id in team_assignments[frame_idx]:
                team_votes.append(team_assignments[frame_idx][player_id])

        if not team_votes:
            for frame_idx, frame_teams in team_assignments.items():
                if player_id in frame_teams:
                    team_votes.append(frame_teams[player_id])

        if not team_votes:
            return None

        counter = Counter(team_votes)
        most_common_team, count = counter.most_common(1)[0]
        agreement = count / len(team_votes)

        if agreement >= 0.6:
            return most_common_team

        return None

    def _extract_touch_events_improved(self, ball_assignments: Dict) -> List[Dict]:
        """Extract touch events."""
        touch_events = []
        prev_player = None
        possession_start = None

        sorted_frames = sorted(ball_assignments.keys())

        for i, frame_idx in enumerate(sorted_frames):
            current_player = ball_assignments[frame_idx]

            if current_player != prev_player:
                if prev_player is not None and possession_start is not None:
                    release_frame = sorted_frames[i - 1] if i > 0 else possession_start
                    touch_events.append({
                        'frame': int(release_frame),
                        'player_id': int(prev_player),
                        'event_type': 'release',
                        'possession_duration': release_frame - possession_start
                    })

                if current_player is not None:
                    touch_events.append({
                        'frame': int(frame_idx),
                        'player_id': int(current_player),
                        'event_type': 'receive'
                    })
                    possession_start = frame_idx

                prev_player = current_player

        if prev_player is not None and len(sorted_frames) > 0:
            release_frame = sorted_frames[-1]
            touch_events.append({
                'frame': int(release_frame),
                'player_id': int(prev_player),
                'event_type': 'release',
                'possession_duration': release_frame - (possession_start or release_frame)
            })

        return touch_events

    def detect_passes(self, possession_events: List[Dict], field_coords_players: Dict,
                     team_assignments: Dict, fps: float = 30, ball_field_coords: Dict = None,
                     ball_assignments: Dict = None, id_mapping: Dict = None) -> List[Dict]:
        """
        Detect passes with robust fallback validation.
        """
        print("      [ROBUST DETECTOR] Starting pass detection with fallback validation...")

        if ball_assignments is None:
            print("      [WARNING] No ball assignments - cannot detect passes")
            return []

        # Check overall ball coordinate quality
        if ball_field_coords:
            total_ball_frames = len(ball_field_coords)
            valid_ball_frames = sum(1 for coords in ball_field_coords.values()
                                   if coords is not None and len(coords) == 2
                                   and (coords[0] != 0.0 or coords[1] != 0.0))
            ball_quality = valid_ball_frames / total_ball_frames if total_ball_frames > 0 else 0.0
            print(f"      [BALL COORDS] Quality: {ball_quality:.1%} ({valid_ball_frames}/{total_ball_frames} frames valid)")

            if ball_quality < 0.3:
                print(f"      [FALLBACK MODE] Low ball coordinate quality - using player positions")
        else:
            print(f"      [FALLBACK MODE] No ball coordinates - using player positions only")

        all_passes = []

        touch_events = self._extract_touch_events_improved(ball_assignments)
        print(f"      [TOUCH EVENTS] Extracted {len(touch_events)} events")

        release_events = [e for e in touch_events if e['event_type'] == 'release']
        receive_events = [e for e in touch_events if e['event_type'] == 'receive']

        print(f"        {len(release_events)} releases, {len(receive_events)} receives")

        for i in range(len(release_events)):
            passer_event = release_events[i]

            if i + 1 < len(receive_events):
                receiver_event = receive_events[i + 1]
            else:
                continue

            passer_id = passer_event['player_id']
            receiver_id = receiver_event['player_id']
            passer_frame = passer_event['frame']
            receiver_frame = receiver_event['frame']

            if passer_id == receiver_id:
                continue

            gap_frames = receiver_frame - passer_frame

            if gap_frames > self.max_gap_frames:
                continue

            # Get trajectory (may be None or unreliable)
            trajectory = None
            if ball_field_coords is not None:
                trajectory = self._get_ball_trajectory(
                    ball_field_coords, passer_frame, receiver_frame
                )

            # Get player positions
            passer_pos = self._get_player_position(passer_id, passer_frame, field_coords_players)
            receiver_pos = self._get_player_position(receiver_id, receiver_frame, field_coords_players)

            if passer_pos is None or receiver_pos is None:
                continue

            # Validate (with fallback if needed)
            is_valid, confidence, reason = self._validate_pass_with_trajectory(
                passer_pos, receiver_pos, trajectory, gap_frames, fps
            )

            if not is_valid:
                print(f"        [REJECTED] P{passer_id}→P{receiver_id}: {reason}")
                continue

            # Team validation
            passer_team = self._get_player_team_robust(passer_id, passer_frame, team_assignments)
            receiver_team = self._get_player_team_robust(receiver_id, receiver_frame, team_assignments)

            if passer_team is None or receiver_team is None:
                confidence *= 0.7
            elif passer_team != receiver_team:
                print(f"        [REJECTED] P{passer_id}({passer_team})→P{receiver_id}({receiver_team}): Different teams")
                continue

            pass_distance = self._calculate_distance(passer_pos, receiver_pos)

            # Apply ID mapping if available to convert ByteTrack IDs to fixed IDs
            if id_mapping is not None:
                passer_id = id_mapping.get(passer_id, passer_id)
                receiver_id = id_mapping.get(receiver_id, receiver_id)

            pass_event = {
                'frame': int(passer_frame),
                'timestamp': round(passer_frame / fps, 2),
                'from_player': int(passer_id),
                'to_player': int(receiver_id),
                'team': int(passer_team) if passer_team is not None else None,
                'distance_m': round(pass_distance, 2),
                'start_position': [round(float(passer_pos[0]), 2), round(float(passer_pos[1]), 2)],
                'end_position': [round(float(receiver_pos[0]), 2), round(float(receiver_pos[1]), 2)],
                'ball_travel_time_s': round(gap_frames / fps, 2),
                'gap_frames': int(gap_frames),
                'confidence_score': round(confidence, 3),
                'validation_reason': reason,
                'detection_method': 'trajectory' if trajectory else 'fallback'
            }

            if trajectory is not None:
                pass_event['ball_distance_m'] = round(trajectory['direct_distance'], 2)
                pass_event['ball_velocity_ms'] = round(trajectory['avg_velocity'], 2)
                pass_event['trajectory_efficiency'] = round(trajectory['efficiency'], 3)

            all_passes.append(pass_event)
            print(f"        [PASS] P{passer_id}→P{receiver_id}: {pass_distance:.1f}m, conf={confidence:.2f}, {reason}")

        print(f"      [ROBUST DETECTOR] Detected {len(all_passes)} passes")

        # Report detection method breakdown
        trajectory_passes = sum(1 for p in all_passes if p['detection_method'] == 'trajectory')
        fallback_passes = sum(1 for p in all_passes if p['detection_method'] == 'fallback')
        print(f"        {trajectory_passes} with trajectory, {fallback_passes} with fallback")

        return all_passes

    def _get_player_position(self, player_id: int, frame_idx: int,
                            field_coords_players: Dict) -> Optional[np.ndarray]:
        """Get player position with nearby frame search."""
        if player_id in field_coords_players:
            player_data = field_coords_players[player_id]
            if 'field_coordinates' in player_data:
                coords = player_data['field_coordinates'].get(frame_idx)
                if coords is not None:
                    return np.array(coords)

                for offset in range(-10, 11):
                    test_frame = frame_idx + offset
                    coords = player_data['field_coordinates'].get(test_frame)
                    if coords is not None:
                        return np.array(coords)

        return None

    def count_player_passes(self, passes: List[Dict]) -> Dict[int, Dict[str, int]]:
        """Count passes per player."""
        player_pass_counts = {}

        for pass_event in passes:
            from_player = pass_event['from_player']
            to_player = pass_event['to_player']

            if from_player not in player_pass_counts:
                player_pass_counts[from_player] = {'passes_completed': 0, 'passes_received': 0}
            if to_player not in player_pass_counts:
                player_pass_counts[to_player] = {'passes_completed': 0, 'passes_received': 0}

            player_pass_counts[from_player]['passes_completed'] += 1
            player_pass_counts[to_player]['passes_received'] += 1

        return player_pass_counts