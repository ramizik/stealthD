"""
Enhanced Pass Detector with Ball Trajectory Analysis and Robust Team Validation.

Key improvements:
1. Ball trajectory validation (primary signal)
2. Robust coordinate system handling
3. Multi-level team validation with fallbacks
4. Adaptive distance thresholds
5. Improved touch event extraction
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class EnhancedPassDetector:
    """
    Enhanced pass detector with ball trajectory analysis and robust validation.
    """

    def __init__(self,
                 min_pass_distance: float = 0.3,      # Even more permissive for short passes
                 max_pass_distance: float = 70.0,     # Allow longer passes
                 max_gap_frames: int = 150,           # 5 seconds at 30fps for long passes
                 velocity_threshold: float = 0.5,     # Very low threshold
                 min_ball_movement: float = 2.0):     # Ball must move at least 2m
        """
        Initialize enhanced pass detector.

        Args:
            min_pass_distance: Minimum distance between players (meters)
            max_pass_distance: Maximum realistic pass distance (meters)
            max_gap_frames: Maximum frames between possessions
            velocity_threshold: Minimum ball velocity (m/s)
            min_ball_movement: Minimum ball movement for valid pass (meters)
        """
        self.min_pass_distance = min_pass_distance
        self.max_pass_distance = max_pass_distance
        self.max_gap_frames = max_gap_frames
        self.velocity_threshold = velocity_threshold
        self.min_ball_movement = min_ball_movement

    def _normalize_to_meters(self, coord: np.ndarray) -> np.ndarray:
        """
        Robust coordinate normalization to meters.

        Handles multiple coordinate systems:
        - Meters (0-120, 0-80): Return as-is
        - Pitch coordinates (12000x7000): Convert to meters
        - Unknown scale: Detect and convert
        """
        coord = np.asarray(coord, dtype=float)[:2]

        # Already in meters
        if 0 <= coord[0] <= 130 and 0 <= coord[1] <= 90:
            return coord

        # Pitch coordinates (12000x7000) -> meters (105x68)
        if coord[0] > 1000 or coord[1] > 1000:
            return np.array([
                coord[0] * (105.0 / 12000.0),
                coord[1] * (68.0 / 7000.0)
            ])

        # Scaled coordinates (likely 10x or similar)
        if coord[0] > 200 or coord[1] > 200:
            # Estimate scale factor
            scale_x = 105.0 / coord[0] if coord[0] > 0 else 0.1
            scale_y = 68.0 / coord[1] if coord[1] > 0 else 0.1
            scale = (scale_x + scale_y) / 2
            return coord * scale

        return coord

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance with coordinate normalization."""
        if point1 is None or point2 is None:
            return float('inf')

        p1 = self._normalize_to_meters(np.asarray(point1)[:2])
        p2 = self._normalize_to_meters(np.asarray(point2)[:2])

        return float(np.linalg.norm(p1 - p2))

    def _get_ball_trajectory(self, ball_field_coords: Dict,
                            start_frame: int, end_frame: int) -> Optional[Dict]:
        """
        Analyze ball trajectory between two frames.

        Returns:
            Dictionary with trajectory analysis or None if insufficient data
        """
        # Get all ball positions in the frame range
        positions = []
        frames = []

        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in ball_field_coords:
                pos = ball_field_coords[frame_idx]
                if pos is not None and len(pos) == 2:
                    positions.append(self._normalize_to_meters(np.array(pos)))
                    frames.append(frame_idx)

        if len(positions) < 2:
            return None

        positions = np.array(positions)

        # Calculate trajectory metrics
        total_distance = 0
        for i in range(len(positions) - 1):
            total_distance += np.linalg.norm(positions[i+1] - positions[i])

        # Straight-line distance
        direct_distance = np.linalg.norm(positions[-1] - positions[0])

        # Trajectory efficiency (how straight the path is)
        efficiency = direct_distance / total_distance if total_distance > 0 else 0

        # Average velocity
        time_elapsed = (frames[-1] - frames[0]) / 30.0  # Assuming 30 fps
        avg_velocity = total_distance / time_elapsed if time_elapsed > 0 else 0

        return {
            'positions': positions,
            'frames': frames,
            'total_distance': total_distance,
            'direct_distance': direct_distance,
            'efficiency': efficiency,
            'avg_velocity': avg_velocity,
            'num_positions': len(positions)
        }

    def _get_player_team_robust(self, player_id: int, center_frame: int,
                               team_assignments: Dict, window: int = 50) -> Optional[int]:
        """
        Get player's team with robust temporal analysis.

        Args:
            player_id: Player ID
            center_frame: Frame to check around
            team_assignments: Team assignment dictionary
            window: Number of frames to check on each side

        Returns:
            Team ID or None
        """
        team_votes = []

        # Check temporal window
        for offset in range(-window, window + 1):
            frame_idx = center_frame + offset
            if frame_idx in team_assignments and player_id in team_assignments[frame_idx]:
                team_votes.append(team_assignments[frame_idx][player_id])

        if not team_votes:
            # Fallback: check entire video
            for frame_idx, frame_teams in team_assignments.items():
                if player_id in frame_teams:
                    team_votes.append(frame_teams[player_id])

        if not team_votes:
            return None

        # Use majority vote
        counter = Counter(team_votes)
        most_common_team, count = counter.most_common(1)[0]

        # Require at least 60% agreement
        agreement = count / len(team_votes)
        if agreement >= 0.6:
            return most_common_team

        return None

    def _extract_touch_events_improved(self, ball_assignments: Dict) -> List[Dict]:
        """
        Extract touch events with improved logic for release frames.

        Returns both receive and release events for each possession.
        """
        touch_events = []
        prev_player = None
        possession_start = None

        sorted_frames = sorted(ball_assignments.keys())

        for i, frame_idx in enumerate(sorted_frames):
            current_player = ball_assignments[frame_idx]

            # Detect player change
            if current_player != prev_player:
                # Record RELEASE for previous player
                if prev_player is not None and possession_start is not None:
                    # Use the last frame where prev_player had the ball
                    release_frame = sorted_frames[i - 1] if i > 0 else possession_start
                    touch_events.append({
                        'frame': int(release_frame),
                        'player_id': int(prev_player),
                        'event_type': 'release',
                        'possession_duration': release_frame - possession_start
                    })

                # Record RECEIVE for new player
                if current_player is not None:
                    touch_events.append({
                        'frame': int(frame_idx),
                        'player_id': int(current_player),
                        'event_type': 'receive'
                    })
                    possession_start = frame_idx

                prev_player = current_player

        # Handle final possession
        if prev_player is not None and len(sorted_frames) > 0:
            release_frame = sorted_frames[-1]
            touch_events.append({
                'frame': int(release_frame),
                'player_id': int(prev_player),
                'event_type': 'release',
                'possession_duration': release_frame - (possession_start or release_frame)
            })

        return touch_events

    def _validate_pass_with_trajectory(self, passer_pos: np.ndarray, receiver_pos: np.ndarray,
                                      trajectory: Optional[Dict], gap_frames: int,
                                      fps: float = 30.0) -> Tuple[bool, float, str]:
        """
        Validate pass using ball trajectory analysis.

        Returns:
            (is_valid, confidence, reason)
        """
        # Normalize positions
        passer_pos = self._normalize_to_meters(passer_pos)
        receiver_pos = self._normalize_to_meters(receiver_pos)

        player_distance = np.linalg.norm(receiver_pos - passer_pos)

        # Check player distance bounds
        if player_distance < self.min_pass_distance:
            return False, 0.0, f"too_close_{player_distance:.1f}m"
        if player_distance > self.max_pass_distance:
            return False, 0.0, f"too_far_{player_distance:.1f}m"

        # If no trajectory data, use basic validation
        if trajectory is None or trajectory['num_positions'] < 2:
            # Allow passes with short gaps even without trajectory
            if gap_frames <= 10:  # ~0.33 seconds
                return True, 0.6, "short_gap_no_trajectory"
            return True, 0.4, "no_trajectory_data"

        # Validate with trajectory
        ball_distance = trajectory['direct_distance']
        ball_total_distance = trajectory['total_distance']
        efficiency = trajectory['efficiency']
        velocity = trajectory['avg_velocity']

        confidence = 1.0
        reasons = []

        # Ball must move minimum distance
        if ball_distance < self.min_ball_movement:
            return False, 0.0, f"ball_didnt_move_{ball_distance:.1f}m"

        # Check if ball moved in the right direction
        # Calculate expected ball movement direction
        passer_to_receiver = receiver_pos - passer_pos

        if len(trajectory['positions']) >= 2:
            ball_start = trajectory['positions'][0]
            ball_end = trajectory['positions'][-1]
            ball_movement = ball_end - ball_start

            # Calculate directional similarity (dot product)
            if np.linalg.norm(passer_to_receiver) > 0 and np.linalg.norm(ball_movement) > 0:
                direction_similarity = np.dot(passer_to_receiver, ball_movement) / (
                    np.linalg.norm(passer_to_receiver) * np.linalg.norm(ball_movement)
                )

                # Ball should move generally toward receiver (allow some deviation)
                if direction_similarity < 0.3:  # Less than ~73 degrees
                    confidence *= 0.5
                    reasons.append(f"direction_mismatch_{direction_similarity:.2f}")
                else:
                    reasons.append(f"good_direction_{direction_similarity:.2f}")

        # Check velocity
        if velocity < self.velocity_threshold:
            confidence *= 0.7
            reasons.append(f"low_velocity_{velocity:.1f}ms")
        else:
            reasons.append(f"good_velocity_{velocity:.1f}ms")

        # Check trajectory efficiency (straight passes have higher efficiency)
        if efficiency < 0.5:  # Very curved path
            confidence *= 0.8
            reasons.append(f"curved_path_{efficiency:.2f}")
        else:
            reasons.append(f"straight_path_{efficiency:.2f}")

        # Ball distance should be comparable to player distance
        distance_ratio = ball_distance / player_distance if player_distance > 0 else 0
        if distance_ratio < 0.5 or distance_ratio > 2.0:
            confidence *= 0.7
            reasons.append(f"distance_mismatch_ratio_{distance_ratio:.2f}")
        else:
            reasons.append(f"good_distance_ratio_{distance_ratio:.2f}")

        reason_str = "+".join(reasons)
        return True, confidence, reason_str

    def detect_passes(self, possession_events: List[Dict], field_coords_players: Dict,
                     team_assignments: Dict, fps: float = 30, ball_field_coords: Dict = None,
                     ball_assignments: Dict = None) -> List[Dict]:
        """
        Detect passes with enhanced trajectory validation.

        Primary method: Touch-based detection with trajectory validation
        Secondary method: Possession-based detection for backup
        """
        print("      [ENHANCED DETECTOR] Starting pass detection with trajectory analysis...")

        if ball_assignments is None:
            print("      [WARNING] No ball assignments provided - detection will be limited")
            return []

        all_passes = []

        # Extract improved touch events
        touch_events = self._extract_touch_events_improved(ball_assignments)
        print(f"      [TOUCH EVENTS] Extracted {len(touch_events)} events")

        release_events = [e for e in touch_events if e['event_type'] == 'release']
        receive_events = [e for e in touch_events if e['event_type'] == 'receive']

        print(f"        {len(release_events)} releases, {len(receive_events)} receives")

        # Match release -> receive events
        for i in range(len(release_events)):
            passer_event = release_events[i]

            # Find next receive event
            if i + 1 < len(receive_events):
                receiver_event = receive_events[i + 1]
            else:
                continue

            passer_id = passer_event['player_id']
            receiver_id = receiver_event['player_id']
            passer_frame = passer_event['frame']
            receiver_frame = receiver_event['frame']

            # Skip same player
            if passer_id == receiver_id:
                continue

            gap_frames = receiver_frame - passer_frame

            # Skip if gap too large
            if gap_frames > self.max_gap_frames:
                continue

            # Get ball trajectory
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

            # Validate with trajectory
            is_valid, confidence, reason = self._validate_pass_with_trajectory(
                passer_pos, receiver_pos, trajectory, gap_frames, fps
            )

            if not is_valid:
                print(f"        [REJECTED] P{passer_id}→P{receiver_id}: {reason}")
                continue

            # Team validation (multi-level)
            passer_team = self._get_player_team_robust(passer_id, passer_frame, team_assignments, window=50)
            receiver_team = self._get_player_team_robust(receiver_id, receiver_frame, team_assignments, window=50)

            if passer_team is None or receiver_team is None:
                print(f"        [WARNING] P{passer_id}→P{receiver_id}: Missing team data")
                confidence *= 0.5  # Reduce confidence but don't reject
            elif passer_team != receiver_team:
                print(f"        [REJECTED] P{passer_id}({passer_team})→P{receiver_id}({receiver_team}): Different teams")
                continue

            # Calculate final distance
            pass_distance = self._calculate_distance(passer_pos, receiver_pos)

            # Create pass event
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
                'detection_method': 'trajectory_enhanced'
            }

            # Add trajectory data if available
            if trajectory is not None:
                pass_event['ball_distance_m'] = round(trajectory['direct_distance'], 2)
                pass_event['ball_velocity_ms'] = round(trajectory['avg_velocity'], 2)
                pass_event['trajectory_efficiency'] = round(trajectory['efficiency'], 3)

            all_passes.append(pass_event)
            print(f"        [PASS] P{passer_id}→P{receiver_id}: {pass_distance:.1f}m, conf={confidence:.2f}, {reason}")

        print(f"      [ENHANCED DETECTOR] Detected {len(all_passes)} passes")
        return all_passes

    def _get_player_position(self, player_id: int, frame_idx: int,
                            field_coords_players: Dict) -> Optional[np.ndarray]:
        """Get player position at frame with fallback search."""
        # Try exact frame
        if player_id in field_coords_players:
            player_data = field_coords_players[player_id]
            if 'field_coordinates' in player_data:
                coords = player_data['field_coordinates'].get(frame_idx)
                if coords is not None:
                    return np.array(coords)

                # Search nearby frames (±10)
                for offset in range(-10, 11):
                    test_frame = frame_idx + offset
                    coords = player_data['field_coordinates'].get(test_frame)
                    if coords is not None:
                        return np.array(coords)

        return None

    def count_player_passes(self, passes: List[Dict]) -> Dict[int, Dict[str, int]]:
        """Count passes completed and received by each player."""
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