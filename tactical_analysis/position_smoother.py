"""Position Smoother using Kalman Filtering for Player Tracking.

Smooths player positions AFTER homography transformation to handle:
- Non-linear perspective transformation discontinuities
- Matrix-level smoothing limitations
- Per-player trajectory coherence

This is the CRITICAL missing piece - matrix smoothing alone cannot prevent
position jumps due to the non-linear nature of cv2.perspectiveTransform.
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np


class SimpleKalmanFilter:
    """
    Improved 2D Kalman filter for position tracking with Mahalanobis outlier detection.

    State: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (position only)

    Optimized for soccer player tracking with realistic motion constraints.
    """

    def __init__(self, dt: float = 0.04, process_noise: float = 0.5,
                 measurement_noise: float = 2.0):
        """
        Initialize Kalman filter with research-backed parameters.

        Args:
            dt: Time step between frames (default: 0.04s for 25fps)
            process_noise: Process noise (higher = trust measurements more)
            measurement_noise: Measurement noise (higher = trust predictions more)
        """
        self.dt = dt
        self.initialized = False

        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],   # x = x + vx*dt
            [0, 1, 0, dt],   # y = y + vy*dt
            [0, 0, 1, 0],    # vx = vx
            [0, 0, 0, 1]     # vy = vy
        ])

        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance - tuned for soccer
        q_pos = 0.5  # Position uncertainty
        q_vel = 2.0  # Velocity uncertainty (higher for rapid acceleration)
        self.Q = np.array([
            [q_pos, 0, 0, 0],
            [0, q_pos, 0, 0],
            [0, 0, q_vel, 0],
            [0, 0, 0, q_vel]
        ])

        # Measurement noise - moderate trust in detections
        r = 0.5  # Adjust based on detection quality
        self.R = np.array([
            [r, 0],
            [0, r]
        ])

        # Initial covariance estimate
        self.P = np.eye(4) * 10.0

    def predict(self) -> np.ndarray:
        """
        Prediction step: Estimate state at current time.

        Returns:
            Predicted position [x, y]
        """
        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:2]  # Return position only

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step: Correct prediction with measurement using Mahalanobis distance.

        Args:
            measurement: Measured position [x, y]

        Returns:
            Corrected position [x, y]
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Mahalanobis distance for outlier detection
        mahalanobis = np.sqrt(y.T @ np.linalg.inv(S) @ y)

        # Adaptive threshold (chi-squared distribution, 95% confidence for 2 DOF)
        threshold = 5.99

        # If outlier, use prediction only
        if mahalanobis > threshold:
            return self.predict()

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.state[:2]

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate [vx, vy] in m/s."""
        return self.state[2:]

    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(4)
        self.P = np.eye(4) * 10.0
        self.initialized = False


class PlayerPositionSmoother:
    """
    Per-player position smoother using Kalman filtering.

    This addresses the ROOT CAUSE of position jumps:
    - Homography matrix smoothing (EMA) reduces matrix discontinuities
    - BUT cv2.perspectiveTransform is NON-LINEAR
    - Small matrix changes can still cause large position jumps (especially at field edges)
    - Solution: Smooth positions AFTER transformation using player-specific filters
    """

    def __init__(self, dt: float = 0.04, max_speed_ms: float = 17.0,
                 process_noise: float = 0.5, measurement_noise: float = 2.0):
        """
        Initialize position smoother.

        Args:
            dt: Time delta between frames (default: 0.04s for 25fps)
            max_speed_ms: Maximum physically plausible speed (m/s) - 17 m/s = 61 km/h
            process_noise: Kalman process noise (0.5 = moderate smoothing)
            measurement_noise: Kalman measurement noise (2.0 = trust measurements)
        """
        self.dt = dt
        self.max_speed_ms = max_speed_ms
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Per-player Kalman filters
        self.player_filters: Dict[int, SimpleKalmanFilter] = {}

        # Track last valid positions for outlier detection
        self.last_positions: Dict[int, np.ndarray] = {}

        # Statistics
        self.total_smoothed = 0
        self.outliers_rejected = 0

        # ADAPTIVE THRESHOLD: Track position stability for each player
        self.position_stability: Dict[int, float] = {}
        # ADAPTIVE THRESHOLD: Track consecutive outliers for each player
        self.consecutive_outliers: Dict[int, int] = {}

    def smooth_position(self, player_id: int, position_raw: np.ndarray,
                       force_update: bool = False, frame_idx: int = None) -> Tuple[np.ndarray, bool]:
        """
        Smooth a player's position using Kalman filtering.

        Args:
            player_id: Unique player/tracker ID
            position_raw: Raw transformed position [x, y] in meters
            force_update: Force update even if outlier detected
            frame_idx: Current frame index (for debug logging)

        Returns:
            Tuple of (smoothed_position, is_outlier)
        """
        # Initialize filter if needed
        if player_id not in self.player_filters:
            self.player_filters[player_id] = SimpleKalmanFilter(
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )

        # Check for outliers (physically impossible jumps)
        is_outlier = False
        distance = 0.0  # Initialize distance to avoid scoping issues

        if player_id in self.last_positions and not force_update:
            distance = np.linalg.norm(position_raw - self.last_positions[player_id])

            # ADAPTIVE THRESHOLD: Adjust based on position stability history
            # If player has been unstable recently, be more lenient
            if player_id in self.position_stability:
                stability_factor = self.position_stability[player_id]
            else:
                stability_factor = 1.0  # Normal threshold initially
                self.position_stability[player_id] = stability_factor

            # Track consecutive outliers for this player
            if player_id not in self.consecutive_outliers:
                self.consecutive_outliers[player_id] = 0

            # Adaptive max distance based on stability
            base_max_distance = self.max_speed_ms * self.dt  # e.g., 17 m/s * 0.04s = 0.68m
            max_distance = base_max_distance * stability_factor

            # If we've had many consecutive outliers, be more lenient
            if self.consecutive_outliers[player_id] > 3:
                max_distance *= 2.0  # Double threshold after 3 consecutive outliers

            # Check for catastrophic jumps (10x normal threshold)
            if distance > max_distance * 10:
                # CATASTROPHIC jump - reject immediately
                is_outlier = True
                self.consecutive_outliers[player_id] += 1
                self.position_stability[player_id] = min(3.0, stability_factor * 1.2)  # Reduce stability
            elif distance > max_distance:
                # Standard outlier - use prediction instead of measurement
                is_outlier = True
                self.consecutive_outliers[player_id] += 1
                self.position_stability[player_id] = min(2.0, stability_factor * 1.1)  # Reduce stability
            else:
                # Good measurement - reset consecutive outlier count
                self.consecutive_outliers[player_id] = 0
                # Gradually improve stability with good measurements
                self.position_stability[player_id] = max(0.5, stability_factor * 0.95)

        # Apply Kalman filter update
        kf = self.player_filters[player_id]
        position_smooth = kf.update(position_raw)

        # Store for next frame comparison
        self.last_positions[player_id] = position_smooth.copy()
        self.total_smoothed += 1

        return position_smooth, is_outlier

    def get_player_velocity(self, player_id: int) -> Optional[np.ndarray]:
        """
        Get player's current velocity estimate.

        Args:
            player_id: Player ID

        Returns:
            Velocity [vx, vy] in m/s or None if not tracked
        """
        if player_id in self.player_filters:
            return self.player_filters[player_id].get_velocity()
        return None

    def reset_player(self, player_id: int):
        """Reset tracking for a specific player (e.g., after occlusion)."""
        if player_id in self.player_filters:
            self.player_filters[player_id].reset()
        if player_id in self.last_positions:
            del self.last_positions[player_id]
        if player_id in self.position_stability:
            del self.position_stability[player_id]
        if player_id in self.consecutive_outliers:
            del self.consecutive_outliers[player_id]

    def get_statistics(self) -> Dict:
        """Get smoother statistics."""
        return {
            'total_smoothed': self.total_smoothed,
            'outliers_rejected': self.outliers_rejected,
            'active_players': len(self.player_filters),
            'outlier_rate': (
                self.outliers_rejected / self.total_smoothed
                if self.total_smoothed > 0 else 0.0
            )
        }

    def print_statistics(self):
        """Print smoother statistics."""
        stats = self.get_statistics()
        print(f"\n[Position Smoother] Statistics:")
        print(f"  Total positions smoothed: {stats['total_smoothed']}")
        print(f"  Outliers rejected: {stats['outliers_rejected']} "
              f"({100*stats['outlier_rate']:.1f}%)")
        print(f"  Active players tracked: {stats['active_players']}")
