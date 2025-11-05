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
    Simple 2D Kalman filter for position tracking.

    State: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (position only)

    Optimized for soccer player tracking with realistic motion constraints.
    """

    def __init__(self, dt: float = 0.04, process_noise: float = 0.5,
                 measurement_noise: float = 2.0):
        """
        Initialize Kalman filter.

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

        # Process noise covariance
        q = process_noise
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q, 0],
            [0, 0, 0, q]
        ])

        # Measurement noise covariance
        r = measurement_noise
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
        Update step: Correct prediction with measurement.

        Args:
            measurement: Measured position [x, y]

        Returns:
            Corrected position [x, y]
        """
        if not self.initialized:
            # First measurement - initialize state
            self.state[:2] = measurement
            self.state[2:] = 0  # Zero velocity
            self.initialized = True
            return measurement

        # Predict
        self.predict()

        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state with measurement
        y = measurement - self.H @ self.state  # Innovation
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.state[:2]  # Return smoothed position

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
        if player_id in self.last_positions and not force_update:
            distance = np.linalg.norm(position_raw - self.last_positions[player_id])
            max_distance = self.max_speed_ms * self.dt  # e.g., 17 m/s * 0.04s = 0.68m

            if distance > max_distance:
                # Outlier detected - use prediction instead of measurement
                is_outlier = True
                self.outliers_rejected += 1

                # Use Kalman prediction (don't update with bad measurement)
                kf = self.player_filters[player_id]
                position_smooth = kf.predict()

                # DEBUG: Enhanced logging for first 20 outliers
                if self.outliers_rejected <= 20:
                    implied_speed_kmh = (distance / self.dt) * 3.6
                    implied_speed_ms = distance / self.dt
                    frame_str = f"Frame {frame_idx}: " if frame_idx is not None else ""
                    print(f"[Position Smoother DEBUG] {frame_str}Player {player_id} Outlier #{self.outliers_rejected}")
                    print(f"  Raw position: ({position_raw[0]:.2f}, {position_raw[1]:.2f})m")
                    print(f"  Last position: ({self.last_positions[player_id][0]:.2f}, {self.last_positions[player_id][1]:.2f})m")
                    print(f"  Predicted position: ({position_smooth[0]:.2f}, {position_smooth[1]:.2f})m")
                    print(f"  Distance: {distance:.2f}m | Speed: {implied_speed_ms:.2f} m/s ({implied_speed_kmh:.1f} km/h)")
                    print(f"  Threshold: {max_distance:.2f}m | Max speed: {self.max_speed_ms:.2f} m/s")

                # Don't update last_positions with outlier - keep using prediction
                return position_smooth, is_outlier

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
