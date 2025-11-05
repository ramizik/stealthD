"""Camera Calibration Utilities - SoccerNet Best Practices.

This module implements advanced calibration techniques from the SoccerNet calibration
challenge, including normalization transforms and PnP refinement for improved accuracy
and numerical stability.

Based on: https://github.com/SoccerNet/sn-calibration
Methods from top-performing teams (87% accuracy vs 13% baseline)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))


def normalization_transform(points: np.ndarray) -> np.ndarray:
    """
    Compute similarity transform to normalize point cloud for numerical stability.

    Centers points at origin (0,0) and normalizes scale so average distance
    to center is sqrt(2). This pre-conditioning significantly improves
    homography estimation accuracy and prevents numerical instability.

    Based on SoccerNet baseline implementation:
    https://github.com/SoccerNet/sn-calibration/blob/main/src/baseline_cameras.py

    Args:
        points: Point cloud to normalize, shape (N, 2) or (N, 3)

    Returns:
        3x3 similarity transformation matrix

    Example:
        >>> src_pts = np.array([[100, 200], [300, 400], [500, 600]])
        >>> T1 = normalization_transform(src_pts)
        >>> normalized_pts = (T1 @ np.c_[src_pts, np.ones(len(src_pts))].T).T
    """
    # Compute centroid
    center = np.mean(points[:, :2], axis=0)

    # Compute average distance from centroid
    d = 0.0
    nelems = 0
    for p in points:
        nelems += 1
        x = p[0] - center[0]
        y = p[1] - center[1]
        di = np.sqrt(x ** 2 + y ** 2)
        d += (di - d) / nelems  # Running average

    # Compute scale factor (target: sqrt(2))
    if d <= 0.:
        s = 1.0
    else:
        s = np.sqrt(2) / d

    # Build similarity transformation matrix
    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = s
    T[0, 2] = -s * center[0]
    T[1, 1] = s
    T[1, 2] = -s * center[1]
    T[2, 2] = 1

    return T


def estimate_homography_normalized(src_points: np.ndarray,
                                   dst_points: np.ndarray,
                                   method: int = cv2.RANSAC,
                                   ransac_threshold: float = 5.0) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Estimate homography with normalization pre-conditioning for stability.

    Applies normalization transform to both point sets before computing
    homography, then denormalizes the result. This dramatically improves
    numerical stability, especially for points far from origin.

    Args:
        src_points: Source points, shape (N, 2)
        dst_points: Destination points, shape (N, 2)
        method: OpenCV homography method (RANSAC, LMEDS, or 0 for all points)
        ransac_threshold: RANSAC reprojection threshold in pixels

    Returns:
        Tuple of (homography_matrix, inlier_mask)
        Returns (None, empty_mask) if estimation fails
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        return None, np.array([])

    # Normalize both point sets
    T_src = normalization_transform(src_points)
    T_dst = normalization_transform(dst_points)

    # Apply normalization
    src_hom = np.c_[src_points, np.ones(len(src_points))]
    dst_hom = np.c_[dst_points, np.ones(len(dst_points))]

    src_normalized = (T_src @ src_hom.T).T
    dst_normalized = (T_dst @ dst_hom.T).T

    # Convert back to 2D
    src_normalized = src_normalized[:, :2] / src_normalized[:, [2]]
    dst_normalized = dst_normalized[:, :2] / dst_normalized[:, [2]]

    # Estimate homography on normalized points
    H_normalized, mask = cv2.findHomography(
        src_normalized.astype(np.float32),
        dst_normalized.astype(np.float32),
        method=method,
        ransacReprojThreshold=ransac_threshold
    )

    if H_normalized is None:
        return None, np.array([])

    # Denormalize: H = T_dst^-1 @ H_normalized @ T_src
    H = np.linalg.inv(T_dst) @ H_normalized @ T_src

    # Normalize so H[2,2] = 1
    H = H / H[2, 2]

    return H, mask if mask is not None else np.array([])


def refine_homography_pnp(initial_homography: np.ndarray,
                          src_points_2d: np.ndarray,
                          dst_points_3d: np.ndarray,
                          image_size: Tuple[int, int] = (1920, 1080),
                          max_iterations: int = 20000) -> Optional[np.ndarray]:
    """
    Refine homography using PnP (Perspective-n-Point) optimization.

    Decomposes homography into camera parameters, then refines using
    non-linear least squares optimization. This significantly improves
    accuracy for mapping image points to 3D pitch coordinates.

    Based on SoccerNet Camera class refine_camera() method:
    https://github.com/SoccerNet/sn-calibration/blob/main/src/camera.py

    Args:
        initial_homography: Initial homography matrix (3x3)
        src_points_2d: Source image points (N, 2)
        dst_points_3d: Destination 3D points on pitch (N, 2) or (N, 3)
                       If 2D, Z=0 is assumed (planar assumption)
        image_size: (width, height) for principal point initialization
        max_iterations: Maximum optimization iterations (default: 20000)

    Returns:
        Refined homography matrix or None if refinement fails
    """
    if initial_homography is None or len(src_points_2d) < 4:
        return None

    # Ensure dst_points are 3D (add Z=0 if planar)
    if dst_points_3d.shape[1] == 2:
        dst_points_3d = np.c_[dst_points_3d, np.zeros(len(dst_points_3d))]

    # Initialize camera calibration matrix (principal point at center)
    fx = fy = image_size[0]  # Initial guess for focal length
    cx = image_size[0] / 2.0
    cy = image_size[1] / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    try:
        # Decompose homography to get initial rotation and translation
        # H = K @ [r1 r2 t] where r1, r2 are first two columns of rotation
        H_normalized = np.linalg.inv(K) @ initial_homography

        # Normalize columns
        lambda1 = 1.0 / np.linalg.norm(H_normalized[:, 0])
        lambda2 = 1.0 / np.linalg.norm(H_normalized[:, 1])

        r1 = H_normalized[:, 0] * lambda1
        r2 = H_normalized[:, 1] * lambda2
        r3 = np.cross(r1, r2)

        # Form rotation matrix and ensure it's proper (det=1)
        R = np.column_stack((r1, r2, r3))
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, 2] *= -1
            R = U @ Vt

        # Get translation
        lambda3 = np.sqrt(lambda1 * lambda2)
        t = H_normalized[:, 2] * lambda3

        # Convert rotation to rodrigues vector
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)

        # Refine using PnP with Levenberg-Marquardt
        # First use RANSAC to find inliers
        rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
            dst_points_3d.astype(np.float32),
            src_points_2d.astype(np.float32),
            K,
            None,  # No distortion
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=1000,
            reprojectionError=5.0,
            confidence=0.99
        )

        if rvec_ransac is None or inliers is None or len(inliers) < 4:
            # Refinement failed, return original
            return initial_homography

        # Refine with Levenberg-Marquardt on inliers only
        inlier_indices = inliers.flatten()
        rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
            dst_points_3d[inlier_indices].astype(np.float32),
            src_points_2d[inlier_indices].astype(np.float32),
            K,
            None,  # No distortion
            rvec_ransac,
            tvec_ransac,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iterations, 1e-6)
        )

        # Convert back to homography
        R_refined, _ = cv2.Rodrigues(rvec_refined)

        # Build homography: H = K @ [r1 r2 t]
        H_refined = K @ np.column_stack((R_refined[:, 0], R_refined[:, 1], tvec_refined.flatten()))
        H_refined = H_refined / H_refined[2, 2]

        return H_refined

    except (cv2.error, np.linalg.LinAlgError, ValueError) as e:
        # Refinement failed, return original homography
        return initial_homography


def validate_homography(H: np.ndarray,
                       max_condition_number: float = 1e6,
                       max_element_value: float = 1e4) -> bool:
    """
    Validate homography matrix for numerical stability.

    Checks for:
    - Non-null matrix
    - Finite values (no NaN or Inf)
    - Reasonable condition number (matrix not nearly singular)
    - Bounded element values

    Args:
        H: Homography matrix (3x3)
        max_condition_number: Maximum acceptable condition number
        max_element_value: Maximum acceptable absolute element value

    Returns:
        True if homography is valid and numerically stable
    """
    if H is None:
        return False

    # Check for NaN or Inf
    if not np.all(np.isfinite(H)):
        return False

    # Check element magnitude
    if np.max(np.abs(H)) > max_element_value:
        return False

    # Check condition number (matrix invertibility)
    try:
        cond = np.linalg.cond(H)
        if cond > max_condition_number:
            return False
    except np.linalg.LinAlgError:
        return False

    # Check determinant (should not be zero)
    det = np.linalg.det(H)
    if abs(det) < 1e-10:
        return False

    return True


def decompose_homography_to_camera(H: np.ndarray,
                                   image_size: Tuple[int, int] = (1920, 1080)
                                   ) -> Optional[dict]:
    """
    Decompose homography into camera parameters (pan, tilt, roll, position).

    Extracts full camera calibration from homography using the method from
    Multiple View Geometry (Hartley & Zisserman).

    Based on SoccerNet Camera class from_homography() method.

    Args:
        H: Homography matrix (3x3)
        image_size: (width, height) for principal point

    Returns:
        Dictionary with camera parameters or None if decomposition fails:
        {
            'rotation': 3x3 rotation matrix,
            'position': 3D position vector,
            'focal_length': (fx, fy) tuple,
            'principal_point': (cx, cy) tuple
        }
    """
    if H is None or not validate_homography(H):
        return None

    try:
        # Initialize calibration matrix
        cx = image_size[0] / 2.0
        cy = image_size[1] / 2.0

        # Extract calibration from homography using Zhang's method
        # Simplified version - assumes square pixels and principal point at center
        h1 = H[:, 0]
        h2 = H[:, 1]

        # Estimate focal length from homography constraints
        # ||K^-1 @ h1|| = ||K^-1 @ h2|| = 1
        # This gives us constraints on focal length
        h1_h2_dot = np.dot(h1, h2)
        h1_norm = np.linalg.norm(h1)
        h2_norm = np.linalg.norm(h2)

        # Average focal length estimate
        fx = fy = (h1_norm + h2_norm) / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Decompose to rotation and translation
        H_normalized = np.linalg.inv(K) @ H

        lambda1 = 1.0 / np.linalg.norm(H_normalized[:, 0])
        lambda2 = 1.0 / np.linalg.norm(H_normalized[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r1 = H_normalized[:, 0] * lambda1
        r2 = H_normalized[:, 1] * lambda2
        r3 = np.cross(r1, r2)

        R = np.column_stack((r1, r2, r3))
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, 2] *= -1
            R = U @ Vt

        t = H_normalized[:, 2] * lambda3
        position = -R.T @ t

        return {
            'rotation': R,
            'position': position,
            'focal_length': (fx, fy),
            'principal_point': (cx, cy)
        }

    except (np.linalg.LinAlgError, ValueError):
        return None
