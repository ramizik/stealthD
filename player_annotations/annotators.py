import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import cv2
import numpy as np
import supervision as sv


class AnnotatorManager:
    """
    Manager class for all annotation functionality.
    Provides a unified interface for annotating players, ball, referees, and keypoints.
    """

    def __init__(self, edges=None):
        """Initialize all annotation tools."""
        # Basic annotators
        self.ellipse_annotator = sv.EllipseAnnotator()
        self.triangle_annotator = sv.TriangleAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()

        # Keypoint annotators
        self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=8)
        self.edge_annotator = sv.EdgeAnnotator(color=sv.Color.BLUE, thickness=3, edges=edges)

        # Goalkeeper logger for separate debug file
        self.gk_logger = None

    def set_goalkeeper_logger(self, gk_logger):
        """
        Set the goalkeeper logger for debug output.

        Args:
            gk_logger: GoalkeeperLogger instance
        """
        self.gk_logger = gk_logger

    def _log_gk(self, message: str):
        """
        Log goalkeeper debug message to separate file if logger is set.

        Args:
            message: Message to log
        """
        if self.gk_logger:
            self.gk_logger.log(message)

    def annotate_players(self, frame: np.ndarray, player_detections: sv.Detections) -> np.ndarray:
        """
        Annotate only players on the frame with goalkeeper indicators.

        Args:
            frame: Input video frame
            player_detections: Player detection results

        Returns:
            Annotated frame with player detections visualized
        """
        if player_detections is None or len(player_detections.xyxy) == 0:
            return frame

        if player_detections.tracker_id is None:
            player_detections.tracker_id = np.arange(len(player_detections.xyxy))

        # Debug: Print all tracker IDs in frame 150 to see what's being annotated
        if not hasattr(self, '_frame_150_debug'):
            import inspect
            frame_obj = inspect.currentframe()
            # Try to get frame index from call stack
            self._debug_frame_count = getattr(self, '_debug_frame_count', 0)
            if self._debug_frame_count == 150:
                self._log_gk(f"[FRAME 150 DEBUG] Total players to annotate: {len(player_detections.tracker_id)}")
                self._log_gk(f"[FRAME 150 DEBUG] Tracker IDs: {sorted(player_detections.tracker_id)}")
                if hasattr(player_detections, 'data') and player_detections.data and 'is_goalkeeper' in player_detections.data:
                    gk_mask = player_detections.data['is_goalkeeper']
                    gk_ids = player_detections.tracker_id[gk_mask]
                    self._log_gk(f"[FRAME 150 DEBUG] Goalkeeper IDs: {gk_ids}")
                self._frame_150_debug = True
            self._debug_frame_count += 1

        # Create labels with GK indicator if available
        player_labels = []
        gk_count_in_frame = 0
        for i, tracker_id in enumerate(player_detections.tracker_id):
            label = f'#{tracker_id}'
            # Check if this player is marked as goalkeeper (using custom data if available)
            is_gk = False
            if hasattr(player_detections, 'data') and player_detections.data is not None:
                if 'is_goalkeeper' in player_detections.data:
                    if player_detections.data['is_goalkeeper'][i]:
                        label = f'GK#{tracker_id}'  # Mark as goalkeeper
                        is_gk = True
                        gk_count_in_frame += 1
            player_labels.append(label)

        # DEBUG: Log goalkeeper annotation every 50 frames
        if not hasattr(self, '_annotation_frame_counter'):
            self._annotation_frame_counter = 0

        if self._annotation_frame_counter % 50 == 0:
            has_gk_data = (hasattr(player_detections, 'data') and
                          player_detections.data is not None and
                          'is_goalkeeper' in player_detections.data)

            if has_gk_data:
                gk_mask = player_detections.data['is_goalkeeper']
                gk_ids = [player_detections.tracker_id[i] for i in range(len(gk_mask)) if gk_mask[i]]
                self._log_gk(f"[ANNOTATION DEBUG] Frame {self._annotation_frame_counter}: Found {len(gk_ids)} GK(s) to annotate: {gk_ids}")
                if len(gk_ids) == 0 and len(player_detections) > 0:
                    self._log_gk(f"[ANNOTATION DEBUG]   WARNING: No GKs in data but {len(player_detections)} players detected")
                    self._log_gk(f"[ANNOTATION DEBUG]   All tracker IDs: {sorted(player_detections.tracker_id)[:10]}...")
            else:
                self._log_gk(f"[ANNOTATION DEBUG] Frame {self._annotation_frame_counter}: No goalkeeper data in detections")

        self._annotation_frame_counter += 1

        frame = self.ellipse_annotator.annotate(frame, player_detections)
        frame = self.label_annotator.annotate(frame, detections=player_detections, labels=player_labels)

        return frame

    def annotate_ball(self, frame: np.ndarray, ball_detections: sv.Detections) -> np.ndarray:
        """
        Annotate ball detections on frame.

        Args:
            frame: Input video frame
            ball_detections: Ball detection results

        Returns:
            Annotated frame with ball detections
        """
        if ball_detections is not None and len(ball_detections.xyxy) > 0:
            return self.triangle_annotator.annotate(frame, ball_detections)
        return frame

    def annotate_referees(self, frame: np.ndarray, referee_detections: sv.Detections) -> np.ndarray:
        """
        Annotate referee detections on frame.

        Args:
            frame: Input video frame
            referee_detections: Referee detection results

        Returns:
            Annotated frame with referee detections
        """
        if referee_detections is not None and len(referee_detections.xyxy) > 0:
            if referee_detections.tracker_id is None:
                referee_detections.tracker_id = np.arange(len(referee_detections.xyxy))
            return self.ellipse_annotator.annotate(frame, referee_detections)
        return frame

    def annotate_all(self, frame: np.ndarray, player_detections, ball_detections, referee_detections) -> np.ndarray:
        """
        Annotate players, ball, and referees on the frame using separate methods.

        Args:
            frame: Input video frame
            player_detections: Player detection results
            ball_detections: Ball detection results
            referee_detections: Referee detection results

        Returns:
            Annotated frame with all detections visualized
        """
        target_frame = frame.copy()

        # Annotate each type separately
        target_frame = self.annotate_players(target_frame, player_detections)
        target_frame = self.annotate_ball(target_frame, ball_detections)
        target_frame = self.annotate_referees(target_frame, referee_detections)

        return target_frame

    def annotate_bboxes(self, frame: np.ndarray, detections: sv.Detections, class_names: dict = None) -> np.ndarray:
        """
        Annotate frame with object detections bboxes.

        Args:
            frame: Input frame
            detections: Supervision Detections object
            class_names: Dictionary mapping class IDs to names

        Returns:
            Annotated frame with detection boxes and labels
        """
        if detections is None or len(detections.xyxy) == 0:
            return frame

        annotated_frame = frame.copy()

        # Annotate with boxes
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)

        # Add labels if class_names provided
        if class_names is not None:
            labels = []
            for class_id, conf in zip(detections.class_id, detections.confidence):
                class_name = class_names.get(class_id, f'Class {class_id}')
                labels.append(f"{class_name} {conf:.2f}")
            annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame

    def annotate_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.5,
                          draw_vertices: bool = True, draw_edges = None, draw_labels: bool = True,
                          KEYPOINT_CONNECTIONS=[], KEYPOINT_NAMES={},
                          KEYPOINT_COLOR=(0, 255, 0), CONNECTION_COLOR=(255, 0, 0), TEXT_COLOR=(255, 255, 255)) -> np.ndarray:
        """
        Annotate frame with detected keypoints using Vertex and Edge annotators.

        Args:
            frame: Input frame to annotate
            keypoints: Detected keypoints array with shape (N, 27, 3)
            confidence_threshold: Minimum confidence to draw keypoint
            draw_vertices: Whether to draw keypoint vertices
            draw_edges: Whether to draw connections between keypoints
            draw_labels: Whether to draw keypoint labels

        Returns:
            Annotated frame
        """

        # Draw keypoints and connections for each detection
        for kpts in keypoints:

            # Draw keypoint connections
            if draw_edges:
                for connection in KEYPOINT_CONNECTIONS:
                    pt1_idx, pt2_idx = connection
                    pt1 = kpts[pt1_idx]
                    pt2 = kpts[pt2_idx]

                    # Only draw if both points are visible
                    if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                        cv2.line(frame,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            CONNECTION_COLOR, 2)

            # Draw keypoints
            for kpt_idx, kpt in enumerate(kpts):
                if kpt[2] > confidence_threshold:  # Check visibility
                    x, y = int(kpt[0]), int(kpt[1])

                    # Draw keypoint circle
                    cv2.circle(frame, (x, y), 5, KEYPOINT_COLOR, -1)

                    # Draw keypoint label
                    if draw_labels:
                        label = f"{kpt_idx}: {KEYPOINT_NAMES.get(kpt_idx, 'Unknown')}"
                        cv2.putText(frame, label, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

        return frame

    def convert_tracks_to_detections(self, player_tracks, ball_tracks, referee_tracks, player_classids=None, player_is_goalkeeper=None):
        """
        Convert tracking data back to supervision detections format.

        Args:
            player_tracks: Player tracking data for a frame
            ball_tracks: Ball tracking data for a frame
            referee_tracks: Referee tracking data for a frame
            player_classids: Player class ID data for a frame (optional)
            player_is_goalkeeper: Player goalkeeper status for a frame (optional)

        Returns:
            Tuple of converted detection objects
        """
        # Get the player detections
        tracker_ids = []  # Initialize for use in goalkeeper metadata restoration
        if player_tracks is not None and isinstance(player_tracks, dict):
            # Extract bboxes from track data (handle both list and dict formats)
            bboxes = []
            for tracker_id, track_data in player_tracks.items():
                if tracker_id == -1:
                    continue
                # Handle both formats: list (bbox) or dict (enhanced format)
                if isinstance(track_data, dict):
                    bbox = track_data.get('bbox', None)
                elif isinstance(track_data, list) and len(track_data) >= 4:
                    bbox = track_data
                else:
                    continue

                if bbox is not None and not any(b is None for b in bbox):
                    bboxes.append(bbox)
                    tracker_ids.append(tracker_id)

            if len(bboxes) > 0:
                if player_classids is not None:
                    # Use stored class IDs
                    class_ids = [player_classids.get(tid, 0) for tid in tracker_ids]
                else:
                    # Fall back to default class ID (0)
                    class_ids = [0] * len(tracker_ids)

                player_detections = sv.Detections(
                    xyxy=np.array(bboxes),
                    class_id=np.array(class_ids),
                    tracker_id=np.array(tracker_ids)
                )
            else:
                player_detections = None
        else:
            player_detections = None

        # Restore goalkeeper metadata if available
        if player_detections is not None and player_is_goalkeeper is not None and len(tracker_ids) > 0:
            # Use .get() with default False for tracker IDs not in goalkeeper dict
            goalkeeper_mask = np.array([player_is_goalkeeper.get(tracker_id, False) for tracker_id in tracker_ids])
            player_detections.data = {'is_goalkeeper': goalkeeper_mask}

            # DEBUG: Log goalkeeper data restoration
            if not hasattr(self, '_gk_restore_debug_counter'):
                self._gk_restore_debug_counter = 0

            if self._gk_restore_debug_counter % 100 == 0:
                gk_count = np.sum(goalkeeper_mask) if player_detections is not None else 0
                if gk_count > 0:
                    gk_ids = [tracker_ids[i] for i in range(len(goalkeeper_mask)) if goalkeeper_mask[i]]
                    self._log_gk(f"[RESTORE DEBUG] Frame ~{self._gk_restore_debug_counter}: Restored {gk_count} goalkeeper(s): {gk_ids}")
                    self._log_gk(f"[RESTORE DEBUG]   player_is_goalkeeper dict: {player_is_goalkeeper}")
                else:
                    self._log_gk(f"[RESTORE DEBUG] Frame ~{self._gk_restore_debug_counter}: No goalkeepers in this frame")
                    self._log_gk(f"[RESTORE DEBUG]   player_is_goalkeeper dict: {player_is_goalkeeper}")
                    if player_tracks is not None:
                        self._log_gk(f"[RESTORE DEBUG]   tracker IDs in frame: {list(player_tracks.keys())}")

            self._gk_restore_debug_counter += 1
        else:
            # DEBUG: Log when no goalkeeper data is provided
            if player_detections is not None and not hasattr(self, '_no_gk_data_logged'):
                print(f"[RESTORE DEBUG] WARNING: No goalkeeper metadata provided for frame")
                self._no_gk_data_logged = True

        # Get the ball detections
        if ball_tracks is not None:
            ball_detections = sv.Detections(
                xyxy=np.array([ball_tracks]),
                class_id=np.array([2]),
            )
        else:
            ball_detections = None

        # Get the referee detections
        if referee_tracks is not None and isinstance(referee_tracks, dict):
            # Extract bboxes from track data (handle both list and dict formats)
            bboxes = []
            referee_ids = []
            for ref_id, track_data in referee_tracks.items():
                if ref_id == -1:
                    continue
                # Handle both formats: list (bbox) or dict (enhanced format)
                if isinstance(track_data, dict):
                    bbox = track_data.get('bbox', None)
                elif isinstance(track_data, list) and len(track_data) >= 4:
                    bbox = track_data
                else:
                    continue

                if bbox is not None and not any(b is None for b in bbox):
                    bboxes.append(bbox)
                    referee_ids.append(ref_id)

            if len(bboxes) > 0:
                referee_detections = sv.Detections(
                    xyxy=np.array(bboxes),
                    class_id=np.array([3] * len(referee_ids)),
                    tracker_id=np.array(referee_ids)
                )
            else:
                referee_detections = None
        else:
            referee_detections = None

        return player_detections, ball_detections, referee_detections