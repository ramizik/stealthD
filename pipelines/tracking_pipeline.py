import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import gc
import os
import pickle
import time

import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

from analytics import BallInterpolator
from constants import (ASPECT_RATIO_THRESH, MIN_BOX_AREA, TRACKER_BUFFER_SIZE,
                       TRACKER_MATCH_THRESH, TRAINING_FRAME_MAX,
                       TRAINING_FRAME_MIN, TRAINING_FRAME_PERCENTAGE,
                       TRAINING_FRAME_STRIDE)
from pipelines.detection_pipeline import DetectionPipeline
from pipelines.processing_pipeline import ProcessingPipeline
from player_annotations import AnnotatorManager
from player_clustering import ClusteringManager, GoalkeeperAssigner
from player_tracking import TrackerManager


class TrackingPipeline:
    """
    Complete tracking pipeline that combines detection, tracking, clustering and annotation.

    This pipeline provides a comprehensive solution for soccer video analysis by:
    - Object detection (players, ball, referees) using YOLO
    - Multi-object tracking using ByteTrack
    - Team assignment using SigLIP embeddings + UMAP + K-means clustering
    - Video annotation and output generation
    - Track extraction and ball interpolation support

    The pipeline is designed to work with the complete soccer analysis system
    and can be used standalone or integrated with other pipelines.
    """

    def __init__(self, model_path):
        """
        Initialize the tracking pipeline with all necessary models.

        Args:
            model_path: Path to the YOLO detection model
        """
        self.detection_pipeline = DetectionPipeline(model_path)
        self.processing_pipeline = ProcessingPipeline()
        self.tracker_manager = None
        self.clustering_manager = None
        self.goalkeeper_assigner = None
        self.annotator_manager = None
        self.ball_interpolator = BallInterpolator()
        # Store goalkeeper tracker IDs detected across all frames
        self.goalkeeper_tracker_ids = set()
        # Track when we last ran goalkeeper detection
        self.last_gk_detection_frame = -1000  # Start with large negative to trigger first detection
        # Store video frame dimensions
        self.frame_width = None
        self.frame_height = None

    def initialize_models(self):
        """Initialize all models required for the pipeline."""
        print("Initializing tracking pipeline models...")
        model_init_time = time.time()

        # Initialize detection pipeline
        print("Initializing detection pipeline...")
        self.detection_pipeline.initialize_model()

        # Initialize tracker
        print("Initializing tracker with SoccerNet-style filtering...")
        self.tracker_manager = TrackerManager(
            match_thresh=TRACKER_MATCH_THRESH,
            track_buffer=TRACKER_BUFFER_SIZE,
            min_box_area=MIN_BOX_AREA,
            aspect_ratio_thresh=ASPECT_RATIO_THRESH
        )

        # Initialize clustering manager
        print("Initializing clustering manager...")
        self.clustering_manager = ClusteringManager()

        # Initialize goalkeeper assigner
        print("Initializing goalkeeper assigner...")
        self.goalkeeper_assigner = GoalkeeperAssigner()

        # Initialize annotator manager
        print("Initializing annotators...")
        self.annotator_manager = AnnotatorManager()

        model_init_time = time.time() - model_init_time
        print(f"Model initialization completed in {model_init_time:.2f}s")

    def collect_training_crops(self, video_path):
        """
        Collect player crops from video for training clustering models.

        Args:
            video_path: Path to training video

        Returns:
            List of player crop images
        """
        print("Collecting player crops for training...")

        # Get video info to avoid requesting frames beyond video length
        video_info = sv.VideoInfo.from_video_path(video_path)

        # Calculate dynamic frame limit based on video length (20% of total frames)
        dynamic_frame_limit = int(video_info.total_frames * TRAINING_FRAME_PERCENTAGE)

        # Apply min/max constraints
        dynamic_frame_limit = max(TRAINING_FRAME_MIN, min(TRAINING_FRAME_MAX, dynamic_frame_limit))

        # Ensure we don't exceed total frames
        max_end_frame = min(dynamic_frame_limit, video_info.total_frames)

        # Calculate expected number of frames to process (for progress bar)
        expected_frames = max_end_frame // TRAINING_FRAME_STRIDE

        print(f"Video has {video_info.total_frames} frames, collecting crops from first {max_end_frame} frames (stride={TRAINING_FRAME_STRIDE})")
        print(f"  Expected to process ~{expected_frames} frames ({TRAINING_FRAME_PERCENTAGE*100:.0f}% of video)")

        # Get video frames
        frame_generator = sv.get_video_frames_generator(video_path, stride=TRAINING_FRAME_STRIDE, end=max_end_frame)

        # Extract player crops with improved progress bar
        crops = []
        frame_count = 0
        for frame in tqdm(frame_generator, desc='collecting_crops', total=expected_frames,
                         mininterval=2.0, ncols=100,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'):
            player_detections, _, _ = self.detection_pipeline.detect_frame_objects(frame)
            cropped_images = self.clustering_manager.embedding_extractor.get_player_crops(frame, player_detections)
            crops += cropped_images

            frame_count += 1
            # Periodic GPU memory cleanup every 50 frames
            if frame_count % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"Collected {len(crops)} player crops")

        # Validation: Check if we have sufficient crops for training
        min_crops_required = 100  # Minimum crops needed for reliable training
        if len(crops) < min_crops_required:
            print(f"  ⚠️ WARNING: Only {len(crops)} player crops collected (recommended: >{min_crops_required}).")
            print(f"  This may result in poor team assignment. Consider:")
            print(f"    - Increasing TRAINING_FRAME_PERCENTAGE (current: {TRAINING_FRAME_PERCENTAGE*100:.0f}%)")
            print(f"    - Decreasing TRAINING_FRAME_STRIDE (current: {TRAINING_FRAME_STRIDE})")
        else:
            print(f"  ✓ Sufficient training data collected ({len(crops)} crops)")

        # Final cleanup after crop collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return crops

    def train_team_assignment_models(self, video_path):
        """
        Train UMAP and K-means models for team assignment.
        Uses cache if available to speed up repeated runs on same video.

        Args:
            video_path: Path to training video

        Returns:
            Trained clustering models
        """
        # Check for cached models
        video_path_obj = Path(video_path)
        cache_dir = video_path_obj.parent / "cache"
        cache_dir.mkdir(exist_ok=True)  # Create cache directory if it doesn't exist
        cache_path = cache_dir / f"{video_path_obj.stem}_team_models_cache.pkl"

        # Get video info for cache validation
        video_info = sv.VideoInfo.from_video_path(video_path)

        if os.path.exists(cache_path):
            print("  Checking cached team assignment models...")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Validate cache matches current video
                if (cached_data.get('total_frames') == video_info.total_frames and
                    cached_data.get('video_width') == video_info.width and
                    cached_data.get('video_height') == video_info.height):

                    # Load cached models
                    reducer = cached_data['reducer']
                    cluster_model = cached_data['cluster_model']

                    # Update clustering manager with cached models
                    self.clustering_manager.reducer = reducer
                    self.clustering_manager.cluster_model = cluster_model
                    self.clustering_manager.is_trained = True

                    print(f"  ✓ Cache loaded successfully - skipping team assignment training")
                    return cached_data.get('cluster_labels'), reducer, cluster_model
                else:
                    print(f"  Cache mismatch - reprocessing...")
            except Exception as e:
                print(f"  Error loading cache: {e} - reprocessing...")

        # Train models (cache miss or invalid cache)
        print("Training team assignment models...")
        training_time = time.time()

        # Collect training crops
        crops = self.collect_training_crops(video_path)

        # Train clustering models
        cluster_labels, reducer, cluster_model = self.clustering_manager.train_clustering_models(crops)

        # Clear crops from memory after training
        del crops
        gc.collect()

        training_time = time.time() - training_time
        print(f"Team assignment training completed in {training_time:.2f}s")

        # Save cache
        print(f"  Saving team assignment models cache to: {cache_path.name}")
        try:
            cached_data = {
                'reducer': reducer,
                'cluster_model': cluster_model,
                'cluster_labels': cluster_labels,
                'total_frames': video_info.total_frames,
                'video_width': video_info.width,
                'video_height': video_info.height
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"  Cache saved successfully")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

        return cluster_labels, reducer, cluster_model

    def detection_callback(self, frame):
        """
        Detection callback for processing individual frames.

        Args:
            frame: Input video frame

        Returns:
            Tuple of detection results (player, ball, referee)
        """
        detection_time = time.time()
        player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
        detection_time = time.time() - detection_time

        return player_detections, ball_detections, referee_detections, detection_time

    def tracking_callback(self, player_detections):
        """
        Tracking callback for updating player tracks.

        Args:
            player_detections: Player detection results

        Returns:
            Updated player detections with tracking information
        """
        return self.tracker_manager.process_tracking_for_frame(player_detections)

    def clustering_callback(self, frame, player_detections, frame_idx=0):
        """
        Clustering callback for team assignment with goalkeeper handling.

        This method:
        1. Periodically detects goalkeepers based on spatial isolation (every 100 frames)
        2. Accumulates goalkeeper tracker IDs across the video
        3. Assigns teams to field players using color clustering
        4. Assigns teams to goalkeepers based on proximity to team centroids
        5. Marks goalkeepers with metadata for visualization

        Strategy for goalkeeper detection:
        - Run detection every 100 frames to catch goalkeepers that appear later
        - Once a tracker ID is identified as goalkeeper, it stays a goalkeeper
        - This handles: GK not in frame 0, GK appearing/disappearing, camera movement

        Args:
            frame: Input video frame
            player_detections: Player detection results
            frame_idx: Frame index

        Returns:
            Updated player detections with team assignments and GK markers
        """
        assignment_time = time.time()

        # Capture frame dimensions on first frame
        if self.frame_width is None and frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
            print(f"\n[FRAME INFO] Video dimensions: {self.frame_width}x{self.frame_height}")

        if len(player_detections.xyxy) > 0:
            # Periodically run goalkeeper detection (every 25 frames = ~1 second)
            # More frequent checks to catch goalkeepers appearing at any time
            GK_DETECTION_INTERVAL = 25
            should_detect_gk = (frame_idx - self.last_gk_detection_frame) >= GK_DETECTION_INTERVAL

            if should_detect_gk and player_detections.tracker_id is not None and len(player_detections) >= 3:
                # Run spatial isolation detection with actual frame width
                goalkeeper_mask, field_player_mask = self._identify_goalkeepers(
                    player_detections,
                    frame_width_px=self.frame_width or 1920
                )

                # Add any newly detected goalkeepers to our set
                goalkeeper_indices = np.where(goalkeeper_mask)[0]
                new_goalkeepers = []
                for idx in goalkeeper_indices:
                    tracker_id = player_detections.tracker_id[idx]
                    if tracker_id not in self.goalkeeper_tracker_ids:
                        new_goalkeepers.append(tracker_id)
                    self.goalkeeper_tracker_ids.add(tracker_id)

                # Log detection attempts for visibility
                positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                x_positions = positions[:, 0]

                if new_goalkeepers:
                    print(f"\n[GK DETECTION] Frame {frame_idx}: Detected {len(new_goalkeepers)} new goalkeeper(s)")
                    print(f"[GK DETECTION] New ByteTrack IDs: {sorted(new_goalkeepers)}")

                    # Show where each GK was detected with isolation info
                    for idx in goalkeeper_indices:
                        if player_detections.tracker_id[idx] in new_goalkeepers:
                            x_pos = positions[idx, 0]
                            y_pos = positions[idx, 1]

                            # Calculate isolation distance
                            other_positions = np.delete(x_positions, idx)
                            if len(other_positions) > 0:
                                distance_to_nearest = np.min(np.abs(other_positions - x_pos))
                            else:
                                distance_to_nearest = 0

                            edge_zone_left = self.frame_width * 0.10
                            edge_zone_right = self.frame_width * 0.90
                            zone = "LEFT" if x_pos < edge_zone_left else "RIGHT"
                            print(f"[GK DETECTION]   ID {player_detections.tracker_id[idx]}: x={x_pos:.0f}px (in {zone} edge zone), isolation={distance_to_nearest:.0f}px")

                    print(f"[GK DETECTION] Total known goalkeepers: {sorted(self.goalkeeper_tracker_ids)}")
                    print(f"[GK DETECTION] Edge zones: <{self.frame_width * 0.10:.0f}px (LEFT) | >{self.frame_width * 0.90:.0f}px (RIGHT), min_isolation=150px")
                elif frame_idx % 100 == 0:  # Only log "no GK" every 100 frames to reduce noise
                    # Show why no GK was detected
                    print(f"[GK SCAN] Frame {frame_idx}: No new goalkeepers (x range: {x_positions.min():.0f}-{x_positions.max():.0f}px)")

                self.last_gk_detection_frame = frame_idx

            # Build goalkeeper mask from stored tracker IDs
            goalkeeper_mask = np.zeros(len(player_detections), dtype=bool)
            if player_detections.tracker_id is not None:
                for i, tracker_id in enumerate(player_detections.tracker_id):
                    if tracker_id in self.goalkeeper_tracker_ids:
                        goalkeeper_mask[i] = True
            field_player_mask = ~goalkeeper_mask

            # Get field player and goalkeeper detections
            field_players = player_detections[field_player_mask]
            goalkeepers = player_detections[goalkeeper_mask]

            # Debug: Print goalkeeper detection info (only first detection)
            if not hasattr(self, '_gk_debug_printed') and len(self.goalkeeper_tracker_ids) > 0:
                print(f"\n[DEBUG] Frame {frame_idx} - First GK detection summary:")
                print(f"[DEBUG] Video frame width: {self.frame_width}px")
                print(f"[DEBUG] Total players detected: {len(player_detections)}")
                print(f"[DEBUG] Goalkeepers identified: {goalkeeper_mask.sum()}")
                print(f"[DEBUG] Field players identified: {field_player_mask.sum()}")

                if len(goalkeepers) > 0:
                    gk_positions = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    print(f"[DEBUG] Goalkeeper positions (x,y): {gk_positions}")

                    # Show all player positions for context
                    all_positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    all_x = all_positions[:, 0]
                    print(f"[DEBUG] All player x positions: min={all_x.min():.0f}, max={all_x.max():.0f}, spread={all_x.max()-all_x.min():.0f}px")

                self._gk_debug_printed = True

            # Assign teams to field players using color clustering
            if len(field_players) > 0:
                field_player_teams = self.clustering_manager.get_cluster_labels(frame, field_players)
            else:
                field_player_teams = np.array([])

            # Assign teams to goalkeepers based on proximity to field player centroids
            if len(goalkeepers) > 0 and len(field_players) > 0:
                goalkeeper_teams = self.goalkeeper_assigner.assign_goalkeeper_teams(
                    field_players, goalkeepers, field_player_teams
                )
            else:
                goalkeeper_teams = np.zeros(len(goalkeepers), dtype=int) if len(goalkeepers) > 0 else np.array([])

            # Merge team assignments back into original detection order
            team_ids = np.zeros(len(player_detections), dtype=int)
            team_ids[field_player_mask] = field_player_teams
            team_ids[goalkeeper_mask] = goalkeeper_teams

            player_detections.class_id = team_ids

            # Debug: Print goalkeeper team assignment
            if not hasattr(self, '_gk_team_debug') and len(goalkeepers) > 0:
                print(f"[DEBUG] Goalkeeper team assignments: {goalkeeper_teams}")
                print(f"[DEBUG] Field player teams: {np.unique(field_player_teams, return_counts=True)}")
                self._gk_team_debug = True

            # Add goalkeeper metadata for visualization
            if player_detections.data is None:
                player_detections.data = {}
            player_detections.data['is_goalkeeper'] = goalkeeper_mask

        assignment_time = time.time() - assignment_time

        return player_detections, assignment_time

    def _identify_goalkeepers(self, player_detections, frame_width_px=1920):
        """
        Identify goalkeepers based on their position at the FRAME EDGES.

        STRICT STRATEGY:
        - Goalkeepers are at the EXTREME left/right edges (within 10% of frame width)
        - Must be VERY isolated from other players (>15% of player spread AND >150px absolute)
        - This ensures we only detect actual goalkeepers at goal positions, not defenders

        Args:
            player_detections: All player detections
            frame_width_px: Actual video frame width in pixels (default 1920)

        Returns:
            Tuple of (goalkeeper_mask, field_player_mask) as boolean numpy arrays
        """
        if len(player_detections) < 3:
            # Need at least 3 players to distinguish GK from field players
            return np.zeros(len(player_detections), dtype=bool), np.ones(len(player_detections), dtype=bool)

        # Get bottom-center positions (feet)
        positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        x_positions = positions[:, 0]

        # Define STRICT edge zones (10% from edges - only extreme edges)
        EDGE_ZONE_PERCENT = 0.10
        left_edge_threshold = frame_width_px * EDGE_ZONE_PERCENT  # e.g., 192px for 1920px
        right_edge_threshold = frame_width_px * (1 - EDGE_ZONE_PERCENT)  # e.g., 1728px

        # Find players in edge zones
        in_left_zone = x_positions < left_edge_threshold
        in_right_zone = x_positions > right_edge_threshold

        # STRICT isolation requirement:
        # 1. Must be >15% of player spread from nearest neighbor
        # 2. Must be >150px absolute distance (to avoid detecting defenders)
        x_min, x_max = x_positions.min(), x_positions.max()
        player_spread = x_max - x_min
        relative_isolation = 0.15 * player_spread if player_spread > 100 else 100
        MIN_ABSOLUTE_ISOLATION = 150  # Minimum 150px gap required
        isolation_threshold = max(relative_isolation, MIN_ABSOLUTE_ISOLATION)

        goalkeeper_indices = []

        # Check for isolated player in LEFT zone
        if np.any(in_left_zone):
            left_zone_indices = np.where(in_left_zone)[0]
            # Take leftmost player in left zone
            leftmost_idx = left_zone_indices[np.argmin(x_positions[left_zone_indices])]

            # Check if this player is isolated from others
            other_positions = np.delete(x_positions, leftmost_idx)
            if len(other_positions) > 0:
                distance_to_nearest = np.min(np.abs(other_positions - x_positions[leftmost_idx]))
                if distance_to_nearest > isolation_threshold:
                    goalkeeper_indices.append(leftmost_idx)

        # Check for isolated player in RIGHT zone
        if np.any(in_right_zone):
            right_zone_indices = np.where(in_right_zone)[0]
            # Take rightmost player in right zone
            rightmost_idx = right_zone_indices[np.argmax(x_positions[right_zone_indices])]

            # Check if this player is isolated from others
            other_positions = np.delete(x_positions, rightmost_idx)
            if len(other_positions) > 0:
                distance_to_nearest = np.min(np.abs(other_positions - x_positions[rightmost_idx]))
                if distance_to_nearest > isolation_threshold:
                    goalkeeper_indices.append(rightmost_idx)

        # Create masks
        goalkeeper_mask = np.zeros(len(player_detections), dtype=bool)
        if len(goalkeeper_indices) > 0:
            goalkeeper_mask[goalkeeper_indices] = True
        field_player_mask = ~goalkeeper_mask

        return goalkeeper_mask, field_player_mask

    def convert_detection_to_tracks(self, player_detections, ball_detections, referee_detections, tracks, index):
        """
        Convert detection results to track format for storage.

        Args:
            player_detections: Player detection results from YOLO
            ball_detections: Ball detection results from YOLO
            referee_detections: Referee detection results from YOLO
            tracks: Existing tracks dictionary to update
            index: Frame index for track storage

        Returns:
            Updated tracks dictionary with current frame detections
        """
        # Store player tracks with tracker IDs and class IDs
        if len(player_detections.xyxy) > 0:
            # Get goalkeeper mask if available
            is_goalkeeper = None
            if hasattr(player_detections, 'data') and player_detections.data is not None:
                is_goalkeeper = player_detections.data.get('is_goalkeeper', None)

            for i, (tracker_id, bbox, class_id) in enumerate(zip(player_detections.tracker_id, player_detections.xyxy, player_detections.class_id)):
                if index not in tracks['player']:
                    tracks['player'][index] = {}
                tracks['player'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]

                # Store class IDs separately
                if index not in tracks['player_classids']:
                    tracks['player_classids'][index] = {}
                tracks['player_classids'][index][tracker_id] = class_id

                # Store goalkeeper status
                if index not in tracks.get('player_is_goalkeeper', {}):
                    if 'player_is_goalkeeper' not in tracks:
                        tracks['player_is_goalkeeper'] = {}
                    tracks['player_is_goalkeeper'][index] = {}
                if is_goalkeeper is not None:
                    tracks['player_is_goalkeeper'][index][tracker_id] = bool(is_goalkeeper[i])
                else:
                    tracks['player_is_goalkeeper'][index][tracker_id] = False
        else:
            tracks['player'][index] = {-1: [None]*4}
            tracks['player_classids'][index] = {-1: None}

        # Store ball tracks (single detection per frame)
        if len(ball_detections.xyxy) > 0:
            for bbox in ball_detections.xyxy:
                tracks['ball'][index] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        else:
            tracks['ball'][index] = [None]*4

        # Store referee tracks with sequential IDs
        if len(referee_detections.xyxy) > 0:
            for tracker_id, bbox in zip(np.arange(len(referee_detections.xyxy)), referee_detections.xyxy):
                if index not in tracks['referee']:
                    tracks['referee'][index] = {}
                tracks['referee'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        else:
            tracks['referee'][index] = {-1: [None]*4}

        return tracks

    def _interpolate_ball_tracks(self, ball_tracks_dict):
        """
        Interpolate missing ball positions using linear interpolation.

        Args:
            ball_tracks_dict: Dictionary {frame_idx: [x1, y1, x2, y2]}

        Returns:
            Dictionary with interpolated ball positions
        """
        # Convert to format expected by BallInterpolator
        # BallInterpolator expects {frame_idx: {ball_id: bbox}}
        ball_tracks_for_interpolator = {}
        for frame_idx, bbox in ball_tracks_dict.items():
            # Only include frames with valid detections (not None)
            if bbox and bbox[0] is not None:
                ball_tracks_for_interpolator[frame_idx] = {0: np.array(bbox)}

        if len(ball_tracks_for_interpolator) < 2:
            print("  ⚠️  Not enough ball detections for interpolation (need at least 2)")
            return ball_tracks_dict

        # Apply interpolation with max gap of 10 frames
        interpolated_tracks = self.ball_interpolator.interpolate_ball_positions(
            ball_tracks_for_interpolator,
            max_gap=10
        )

        # Get statistics
        stats = self.ball_interpolator.get_interpolation_stats(
            ball_tracks_for_interpolator,
            interpolated_tracks
        )

        print(f"  Ball interpolation stats:")
        print(f"    Original detections: {stats['original_detections']}")
        print(f"    After interpolation: {stats['after_interpolation']}")
        print(f"    Interpolated frames: {stats['interpolated_frames']}")
        print(f"    Interpolation rate: {stats['interpolation_rate']:.1%}")

        # Convert back to original format
        result_tracks = {}
        for frame_idx in ball_tracks_dict.keys():
            if frame_idx in interpolated_tracks:
                result_tracks[frame_idx] = interpolated_tracks[frame_idx][0].tolist()
            else:
                result_tracks[frame_idx] = ball_tracks_dict[frame_idx]

        return result_tracks

    def get_tracks(self, frames):
        """
        Process video frames and extract tracks for all objects.

        This method processes each frame through detection and tracking pipelines,
        then converts the results to a structured track format for storage.

        Args:
            frames: List of video frames to process

        Returns:
            Dictionary containing tracks with structure:
            {
                'player': {frame_idx: {tracker_id: [x1, y1, x2, y2]}},
                'ball': {frame_idx: [x1, y1, x2, y2]},
                'referee': {frame_idx: {referee_id: [x1, y1, x2, y2]}}
            }
        """
        print("Processing frames and extracting tracks...")
        tracks = {
            'player': {},
            'ball': {},
            'referee': {},
            'player_classids': {},
        }

        # Configure progress bar to update less frequently
        update_interval = max(1, len(frames) // 20)  # Update 20 times max
        for index, frame in tqdm(enumerate(frames), total=len(frames),
                                mininterval=2.0, miniters=update_interval,
                                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            # Detection pipeline - detect objects in frame
            player_detections, ball_detections, referee_detections, det_time = self.detection_callback(frame)

            # Tracking pipeline - update player tracking with ByteTrack
            player_detections = self.tracking_callback(player_detections)

            # Team assignment (clustering callback)
            if player_detections is not None:
                # DEBUG: Print index value ALWAYS for first few frames
                if index <= 4:
                    print(f">>> [BEFORE CALL] Loop index={index}, type={type(index)}", flush=True)
                player_detections, _ = self.clustering_callback(frame, player_detections, frame_idx=index)
                if index <= 4:
                    print(f">>> [AFTER CALL] Completed for index={index}", flush=True)

            # Convert detections to structured track format
            tracks = self.convert_detection_to_tracks(player_detections, ball_detections, referee_detections, tracks, index)

            # Periodic GPU memory cleanup every 100 frames
            if (index + 1) % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Apply ball interpolation to fill gaps in ball tracking
        print("Applying ball interpolation to fill gaps in ball tracking...")
        tracks['ball'] = self._interpolate_ball_tracks(tracks['ball'])

        return tracks

    def annotate_frames(self, frames, tracks):
        """
        Annotate video frames with tracking and team assignment results.

        Args:
            frames: List of video frames
            tracks: Tracking results dictionary

        Returns:
            List of annotated frames
        """
        print("Annotating frames...")
        annotated_frames = []

        # Configure progress bar to update less frequently
        update_interval = max(1, len(frames) // 20)  # Update 20 times max
        for index, frame in tqdm(enumerate(frames), total=len(frames),
                                mininterval=2.0, miniters=update_interval,
                                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            # Get tracks for this frame
            player_tracks = tracks['player'][index]
            ball_tracks = tracks['ball'][index]
            referee_tracks = tracks['referee'][index]
            player_classids = tracks.get('player_classids', {}).get(index, None)
            player_is_goalkeeper = tracks.get('player_is_goalkeeper', {}).get(index, None)

            # Clean up invalid tracks
            if -1 in player_tracks:
                player_tracks = None
                player_classids = None
                player_is_goalkeeper = None
            if -1 in referee_tracks:
                referee_tracks = None
            if (not all(ball_tracks)) or np.isnan(ball_tracks).all():
                ball_tracks = None

            # Convert to detections with stored class IDs and goalkeeper status
            player_detections, ball_detections, referee_detections = self.annotator_manager.convert_tracks_to_detections(
                player_tracks, ball_tracks, referee_tracks, player_classids, player_is_goalkeeper
            )

            # Annotate frame
            annotated_frame = self.annotator_manager.annotate_all(
                frame, player_detections, ball_detections, referee_detections
            )
            annotated_frames.append(annotated_frame)

            # Periodic memory cleanup every 100 frames
            if (index + 1) % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return annotated_frames

    def annotate_single_frame(self, frame, frame_idx, tracks):
        """
        Annotate a single frame with tracking results.

        Args:
            frame: Input video frame
            frame_idx: Frame index
            tracks: Tracking results dictionary

        Returns:
            Annotated frame
        """
        # Get tracks for this frame
        player_tracks = tracks['player'].get(frame_idx, {})
        ball_tracks = tracks['ball'].get(frame_idx, [None]*4)
        referee_tracks = tracks['referee'].get(frame_idx, {})
        player_classids = tracks.get('player_classids', {}).get(frame_idx, None)
        player_is_goalkeeper = tracks.get('player_is_goalkeeper', {}).get(frame_idx, None)

        # Clean up invalid tracks
        if -1 in player_tracks:
            player_tracks = None
            player_classids = None
            player_is_goalkeeper = None
        if -1 in referee_tracks:
            referee_tracks = None
        if (not all(ball_tracks)) or np.isnan(ball_tracks).all():
            ball_tracks = None

        # Convert to detections with stored class IDs and goalkeeper status
        player_detections, ball_detections, referee_detections = self.annotator_manager.convert_tracks_to_detections(
            player_tracks, ball_tracks, referee_tracks, player_classids, player_is_goalkeeper
        )

        # Annotate frame
        annotated_frame = self.annotator_manager.annotate_all(
            frame, player_detections, ball_detections, referee_detections
        )

        return annotated_frame

    def track_in_video(self, video_path: str, output_path: str, frame_count: int = -1):
        """Analyze a complete video with tracking and team assignment.

        Args:
            video_path: Path to input video
            output_path: Path to save tracked video
            frame_count: Number of frames to process (-1 for all frames)
            train_models: Whether to train team assignment models from the video
        """
        self.initialize_models()

        # Train team assignment models
        print("Training team assignment models...")
        self.train_team_assignment_models(video_path)

        print("Reading video frames...")
        frames = self.processing_pipeline.read_video_frames(video_path, frame_count)

        print("Extracting tracks from video...")
        tracks = self.get_tracks(frames)

        print("Interpolating ball tracks...")
        tracks = self.processing_pipeline.interpolate_ball_tracks(tracks)

        print("Annotating frames with tracking results...")
        annotated_frames = self.annotate_frames(frames, tracks)

        print("Writing tracked video...")
        self.processing_pipeline.write_video_output(annotated_frames, output_path)

        print(f"Tracking complete! Output saved to: {output_path}")
        return tracks

    def track_realtime(self, video_path: str, display_metadata: bool = True):
        """Run real-time tracking analysis on a video stream.

        Args:
            video_path: Path to input video or camera index (0 for webcam)
            train_models: Whether to train team assignment models first
            display_metadata: Whether to display tracking metadata
        """
        import cv2

        self.initialize_models()

        # Train team assignment models
        print("Training team assignment models...")
        self.train_team_assignment_models(video_path)

        print("Opening video stream...")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        print("Starting real-time tracking analysis. Press 'q' to quit.")

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Detection
            player_detections, ball_detections, referee_detections, det_time = self.detection_callback(frame)

            # Tracking
            player_detections = self.tracking_callback(player_detections)

            # Team assignment (if models are trained)
            if player_detections is not None:
                player_detections, _ = self.clustering_callback(frame, player_detections, frame_idx=frame_count)

            # Annotate frame
            annotated_frame = self.annotator_manager.annotate_all(
                frame, player_detections, ball_detections, referee_detections
            )

            # Add metadata overlay if requested
            if display_metadata:
                text_y = 30
                cv2.putText(annotated_frame, f"Frame: {frame_count}",
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Players: {len(player_detections.xyxy) if player_detections is not None else 0}",
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Ball: {len(ball_detections.xyxy) if ball_detections is not None else 0}",
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Referees: {len(referee_detections.xyxy) if referee_detections is not None else 0}",
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Detection Time: {det_time:.3f}s",
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Soccer Analysis - Real-time Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time tracking analysis stopped.")


if __name__ == "__main__":
    from player_detection.detection_constants import model_path, test_video

    # Example usage
    pipeline = TrackingPipeline(model_path)

    # Choose analysis mode (uncomment desired option)

    # Option 1: Video tracking analysis (saves to file)
    # output_path = test_video.replace('.mp4', '_tracked.mp4')
    # pipeline.track_in_video(test_video, output_path, frame_count=300, train_models=True)

    # Option 2: Real-time tracking analysis
    pipeline.track_realtime(test_video, display_metadata=True)