import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

import gc
import json
import os
import pickle
import time

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

from constants import GPU_DEVICE, REQUIRE_GPU, model_path, test_video
from keypoint_detection.keypoint_constants import keypoint_model_path
from pipelines import (DetectionPipeline, KeypointPipeline, ProcessingPipeline,
                       TrackingPipeline)
from utils.logger import (DualLogger, GoalkeeperLogger,
                          create_goalkeeper_log_path, create_log_path)


class CompleteSoccerAnalysisPipeline:
    """Complete end-to-end soccer analysis pipeline integrating all functionalities."""

    def __init__(self, detection_model_path: str, keypoint_model_path: str):
        """Initialize all pipeline components.

        Args:
            detection_model_path: Path to YOLO detection model
            keypoint_model_path: Path to YOLO keypoint detection model
        """
        self.detection_pipeline = DetectionPipeline(detection_model_path)
        self.keypoint_pipeline = KeypointPipeline(keypoint_model_path)
        self.tracking_pipeline = TrackingPipeline(detection_model_path)
        self.processing_pipeline = ProcessingPipeline()

        # Initialize ball tracker for anomaly filtering
        from player_tracking import BallTracker
        self.ball_tracker = BallTracker(buffer_size=10, max_distance=200)  # 200px max jump

    def initialize_models(self):
        """Initialize all models required for complete analysis."""

        print("Initializing all pipeline models...")
        start_time = time.time()

        # Initialize all pipeline models
        self.detection_pipeline.initialize_model()
        self.keypoint_pipeline.initialize_model()
        self.tracking_pipeline.initialize_models()

        init_time = time.time() - start_time
        print(f"All models initialized in {init_time:.2f}s")
        print(f"✓ Ball tracker enabled (buffer=10, max_distance=200px for anomaly filtering)")

        # Print memory status after initialization
        self._print_memory_status("After Model Initialization")

    def analyze_video(self, video_path: str, frame_count: int = -1, output_suffix: str = "_complete_analysis"):
        """Run complete end-to-end soccer analysis.

        Flow:
        1. Read video
        2. Detect keypoints and objects (players, ball, referees)
        3. Update with tracking
        4. Assign Teams
        5. Interpolate ball tracks
        6. Calculate analytics metrics
        7. Annotate frames
        8. Save Video

        Args:
            video_path: Path to input video
            frame_count: Number of frames to process (-1 for all)
            output_suffix: Suffix for output video file

        Returns:
            Path to output video
        """
        print("=== Starting Complete Soccer Analysis Pipeline ===")
        total_start_time = time.time()

        # Step 1: Initialize all models
        self.initialize_models()

        # Step 2: Train team assignment models
        print("\n[Step 2/8] Training team assignment models...")
        self.tracking_pipeline.train_team_assignment_models(video_path)

        # Step 3: Get video info (no frame loading yet!)
        print("\n[Step 3/8] Getting video information...")
        video_info = sv.VideoInfo.from_video_path(video_path)
        total_frames = video_info.total_frames if frame_count == -1 else min(frame_count, video_info.total_frames)
        print(f"Video: {total_frames} frames at {video_info.fps} fps ({video_info.width}x{video_info.height})")
        self._print_memory_status("After Video Info")

        # Setup cache directory
        video_path_obj = Path(video_path)
        cache_dir = video_path_obj.parent / "cache"
        cache_dir.mkdir(exist_ok=True)  # Create cache directory if it doesn't exist

        # Step 3.5: Detect camera shot boundaries EARLY (before frame processing)
        print("\n[Step 3.5/8] Detecting camera shot boundaries...")
        from camera_analysis import ShotDetector

        shot_stub_path = str(cache_dir / f"{video_path_obj.stem}_shot_boundaries.pkl")

        shot_boundaries = None
        shot_segments = None

        try:
            shot_detector = ShotDetector(threshold=30.0, min_scene_len=15)
            shot_boundaries = shot_detector.detect_shots(
                str(video_path),
                read_from_stub=True,
                stub_path=shot_stub_path,
                downscale_factor=2  # Faster processing
            )

            # Get shot segments for potential use
            shot_segments = shot_detector.get_shot_segments(shot_boundaries, total_frames)
            print(f"  Shot detection complete: {len(shot_boundaries)} shots, {len(shot_segments)} segments")

        except ImportError as e:
            print(f"  PySceneDetect not available - skipping shot detection")
            print(f"  Install with: pip install scenedetect[opencv]")
        except Exception as e:
            print(f"  Error during shot detection: {e}")
            print(f"  Continuing without shot boundary information")

        # Check for cached frame processing data
        cache_path = cache_dir / f"{video_path_obj.stem}_frame_cache.pkl"

        # Initialize goalkeeper debug logger
        gk_log_path = create_goalkeeper_log_path(video_path, output_suffix)
        gk_logger = GoalkeeperLogger.initialize(str(gk_log_path))
        print(f"📝 Goalkeeper debug logging to: {gk_log_path.name}")

        # Set goalkeeper logger in tracking pipeline
        self.tracking_pipeline.set_goalkeeper_logger(gk_logger)

        # Step 4: Process all frames with detections, tracking, and team assignment
        print("\n[Step 4/8] Processing frames with detections and tracking...")

        if os.path.exists(cache_path):
            print(f"  Checking cached frame processing data: {cache_path.name}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Check cache version (version 3 uses adaptive mapper)
                cache_version = cached_data.get('cache_version', 1)

                # Validate cache matches current video and is correct version
                if (cached_data.get('num_frames') == total_frames and
                    cached_data.get('frame_count') == frame_count and
                    cache_version == 3):  # Must be version 3 (adaptive mapper)
                    all_tracks = cached_data['all_tracks']
                    adaptive_mapper = cached_data['adaptive_mapper']
                    print(f"  ✓ Cache loaded successfully - skipping frame processing")
                else:
                    if cache_version < 3:
                        print(f"  Old cache format detected (version {cache_version}) - reprocessing with adaptive mapper...")
                    else:
                        print(f"  Cache mismatch (frames: {cached_data.get('num_frames')} vs {total_frames}, "
                              f"frame_count: {cached_data.get('frame_count')} vs {frame_count}) - reprocessing...")
                    all_tracks, adaptive_mapper = self._process_frames_streaming(video_path, total_frames, cache_path, frame_count, shot_boundaries)
            except Exception as e:
                print(f"  Error loading cache: {e} - reprocessing...")
                all_tracks, adaptive_mapper = self._process_frames_streaming(video_path, total_frames, cache_path, frame_count, shot_boundaries)
        else:
            all_tracks, adaptive_mapper = self._process_frames_streaming(video_path, total_frames, cache_path, frame_count, shot_boundaries)

        # Clear memory after frame processing
        self._clear_memory()
        self._print_memory_status("After Frame Processing")

        # Store shot boundaries in tracks for JSON output
        if shot_boundaries is not None:
            all_tracks['shot_boundaries'] = shot_boundaries
            all_tracks['shot_segments'] = shot_segments

        # Step 5: Ball track interpolation
        print("\n[Step 5/8] Interpolating ball tracks...")
        all_tracks = self.processing_pipeline.interpolate_ball_tracks(all_tracks)

        # Step 5.05: Estimate camera movement (but don't compensate yet)
        print("\n[Step 5.05/8] Estimating camera movement...")
        from camera_analysis import CameraMovementEstimator

        stub_path = str(cache_dir / f"{video_path_obj.stem}_camera_movement.pkl")

        # Check if camera movement cache exists
        if os.path.exists(stub_path):
            print(f"  Loading cached camera movement from: {Path(stub_path).name}")
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
            # Create estimator with first frame (load only one frame)
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            cap.release()
            if ret:
                camera_estimator = CameraMovementEstimator(first_frame)
            else:
                print(f"  WARNING: Could not read first frame")
                camera_estimator = None
        else:
            # Use STREAMING camera movement estimation (memory-efficient)
            print(f"  Estimating camera movement with streaming (memory-efficient)...")
            # Get first frame for estimator initialization
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            cap.release()

            if ret:
                camera_estimator = CameraMovementEstimator(first_frame)
                # Use streaming method instead of loading all frames
                camera_movement = camera_estimator.get_camera_movement_streaming(
                    str(video_path), total_frames, read_from_stub=True, stub_path=stub_path
                )
            else:
                print(f"  WARNING: Could not read first frame")
                camera_estimator = None
                camera_movement = None

        print(f"  Camera movement estimated (will apply compensation after speed calculation)")

        # Step 5.1: Stitching fragmented tracks (re-identification)
        print("\n[Step 5.1/8] Stitching fragmented player tracks...")
        from player_tracking import TrackStitcher

        track_stitcher = TrackStitcher(max_gap_frames=150, max_position_distance=200)
        # Pass goalkeeper metadata to prevent goalkeepers from being stitched with field players
        goalkeeper_metadata = all_tracks.get('player_is_goalkeeper', None)
        merge_map = track_stitcher.find_mergeable_tracks(
            all_tracks['player'],
            all_tracks['player_classids'],
            goalkeeper_metadata
        )

        if merge_map:
            print(f"  Found {len(merge_map)} track fragments to merge")
            all_tracks['player'], all_tracks['player_classids'] = track_stitcher.apply_merges(
                all_tracks['player'],
                all_tracks['player_classids'],
                merge_map
            )

            # Also merge goalkeeper metadata for stitched tracks
            if 'player_is_goalkeeper' in all_tracks:
                print(f"  Merging goalkeeper metadata for stitched tracks...")
                # Apply the same ID remapping to goalkeeper metadata
                merged_goalkeeper = {}
                for frame_idx, frame_goalkeepers in all_tracks['player_is_goalkeeper'].items():
                    merged_frame = {}
                    for bytetrack_id, is_gk in frame_goalkeepers.items():
                        # Use main ID if this ID was merged
                        main_id = merge_map.get(bytetrack_id, bytetrack_id)
                        # If multiple fragments merge to same main_id, keep goalkeeper=True if any was True
                        if main_id in merged_frame:
                            merged_frame[main_id] = merged_frame[main_id] or is_gk
                        else:
                            merged_frame[main_id] = is_gk
                    merged_goalkeeper[frame_idx] = merged_frame

                all_tracks['player_is_goalkeeper'] = merged_goalkeeper
                print(f"  Goalkeeper metadata remapped for track stitching")

            print(f"  Track stitching complete: {len(merge_map)} fragments merged")
        else:
            print(f"  No track fragments found - tracking is continuous")

        # Step 5.25: Map ByteTrack IDs to consistent fixed IDs (1-22 for players, 0 for referee)
        print("\n[Step 5.25/8] Mapping player IDs to consistent format (1-22 for players, 0 for referee)...")
        from player_tracking import PlayerIDMapper

        id_mapper = PlayerIDMapper()
        # Pass goalkeeper metadata to ID mapper so goalkeepers are prioritized
        goalkeeper_metadata = all_tracks.get('player_is_goalkeeper', None)
        id_mapper.analyze_tracks(all_tracks['player'], all_tracks['player_classids'], goalkeeper_metadata)

        # Remap all tracks to use fixed IDs
        all_tracks['player'] = id_mapper.map_player_tracks(all_tracks['player'])
        all_tracks['player_classids'] = id_mapper.map_team_assignments(all_tracks['player_classids'])
        all_tracks['referee'] = id_mapper.map_referee_tracks(all_tracks['referee'])

        # Remap goalkeeper metadata to use fixed IDs and propagate to all frames
        if 'player_is_goalkeeper' in all_tracks:
            # First, identify which fixed IDs are goalkeepers
            goalkeeper_fixed_ids = set()
            bytetrack_gk_mapping = {}  # Track which ByteTrack IDs mapped to which fixed IDs

            for frame_idx, frame_goalkeepers in all_tracks['player_is_goalkeeper'].items():
                for old_id, is_gk in frame_goalkeepers.items():
                    if is_gk:
                        # Map old ByteTrack ID to new fixed ID
                        # Skip if this ByteTrack ID was filtered (not in mapping)
                        if old_id in id_mapper.bytetrack_to_fixed:
                            new_id = id_mapper.bytetrack_to_fixed[old_id]
                            goalkeeper_fixed_ids.add(new_id)
                            bytetrack_gk_mapping[old_id] = new_id
                        else:
                            gk_logger.log(f"[GK MAPPING WARNING] ByteTrack GK ID {old_id} not in fixed ID mapping!")

            # Print goalkeeper ID mapping with details
            if goalkeeper_fixed_ids:
                print(f"\n[GK ID MAPPING DEBUG] Goalkeeper fixed IDs: {sorted(goalkeeper_fixed_ids)}")
                gk_logger.log(f"[GK ID MAPPING] Goalkeeper fixed IDs: {sorted(goalkeeper_fixed_ids)}")
                gk_logger.log(f"[GK ID MAPPING] ByteTrack→Fixed mapping: {bytetrack_gk_mapping}")

            # Now propagate goalkeeper status to ALL frames where these players appear
            remapped_goalkeeper = {}
            gk_frame_count = {gk_id: 0 for gk_id in goalkeeper_fixed_ids}

            for frame_idx, frame_players in all_tracks['player'].items():
                remapped_goalkeeper[frame_idx] = {}
                for player_id in frame_players.keys():
                    # Mark as goalkeeper if this player ID is a goalkeeper
                    is_gk = (player_id in goalkeeper_fixed_ids)
                    remapped_goalkeeper[frame_idx][player_id] = is_gk
                    if is_gk:
                        gk_frame_count[player_id] += 1

            all_tracks['player_is_goalkeeper'] = remapped_goalkeeper
            gk_logger.log(f"[GK PROPAGATION] Goalkeeper status propagated to all {len(remapped_goalkeeper)} frames")
            gk_logger.log(f"[GK PROPAGATION] Goalkeeper frame counts: {gk_frame_count}")

            # Debug: Show which frames have each goalkeeper
            for gk_id in sorted(goalkeeper_fixed_ids):
                frames_with_gk = [f for f, players in all_tracks['player'].items() if gk_id in players]
                if frames_with_gk:
                    gk_logger.log(f"[GK FRAMES] GK ID {gk_id} appears in {len(frames_with_gk)} frames: {frames_with_gk[:10]}{'...' if len(frames_with_gk) > 10 else ''}")
                else:
                    gk_logger.log(f"[GK WARNING] GK ID {gk_id} marked as goalkeeper but NEVER appears in player tracks!")

        # Store mapping info for JSON output
        id_mapping_info = id_mapper.get_mapping_summary()

        print(f"  Player ID mapping complete: {len(id_mapper.bytetrack_to_fixed)} players mapped to IDs 1-22")
        print(f"  Referee mapped to ID 0")

        # Step 5.4: Stabilize team assignments using majority voting
        print("\n[Step 5.4/8] Stabilizing team assignments across entire video...")
        from player_clustering import TeamStabilizer

        team_stabilizer = TeamStabilizer(min_frames_threshold=10)
        all_tracks['player_classids'] = team_stabilizer.stabilize_teams(all_tracks['player_classids'])
        all_tracks['player_classids'] = team_stabilizer.refine_with_color(
            all_tracks['player'],
            all_tracks['player_classids'],
            video_path,
            ambiguous_threshold=70.0,
            seed_threshold=85.0,
            max_samples_per_player=20
        )

        team_summary = team_stabilizer.get_team_summary()
        print(f"  Team stabilization complete:")
        print(f"    Team 0: {team_summary.get('team_0_count', 0)} players")
        print(f"    Team 1: {team_summary.get('team_1_count', 0)} players")

        # Step 5.45: Convert tracks to enhanced format and apply ViewTransformer
        print("\n[Step 5.45/8] Converting tracks to enhanced format and applying ViewTransformer...")
        from utils import convert_tracks_to_enhanced_format
        from tactical_analysis.view_transformer import ViewTransformer

        # Convert tracks to enhanced format (adds position_adjusted and bbox in track_info)
        enhanced_tracks = convert_tracks_to_enhanced_format(all_tracks)

        # Initialize ViewTransformer (using default vertices, can be customized)
        view_transformer = ViewTransformer()

        # Add transformed positions to tracks
        view_transformer.add_transformed_position_to_tracks(enhanced_tracks)

        # Merge enhanced tracks back into all_tracks (preserve existing structure)
        # Only update tracks that have enhanced data
        for object_type in ['player', 'ball', 'referee']:
            if object_type not in enhanced_tracks:
                continue

            if object_type not in all_tracks:
                all_tracks[object_type] = {}

            for frame_idx, frame_tracks in enhanced_tracks[object_type].items():
                if not isinstance(frame_idx, int):
                    continue

                # Handle ball tracks specially (original: {frame_idx: list}, enhanced: {frame_idx: {0: dict}})
                if object_type == 'ball':
                    if frame_idx in all_tracks[object_type]:
                        original_ball = all_tracks[object_type][frame_idx]
                        if isinstance(frame_tracks, dict) and 0 in frame_tracks:
                            track_info = frame_tracks[0]
                            if isinstance(track_info, dict):
                                # Convert ball to dict format
                                if isinstance(original_ball, list):
                                    all_tracks[object_type][frame_idx] = {
                                        0: {
                                            'bbox': original_ball,
                                            'position_adjusted': track_info.get('position_adjusted'),
                                            'position_transformed': track_info.get('position_transformed')
                                        }
                                    }
                                elif isinstance(original_ball, dict):
                                    # Already in dict format, update it
                                    if 0 in original_ball:
                                        if isinstance(original_ball[0], dict):
                                            original_ball[0].update({
                                                'position_adjusted': track_info.get('position_adjusted'),
                                                'position_transformed': track_info.get('position_transformed')
                                            })
                                        else:
                                            original_ball[0] = {
                                                'bbox': original_ball[0] if isinstance(original_ball[0], list) else None,
                                                'position_adjusted': track_info.get('position_adjusted'),
                                                'position_transformed': track_info.get('position_transformed')
                                            }
                    continue

                # Handle player and referee tracks
                if frame_idx not in all_tracks[object_type]:
                    continue

                if not isinstance(frame_tracks, dict):
                    continue

                for track_id, track_info in frame_tracks.items():
                    if not isinstance(track_info, dict):
                        continue

                    if track_id not in all_tracks[object_type][frame_idx]:
                        continue

                    # Update existing track with enhanced info
                    existing_track = all_tracks[object_type][frame_idx][track_id]
                    if not isinstance(existing_track, dict):
                        # Convert to dict format
                        bbox = existing_track if isinstance(existing_track, list) else None
                        all_tracks[object_type][frame_idx][track_id] = {
                            'bbox': bbox,
                            'position_adjusted': track_info.get('position_adjusted'),
                            'position_transformed': track_info.get('position_transformed')
                        }
                    else:
                        # Merge with existing dict
                        existing_track.update({
                            'position_adjusted': track_info.get('position_adjusted'),
                            'position_transformed': track_info.get('position_transformed')
                        })

        print(f"  ViewTransformer applied: added position_transformed to tracks")

        # Step 5.46: Calculate speed and distance using SpeedAndDistance_Estimator
        print("\n[Step 5.46/8] Calculating speed and distance using SpeedAndDistance_Estimator...")
        from analytics import SpeedAndDistance_Estimator

        video_info = sv.VideoInfo.from_video_path(video_path)
        speed_distance_estimator = SpeedAndDistance_Estimator(
            frame_window=5,
            frame_rate=video_info.fps
        )

        # Add speed and distance to tracks (works with position_transformed)
        speed_distance_estimator.add_speed_and_distance_to_tracks(all_tracks)

        print(f"  SpeedAndDistance_Estimator applied: added speed and distance to tracks")

        # Step 5.5: Calculate analytics metrics (BEFORE camera compensation)
        # Use original pixel positions with adaptive homography mapper
        print("\n[Step 5.5/8] Calculating player speeds, possession, and passes...")

        # Force reload of analytics modules to get latest pass detection changes
        import importlib
        import analytics
        import analytics.enhanced_pass_detector
        import analytics.improved_player_ball_assigner
        importlib.reload(analytics)
        importlib.reload(analytics.enhanced_pass_detector)
        importlib.reload(analytics.improved_player_ball_assigner)
        from analytics import MetricsCalculator

        metrics_calculator = MetricsCalculator(fps=video_info.fps)

        # Calculate speeds using original tracks (not camera-compensated)
        # Adaptive homography provides per-frame transformation
        metrics_data = metrics_calculator.calculate_all_metrics(
            all_tracks,
            adaptive_mapper,
            video_info,
            camera_movement,  # Pass camera movement for transformation
            id_mapper.bytetrack_to_fixed,  # Pass ID mapping for pass detection
            team_override=team_stabilizer.color_based_teams
        )

        # Step 5.55: Print adaptive homography statistics
        print("\n[Step 5.55/8] Adaptive homography statistics...")
        adaptive_mapper.print_statistics()

        # Step 5.6: Now apply camera compensation for track stitching and visualization
        print("\n[Step 5.6/8] Applying camera movement compensation for visualization...")
        if camera_estimator is not None and camera_movement is not None:
            all_tracks = camera_estimator.compensate_track_positions(
                all_tracks, camera_movement
            )
            print(f"  Camera movement compensation applied to tracks")
        else:
            print(f"  Skipping camera movement compensation (estimator not available)")

        # Clear memory before annotation
        if camera_estimator is not None:
            del camera_estimator
        self._clear_memory()
        self._print_memory_status("After Analytics")

        # Step 6: Player Annotation
        print("\n[Step 6/8] Assigning teams and Annotating frames with detections...")
        # Restore camera-relative positions for visualization on original frames
        # (Analytics used original positions, but tracks are now compensated for storage)
        import copy
        annotation_tracks = copy.deepcopy(all_tracks)
        if camera_movement is not None:
            from camera_analysis import CameraMovementEstimator

            # Recreate estimator for restoration (lightweight operation)
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            cap.release()
            if ret:
                temp_estimator = CameraMovementEstimator(first_frame)
                annotation_tracks = temp_estimator.restore_camera_positions(annotation_tracks, camera_movement)
                del temp_estimator

        # Step 7: Write final output video with STREAMING annotation (memory-efficient)
        print("\n[Step 7/8] Annotating and writing video with streaming (memory-efficient)...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)

        # Get video FPS
        video_info = sv.VideoInfo.from_video_path(video_path)

        # Define annotation callback for streaming
        def annotate_callback(frame, frame_idx, tracks):
            return self.tracking_pipeline.annotate_single_frame(frame, frame_idx, tracks)

        # Stream, annotate, and write frames without loading all into memory
        self.processing_pipeline.write_video_streaming(
            video_path, output_path, annotation_tracks, annotate_callback, total_frames, fps=video_info.fps
        )

        # Clear annotation tracks from memory
        del annotation_tracks
        self._clear_memory()
        self._print_memory_status("After Video Output")

        # Summary
        total_time = time.time() - total_start_time
        print(f"\n=== Complete Soccer Analysis Finished ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Frames processed: {total_frames}")
        print(f"Average time per frame: {total_time/total_frames:.3f}s")
        print(f"Output saved to: {output_path}")

        # Step 8: Save analysis data to JSON
        print("\n[Step 8/8] Saving analysis data to JSON...")
        # Generate JSON paths in same output folder as the video
        output_path_obj = Path(output_path)
        json_output_path = str(output_path_obj.parent / f"{output_path_obj.stem}_analysis_data.json")
        analysis_data = self._prepare_analysis_json(all_tracks, total_time, total_frames, metrics_data, id_mapping_info)

        # JSON conversion is now handled in _prepare_analysis_json via convert_to_serializable
        # Apply conversion to ensure all numpy types are converted before serialization
        analysis_data = self._deep_convert_numpy(analysis_data)

        try:
            with open(json_output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"Analysis data saved to: {json_output_path}")
        except (TypeError, ValueError) as e:
            # This should rarely happen now, but keep as safety net
            print(f"WARNING: JSON serialization issue detected: {e}")
            print(f"Applying additional conversion...")
            # Final attempt with even more aggressive conversion
            analysis_data_fixed = self._deep_convert_numpy(analysis_data)
            with open(json_output_path, 'w') as f:
                json.dump(analysis_data_fixed, f, indent=2)
            print(f"Analysis data saved to: {json_output_path} (after additional conversion)")

        # Create LLM-ready compact JSON
        print("Creating LLM-ready compact JSON...")
        from analytics import LLMDataFormatter

        llm_formatter = LLMDataFormatter()
        llm_data = llm_formatter.format_for_llm(analysis_data, video_path)

        # Save LLM JSON in same output folder as video
        llm_json_path = str(output_path_obj.parent / f"{output_path_obj.stem}_analysis_data_llm.json")
        with open(llm_json_path, 'w') as f:
            json.dump(llm_data, f, indent=2)
        print(f"LLM-ready compact data saved to: {llm_json_path}")

        # Close goalkeeper logger
        if gk_logger:
            gk_logger.close()

        return output_path, json_output_path, llm_json_path

    def _prepare_analysis_json(self, all_tracks, processing_time, num_frames, metrics_data=None, id_mapping_info=None):
        """
        Prepare analysis data for JSON export.

        Args:
            all_tracks: Dictionary containing all tracking data
            processing_time: Total processing time in seconds
            num_frames: Number of frames processed
            metrics_data: Dictionary containing analytics metrics (optional)
            id_mapping_info: Player ID mapping information (optional)

        Returns:
            Dictionary containing all analysis data
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_key(key):
            """Convert numpy key types to Python types."""
            # First try: Direct isinstance checks for numpy integer types
            # This catches np.int32, np.int64, etc. instances
            if isinstance(key, np.integer):
                return int(key)

            # Check specific numpy integer types
            if type(key).__name__ in ['int32', 'int64', 'int_', 'intc', 'intp',
                                       'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64']:
                if hasattr(np, type(key).__name__):
                    return int(key)

            # Check if it's a numpy type using issubdtype (most reliable)
            try:
                if np.issubdtype(type(key), np.integer):
                    return int(key)
                elif np.issubdtype(type(key), np.floating):
                    return float(key)
                elif np.issubdtype(type(key), np.bool_):
                    return bool(key)
            except (TypeError, ValueError):
                pass

            # Try item() method for numpy scalars
            if hasattr(key, 'item'):
                try:
                    return key.item()
                except (ValueError, TypeError, AttributeError):
                    pass

            # Check if it's from numpy module
            if hasattr(key, '__class__') and hasattr(key.__class__, '__module__'):
                if key.__class__.__module__ == 'numpy':
                    try:
                        # Try direct conversion
                        if hasattr(np, 'int32') and isinstance(key, (np.int32, np.int64, np.int_, np.intc)):
                            return int(key)
                        # Fallback: use issubdtype
                        if np.issubdtype(type(key), np.integer):
                            return int(key)
                        elif np.issubdtype(type(key), np.floating):
                            return float(key)
                        elif np.issubdtype(type(key), np.bool_):
                            return bool(key)
                    except (ValueError, TypeError, OverflowError):
                        pass

            # Handle tuples (convert each element recursively)
            if isinstance(key, tuple):
                return tuple(convert_key(k) for k in key)

            return key

        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python types."""
            # Handle None
            if obj is None:
                return None

            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()

            # Direct numpy scalar type checks (more robust)
            if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                               np.int16, np.int32, np.int64, np.uint8, np.uint16,
                               np.uint32, np.uint64)):
                return int(obj)

            if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)

            if isinstance(obj, np.bool_):
                return bool(obj)

            # Check if it's a numpy scalar with item() method (most reliable)
            if hasattr(obj, 'item'):
                try:
                    return obj.item()
                except (ValueError, TypeError, AttributeError):
                    pass

            # Fallback: Check if it's a numpy type using module check
            if hasattr(obj, '__class__') and hasattr(obj.__class__, '__module__'):
                if obj.__class__.__module__ == 'numpy':
                    try:
                        if np.issubdtype(type(obj), np.integer):
                            return int(obj)
                        elif np.issubdtype(type(obj), np.floating):
                            return float(obj)
                        elif np.issubdtype(type(obj), np.bool_):
                            return bool(obj)
                    except (ValueError, TypeError, OverflowError):
                        pass

            # Handle dictionaries - convert both keys and values recursively
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    converted_key = convert_key(k)
                    converted_value = convert_to_serializable(v)
                    result[converted_key] = converted_value
                return result

            # Handle lists
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]

            # Handle tuples
            if isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)

            # Handle sets (convert to list for JSON)
            if isinstance(obj, set):
                return [convert_to_serializable(item) for item in obj]

            return obj

        analysis_data = {
            "metadata": {
                "processing_time_seconds": round(processing_time, 2),
                "frames_processed": num_frames,
                "average_time_per_frame": round(processing_time / num_frames, 3) if num_frames > 0 else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": metrics_data.get('fps', 30) if metrics_data else 30,
                "field_dimensions": {"width": 105, "height": 68},
                "id_scheme": {
                    "players": "1-22 (1-11 for Team 0, 12-22 for Team 1)",
                    "referee": "0",
                    "note": "Exactly 22 players (11 per team) get fixed IDs 1-22. Track stitching maintains IDs across occlusions."
                }
            },
            "player_tracks": convert_to_serializable(all_tracks['player']),
            "ball_tracks": convert_to_serializable(all_tracks['ball']),
            "referee_tracks": convert_to_serializable(all_tracks['referee']),
            "team_assignments": convert_to_serializable(all_tracks.get('player_classids', {})),
            "statistics": self._calculate_statistics(all_tracks, num_frames)
        }

        # Add ID mapping information if available
        if id_mapping_info:
            analysis_data["id_mapping"] = convert_to_serializable(id_mapping_info)

        # Add analytics metrics if available
        if metrics_data:
            analysis_data["player_speeds"] = convert_to_serializable(metrics_data.get('player_speeds', {}))
            analysis_data["ball_possession"] = convert_to_serializable(metrics_data.get('possession_data', {}))
            analysis_data["passes"] = convert_to_serializable(metrics_data.get('passes', []))
            analysis_data["ball_assignments"] = convert_to_serializable(metrics_data.get('ball_assignments', {}))
            analysis_data["player_touches"] = convert_to_serializable(metrics_data.get('player_touches', {}))
            analysis_data["player_analytics"] = convert_to_serializable(metrics_data.get('player_analytics', {}))

        return analysis_data


    def _visualize_keypoints(self, frame: np.ndarray, keypoints: 'sv.KeyPoints',
                            frame_idx: int, output_dir: Path):
        """Visualize detected keypoints on a frame and save debug image.

        Args:
            frame: Frame to annotate
            keypoints: sv.KeyPoints object from sv.KeyPoints.from_ultralytics()
            frame_idx: Frame index
            output_dir: Directory to save visualization
        """
        from keypoint_detection.keypoint_constants import (
            KEYPOINT_CONNECTIONS, KEYPOINT_NAMES)

        if len(keypoints) == 0:
            return

        # Get keypoint coordinates and confidence from sv.KeyPoints
        kpts_xy = keypoints.xy[0]  # (32, 2) - (x, y) coordinates
        if hasattr(keypoints, 'confidence') and keypoints.confidence is not None:
            kpts_conf = keypoints.confidence[0]  # (32,) - confidence values
        else:
            kpts_conf = np.ones(len(kpts_xy))  # Default to 1.0 if no confidence

        # Combine into (32, 3) format for compatibility
        kpts = np.concatenate([kpts_xy, kpts_conf[:, np.newaxis]], axis=1)

        # Count visible keypoints
        visible_count = (kpts[:, 2] > 0.5).sum()
        avg_conf = kpts[:, 2].mean()

        # Draw keypoints
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.5:
                # Draw circle
                color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)  # Green if high conf, orange otherwise
                cv2.circle(frame, (int(x), int(y)), 6, color, -1)
                cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), 2)

                # Draw keypoint ID
                cv2.putText(frame, str(i), (int(x) + 12, int(y) - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, str(i), (int(x) + 12, int(y) - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw connections
        for conn in KEYPOINT_CONNECTIONS:
            pt1_idx, pt2_idx = conn
            if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                x1, y1, conf1 = kpts[pt1_idx]
                x2, y2, conf2 = kpts[pt2_idx]

                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            (255, 255, 0), 2)  # Cyan lines

        # Add info overlay
        info_bg = np.zeros((80, frame.shape[1], 3), dtype=np.uint8)
        info_bg[:] = (0, 0, 0)
        cv2.putText(info_bg, f"Frame {frame_idx} | Keypoints: {visible_count}/32 | Avg Conf: {avg_conf:.2f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_bg, f"Green=High Conf (>0.7) | Orange=Medium Conf (0.5-0.7)",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Combine frame and info
        annotated = np.vstack([info_bg, frame])

        # Save
        output_path = output_dir / f"frame_{frame_idx:06d}_keypoints.jpg"
        cv2.imwrite(str(output_path), annotated)

    def _process_frames_streaming(self, video_path, total_frames, cache_path, frame_count, shot_boundaries=None):
        """Process frames using streaming (generator) to avoid loading all into RAM.

        Args:
            video_path: Path to video file
            total_frames: Total number of frames to process
            cache_path: Path to save cache file
            frame_count: Frame count limit (-1 for all)
            shot_boundaries: Optional list of frame indices where camera shots change

        Returns:
            Tuple of (all_tracks, adaptive_mapper)
        """
        all_tracks = {'player': {}, 'ball': {}, 'referee': {}, 'player_classids': {}}

        # Initialize adaptive homography mapper for moving camera
        from tactical_analysis.adaptive_homography_mapper import \
            AdaptiveHomographyMapper
        adaptive_mapper = AdaptiveHomographyMapper()

        # Set shot boundaries if provided (for scene-aware homography)
        if shot_boundaries is not None and len(shot_boundaries) > 0:
            adaptive_mapper.set_shot_boundaries(shot_boundaries)

        print(f"  [Adaptive Homography] Processing {total_frames} frames with per-frame homography...")

        # Setup keypoint visualization directory
        output_dir = Path("output_videos") / "keypoint_debug"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Frames to visualize (sample evenly across video)
        viz_frames = set([0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1])

        # Use supervision's frame generator (does NOT load all frames into memory)
        frame_generator = sv.get_video_frames_generator(
            video_path,
            end=total_frames if frame_count != -1 else None
        )

        # Configure progress bar to update less frequently
        update_interval = max(1, total_frames // 20)  # Update 20 times max

        for i, frame in enumerate(tqdm(frame_generator, total=total_frames, desc="Processing frames",
                                       mininterval=2.0, miniters=update_interval,
                                       ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):

            # Detect keypoints and objects
            keypoints, metadata = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
            player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)

            # Visualize keypoints on sample frames
            if i in viz_frames and keypoints is not None and len(keypoints) > 0:
                self._visualize_keypoints(frame.copy(), keypoints, i, output_dir)

            # Filter ball detections using tracker (removes anomalies)
            ball_detections = self.ball_tracker.update(ball_detections)

            # Add frame data to adaptive mapper (sv.KeyPoints object)
            adaptive_mapper.add_frame_data(i, keypoints)

            # Update with tracking
            player_detections = self.tracking_pipeline.tracking_callback(player_detections)

            # Team assignment
            player_detections, _ = self.tracking_pipeline.clustering_callback(frame, player_detections, frame_idx=i)

            # Store tracks for interpolation
            all_tracks = self.tracking_pipeline.convert_detection_to_tracks(
                player_detections, ball_detections, referee_detections, all_tracks, i
            )

            # Periodic memory cleanup every 100 frames to prevent accumulation
            if (i + 1) % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Print statistics
        print(f"\n  [Step 4.5/8] Adaptive homography processing complete")
        adaptive_mapper.print_statistics()

        # Print keypoint visualization info
        print(f"\n  📊 Keypoint Debug Visualizations:")
        print(f"     Saved {len(viz_frames)} sample frames to: {output_dir}")
        print(f"     Check visualizations to verify pitch detection accuracy")

        # Save cache (version 3 uses adaptive mapper)
        print(f"\n  Saving frame processing cache to: {cache_path.name}")
        try:
            cached_data = {
                'all_tracks': all_tracks,
                'adaptive_mapper': adaptive_mapper,
                'num_frames': total_frames,
                'frame_count': frame_count,
                'cache_version': 3  # Version 3 uses adaptive mapper
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"  Cache saved successfully")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

        return all_tracks, adaptive_mapper

    def _calculate_statistics(self, all_tracks, num_frames):
        """
        Calculate summary statistics from tracking data.

        Args:
            all_tracks: Dictionary containing all tracking data
            num_frames: Number of frames processed

        Returns:
            Dictionary containing statistics
        """
        player_ids = set(all_tracks['player'].keys())
        ball_detections = sum(1 for frame_data in all_tracks['ball'].values() if len(frame_data) > 0)

        # Count team assignments
        team_assignments = all_tracks.get('player_classids', {})
        team_counts = {}
        for player_id, class_data in team_assignments.items():
            for frame_id, class_id in class_data.items():
                team_counts[class_id] = team_counts.get(class_id, 0) + 1

        stats = {
            "total_players_detected": len(player_ids),
            "total_frames": num_frames,
            "ball_detection_rate": round(ball_detections / num_frames, 3) if num_frames > 0 else 0,
            "team_distribution": team_counts
        }

        return stats

    def _deep_convert_numpy(self, obj):
        """
        Deep conversion of numpy types with more aggressive approach.
        Used as fallback if standard conversion fails.
        """
        if obj is None:
            return None

        # Convert numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Convert numpy scalars - try multiple methods
        if type(obj).__module__ == 'numpy':
            try:
                if hasattr(obj, 'item'):
                    return obj.item()  # Convert numpy scalar to Python type
                elif np.issubdtype(type(obj), np.integer):
                    return int(obj)
                elif np.issubdtype(type(obj), np.floating):
                    return float(obj)
                elif np.issubdtype(type(obj), np.bool_):
                    return bool(obj)
            except:
                return str(obj)  # Last resort: convert to string

        # Handle dictionaries - convert keys aggressively
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Convert key using multiple methods
                # Method 1: Check if it's a numpy integer type
                if isinstance(k, np.integer):
                    try:
                        k = int(k)
                    except:
                        pass

                # Method 2: Check numpy module
                if type(k).__module__ == 'numpy':
                    try:
                        if np.issubdtype(type(k), np.integer):
                            k = int(k)
                        elif np.issubdtype(type(k), np.floating):
                            k = float(k)
                        elif np.issubdtype(type(k), np.bool_):
                            k = bool(k)
                    except:
                        pass

                # Method 3: Try item() method for numpy scalars
                if hasattr(k, 'item'):
                    try:
                        k = k.item()
                    except:
                        pass

                # Method 4: Check type name for int32, int64, etc.
                if hasattr(type(k), '__name__'):
                    type_name = type(k).__name__
                    if type_name in ['int32', 'int64', 'int_', 'intc', 'intp',
                                     'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64']:
                        try:
                            k = int(k)
                        except:
                            pass

                # Method 5: Final fallback - ensure it's a valid JSON key type
                if not isinstance(k, (str, int, float, bool)) and k is not None:
                    # Try to convert to int if possible
                    try:
                        if np.issubdtype(type(k), np.integer):
                            k = int(k)
                        else:
                            k = str(k)
                    except:
                        k = str(k)

                result[k] = self._deep_convert_numpy(v)
            return result

        # Handle iterables
        if isinstance(obj, (list, tuple, set)):
            converted = [self._deep_convert_numpy(item) for item in obj]
            return list(converted) if isinstance(obj, (list, set)) else tuple(converted)

        return obj

    def _clear_memory(self):
        """Clear GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _print_memory_status(self, step_name: str):
        """Print current memory usage for debugging."""
        try:
            import psutil
            process = psutil.Process()
            ram_gb = process.memory_info().rss / 1024**3

            if torch.cuda.is_available():
                gpu_allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                gpu_reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  💾 [{step_name}] RAM: {ram_gb:.2f}GB | GPU Allocated: {gpu_allocated_gb:.2f}GB | GPU Reserved: {gpu_reserved_gb:.2f}GB")
            else:
                print(f"  💾 [{step_name}] RAM: {ram_gb:.2f}GB")
        except ImportError:
            # psutil not installed, skip memory reporting
            pass

    def launch_coach_chat(self, llm_json_path: str):
        """
        Launch interactive coach chat after analysis completes.

        Args:
            llm_json_path: Path to LLM-formatted JSON output
        """
        from pathlib import Path

        print("\n" + "="*70)
        print("🎯 ANALYSIS COMPLETE - LAUNCHING COACH CHAT")
        print("="*70)

        # Check API key (should be loaded from .env by now)
        if not os.environ.get('ANTHROPIC_API_KEY'):
            print("\n⚠️  ANTHROPIC_API_KEY not set")
            print("\nTo enable AI coach chat:")
            print("  1. Get API key from: https://console.anthropic.com/")
            print("  2. Add to .env file in project root:")
            print("     ANTHROPIC_API_KEY=sk-ant-...")
            print("  3. Or set environment variable:")
            print("     export ANTHROPIC_API_KEY='sk-ant-...'")
            return

        # Import and launch chat interface
        try:
            from llm_assistant.interface import CoachChatInterface

            print(f"\n🚀 Starting interactive chat with analysis from:")
            print(f"   {Path(llm_json_path).name}")
            print("\nPress Ctrl+C to exit chat and return to terminal\n")

            # Initialize and run chat interface
            chat = CoachChatInterface(llm_json_path)
            chat.run_chat_loop()

        except KeyboardInterrupt:
            print("\n\n✓ Chat session ended")
        except Exception as e:
            print(f"\n❌ Error launching chat: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nManual launch:")
            print(f"  python -m llm_assistant.interface {llm_json_path}")



if __name__ == "__main__":
    # Create log file path (before starting logger)
    log_path = create_log_path(test_video, suffix="_complete_analysis")

    # Start dual logging (console + file)
    with DualLogger(str(log_path)):
        # Check GPU availability before starting
        print("="*70)
        print("SYSTEM GPU CHECK")
        print("="*70)

        if torch.cuda.is_available():
            print(f"✓ CUDA Available: YES")
            print(f"✓ GPU Device: {torch.cuda.get_device_name(GPU_DEVICE)}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(GPU_DEVICE).total_memory / 1024**3:.2f} GB")
            print(f"✓ Using GPU {GPU_DEVICE} for all processing")
        else:
            print(f"✗ CUDA Available: NO")
            if REQUIRE_GPU:
                print("\n" + "="*70)
                print("ERROR: GPU REQUIRED BUT NOT AVAILABLE")
                print("="*70)
                print("This script requires GPU for processing.")
                print("\nInstall CUDA-enabled PyTorch:")
                print("  pip uninstall torch torchvision torchaudio")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("\nCheck NVIDIA drivers:")
                print("  nvidia-smi")
                print("\nTo allow CPU usage (VERY SLOW), set REQUIRE_GPU=False in constants.py")
                print("="*70)
                sys.exit(1)
            else:
                print("⚠️ WARNING: Running on CPU - This will be EXTREMELY slow!")

        print("="*70 + "\n")

        # Run Complete End-to-End Soccer Analysis Pipeline
        print("Starting Soccer Analysis...")
        pipeline = CompleteSoccerAnalysisPipeline(model_path, keypoint_model_path)
        output_video, json_output, llm_json_output = pipeline.analyze_video(str(test_video), frame_count=-1)
        print(f"\nAnalysis finished!")
        print(f"Output video: {output_video}")
        print(f"Analysis data: {json_output}")
        print(f"LLM-formatted data: {llm_json_output}")

        try:
            pipeline.launch_coach_chat(llm_json_output)
        except Exception as e:
            print(f"\n⚠️  Could not auto-launch chat: {e}")