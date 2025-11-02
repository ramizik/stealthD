import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline
from constants import model_path, test_video
from keypoint_detection.keypoint_constants import keypoint_model_path
import numpy as np
import time
from tqdm import tqdm
import supervision as sv
import json
import pickle
import os


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

        # Step 3: Read video frames
        print("\n[Step 3/8] Reading video frames...")
        frames = self.processing_pipeline.read_video_frames(video_path, frame_count)
        print(f"Loaded {len(frames)} frames for processing")

        # Setup cache directory
        video_path_obj = Path(video_path)
        cache_dir = video_path_obj.parent / "cache"
        cache_dir.mkdir(exist_ok=True)  # Create cache directory if it doesn't exist

        # Check for cached frame processing data
        cache_path = cache_dir / f"{video_path_obj.stem}_frame_cache.pkl"

        # Step 4: Process all frames with detections, tracking, and team assignment
        print("\n[Step 4/8] Processing frames with detections and tracking...")

        if os.path.exists(cache_path):
            print(f"  Checking cached frame processing data: {cache_path.name}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Validate cache matches current video
                if (cached_data.get('num_frames') == len(frames) and
                    cached_data.get('frame_count') == frame_count):
                    all_tracks = cached_data['all_tracks']
                    view_transformers = cached_data['view_transformers']
                    print(f"  ✓ Cache loaded successfully - skipping frame processing")
                else:
                    print(f"  Cache mismatch (frames: {cached_data.get('num_frames')} vs {len(frames)}, "
                          f"frame_count: {cached_data.get('frame_count')} vs {frame_count}) - reprocessing...")
                    all_tracks, view_transformers = self._process_frames(frames, video_path, cache_path, frame_count)
            except Exception as e:
                print(f"  Error loading cache: {e} - reprocessing...")
                all_tracks, view_transformers = self._process_frames(frames, video_path, cache_path, frame_count)
        else:
            all_tracks, view_transformers = self._process_frames(frames, video_path, cache_path, frame_count)

        # Step 5: Ball track interpolation
        print("\n[Step 5/8] Interpolating ball tracks...")
        all_tracks = self.processing_pipeline.interpolate_ball_tracks(all_tracks)

        # Step 5.05: Estimate camera movement (but don't compensate yet)
        print("\n[Step 5.05/8] Estimating camera movement...")
        from camera_analysis import CameraMovementEstimator

        camera_estimator = CameraMovementEstimator(frames[0])
        stub_path = str(cache_dir / f"{video_path_obj.stem}_camera_movement.pkl")
        camera_movement = camera_estimator.get_camera_movement(
            frames, read_from_stub=True, stub_path=stub_path
        )
        print(f"  Camera movement estimated (will apply compensation after speed calculation)")

        # Step 5.1: Stitch fragmented tracks (re-identification)
        print("\n[Step 5.1/8] Stitching fragmented player tracks...")
        from player_tracking import TrackStitcher

        track_stitcher = TrackStitcher(max_gap_frames=150, max_position_distance=200)
        merge_map = track_stitcher.find_mergeable_tracks(all_tracks['player'], all_tracks['player_classids'])

        if merge_map:
            print(f"  Found {len(merge_map)} track fragments to merge")
            all_tracks['player'], all_tracks['player_classids'] = track_stitcher.apply_merges(
                all_tracks['player'],
                all_tracks['player_classids'],
                merge_map
            )
            print(f"  Track stitching complete: {len(merge_map)} fragments merged")
        else:
            print(f"  No track fragments found - tracking is continuous")

        # Step 5.25: Map ByteTrack IDs to consistent fixed IDs (1-22 for players, 0 for referee)
        print("\n[Step 5.25/8] Mapping player IDs to consistent format (1-22 for players, 0 for referee)...")
        from player_tracking import PlayerIDMapper

        id_mapper = PlayerIDMapper()
        id_mapper.analyze_tracks(all_tracks['player'], all_tracks['player_classids'])

        # Remap all tracks to use fixed IDs
        all_tracks['player'] = id_mapper.map_player_tracks(all_tracks['player'])
        all_tracks['player_classids'] = id_mapper.map_team_assignments(all_tracks['player_classids'])
        all_tracks['referee'] = id_mapper.map_referee_tracks(all_tracks['referee'])

        # Store mapping info for JSON output
        id_mapping_info = id_mapper.get_mapping_summary()

        print(f"  Player ID mapping complete: {len(id_mapper.bytetrack_to_fixed)} players mapped to IDs 1-22")
        print(f"  Referee mapped to ID 0")

        # Step 5.4: Stabilize team assignments using majority voting
        print("\n[Step 5.4/8] Stabilizing team assignments across entire video...")
        from player_clustering import TeamStabilizer

        team_stabilizer = TeamStabilizer(min_frames_threshold=10)
        all_tracks['player_classids'] = team_stabilizer.stabilize_teams(all_tracks['player_classids'])

        team_summary = team_stabilizer.get_team_summary()
        print(f"  Team stabilization complete:")
        print(f"    Team 0: {team_summary.get('team_0_count', 0)} players")
        print(f"    Team 1: {team_summary.get('team_1_count', 0)} players")

        # Step 5.5: Calculate analytics metrics (BEFORE camera compensation)
        # Use original pixel positions with frame-specific view transformers
        print("\n[Step 5.5/8] Calculating player speeds, possession, and passes...")
        from analytics import MetricsCalculator

        video_info = sv.VideoInfo.from_video_path(video_path)
        metrics_calculator = MetricsCalculator(fps=video_info.fps)

        # Calculate speeds using original tracks (not camera-compensated)
        # Each frame's view transformer already accounts for camera position in that frame
        metrics_data = metrics_calculator.calculate_all_metrics(
            all_tracks,
            view_transformers,
            video_info
        )

        # Step 5.6: Now apply camera compensation for track stitching and visualization
        print("\n[Step 5.6/8] Applying camera movement compensation for visualization...")
        all_tracks = camera_estimator.compensate_track_positions(
            all_tracks, camera_movement
        )
        print(f"  Camera movement compensation applied to tracks")

        # Step 6: Player Annotation
        print("\n[Step 6/8] Assigning teams and Annotating frames with detections...")
        # Restore camera-relative positions for visualization on original frames
        # (Analytics used original positions, but tracks are now compensated for storage)
        import copy
        annotation_tracks = copy.deepcopy(all_tracks)
        annotation_tracks = camera_estimator.restore_camera_positions(annotation_tracks, camera_movement)
        object_annotated_frames = self.tracking_pipeline.annotate_frames(frames, annotation_tracks)

        # Step 7: Write final output video
        output_frames = object_annotated_frames
        print("\n[Step 7/8] Writing complete analysis video...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        self.processing_pipeline.write_video_output(output_frames, output_path)

        # Summary
        total_time = time.time() - total_start_time
        print(f"\n=== Complete Soccer Analysis Finished ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Frames processed: {len(frames)}")
        print(f"Average time per frame: {total_time/len(frames):.3f}s")
        print(f"Output saved to: {output_path}")

        # Step 8: Save analysis data to JSON
        print("\n[Step 8/8] Saving analysis data to JSON...")
        # Generate JSON path properly (replace extension, not add to .mp4)
        video_path_obj = Path(video_path)
        json_output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_analysis_data.json")
        analysis_data = self._prepare_analysis_json(all_tracks, total_time, len(frames), metrics_data, id_mapping_info)

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

        llm_json_path = str(video_path_obj.parent / f"{video_path_obj.stem}_analysis_data_llm.json")
        with open(llm_json_path, 'w') as f:
            json.dump(llm_data, f, indent=2)
        print(f"LLM-ready compact data saved to: {llm_json_path}")

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

    def _process_frames(self, frames, video_path, cache_path, frame_count):
        """Process all frames with detections, tracking, and team assignment."""
        view_transformers = {}
        all_tracks = {'player': {}, 'ball': {}, 'referee': {}, 'player_classids': {}}

        # Initialize homography transformer for view transformers
        from tactical_analysis.homography import HomographyTransformer
        homography_transformer = HomographyTransformer()

        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):

            # Detect keypoints and objects
            keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
            player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)

            # Store view transformer for this frame (for metrics calculation)
            if keypoints is not None:
                try:
                    view_transformer = homography_transformer.transform_to_pitch_keypoints(keypoints)
                    view_transformers[i] = view_transformer
                except:
                    view_transformers[i] = None
            else:
                view_transformers[i] = None

            # Update with tracking
            player_detections = self.tracking_pipeline.tracking_callback(player_detections)

            # Team assignment
            player_detections, _ = self.tracking_pipeline.clustering_callback(frame, player_detections)

            # Store tracks for interpolation
            all_tracks = self.tracking_pipeline.convert_detection_to_tracks(player_detections, ball_detections, referee_detections, all_tracks, i)

        # Save cache
        print(f"  Saving frame processing cache to: {cache_path.name}")
        try:
            cached_data = {
                'all_tracks': all_tracks,
                'view_transformers': view_transformers,
                'num_frames': len(frames),
                'frame_count': frame_count  # Store frame_count to validate cache
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"  Cache saved successfully")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

        return all_tracks, view_transformers

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



if __name__ == "__main__":
    # Run Complete End-to-End Soccer Analysis Pipeline
    print("Starting Soccer Analysis...")
    pipeline = CompleteSoccerAnalysisPipeline(model_path, keypoint_model_path)
    output_video, json_output, llm_json_output = pipeline.analyze_video(str(test_video), frame_count=-1)
    print(f"\nAnalysis finished!")
    print(f"Output video: {output_video}")
    print(f"Analysis data (JSON): {json_output}")
    print(f"LLM-ready data (JSON): {llm_json_output}")