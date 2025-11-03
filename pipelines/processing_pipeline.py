import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import pandas as pd
from utils import read_video, write_video


class ProcessingPipeline:
    """
    Pipeline for video processing utilities like reading, writing, and ball interpolation.
    """

    def __init__(self):
        pass

    @staticmethod
    def read_video_frames(video_path, frame_count=-1):
        """
        Read video frames from a file.

        Args:
            video_path: Path to the video file
            frame_count: Number of frames to read (-1 for all frames)

        Returns:
            List of video frames
        """
        print(f"Reading video from {video_path}...")
        return read_video(video_path, frame_count=frame_count)

    @staticmethod
    def write_video_output(frames, output_path, fps=30):
        """
        Write video frames to an output file.

        Args:
            frames: List of video frames to write
            output_path: Output video file path
            fps: Frames per second for output video
        """
        print(f"Writing video to {output_path}...")
        write_video(frames, output_path, fps=fps)

    @staticmethod
    def interpolate_ball_tracks(tracks):
        """
        Interpolate ball tracks to fill in missing detections.

        Args:
            tracks: Dictionary containing tracking data

        Returns:
            Updated tracks with interpolated ball positions
        """
        print("Interpolating ball tracks...")

        # Get ball tracks
        ball_tracks = tracks['ball']

        # Convert to DataFrame for interpolation
        df = pd.DataFrame.from_dict(ball_tracks, orient='index')
        df.columns = ['x1', 'y1', 'x2', 'y2']

        # Perform linear interpolation
        df = df.interpolate(method='linear', limit_direction='both', limit=30)

        # Convert back to dictionary format
        new_tracks = {}
        for i, box in enumerate(df.to_numpy()):
            new_tracks[i] = box

        # Update original tracks
        tracks['ball'] = new_tracks
        return tracks

    @staticmethod
    def generate_output_path(input_path, suffix="_tracked"):
        """
        Generate output path based on input path with a suffix.
        Saves to output_videos folder instead of input folder.

        Args:
            input_path: Original video path
            suffix: Suffix to add before file extension

        Returns:
            Generated output path
        """
        from pathlib import Path

        input_path_obj = Path(input_path)
        # Get the project directory (assumes input_videos is in project root)
        # If input is in input_videos, go up one level to project root
        if input_path_obj.parent.name == "input_videos":
            project_dir = input_path_obj.parent.parent
        else:
            # Otherwise, assume we're already at project level or need to find it
            project_dir = input_path_obj.parent

        # Create output_videos directory if it doesn't exist
        output_dir = project_dir / "output_videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename with suffix
        output_filename = f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}"

        return str(output_dir / output_filename)